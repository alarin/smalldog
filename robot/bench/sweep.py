"""
sweep.py — drive one servo through the trajectories the fit needs, log everything.

    python bench/sweep.py --check                        # preflight, no motion
    python bench/sweep.py --traj rock                    # the 30-second question
    python bench/sweep.py --traj hold --mass 0.25 --radius 0.10 --volts 12.6
    python bench/sweep.py --traj all  --mass 0.50 --radius 0.15 --volts 11.1
    python bench/sweep.py --dry-run --traj all           # no hardware

Each trajectory is here because it separates something the others cannot. The
identification problem has a genuine degeneracy — at one supply voltage, with no
ammeter, the back-EMF damping and the viscous friction are the same column of the
regressor and no amount of data will split them — so the set below is designed
around that, not around covering the workspace:

  rock       Torque off, rock the horn by hand, watch Present Position. THE first
             measurement, and it decides a modelling question rather than fitting
             a number: does the encoder sit after the gearbox (position moves by
             the backlash) or before it (position does not move)? The vendor wiki
             implies after. Step 4's observation wiring depends on the answer.

  freeswing  Torque off, arm released from horizontal, log the decay. No motor
             torque, so this sees inertia and friction alone: the oscillation
             period gives J_m + J_load with J_load known from the arm, and the
             decay envelope separates Coulomb from viscous — Coulomb decays the
             amplitude linearly in time, viscous exponentially. This is the one
             direct measurement of `armature`, which rl/checks/check_model.py
             found to be 73x the knee link's own inertia and currently a guess.
             If the arm will not swing at all, the gearbox is effectively
             self-locking; that is a result too, and the fit falls back to
             identifying J_m from the driven runs.

  hold       Step to an angle, hold, with a known mass on a known radius. At
             steady state omega = 0, so the back-EMF term vanishes and the torque
             balance is exact: measured current against m*g*r*sin(q) gives the
             torque constant directly, and the standing position error against
             that current gives the inner loop's gain. Run it at several angles,
             several masses and SEVERAL VOLTAGES — the voltage sweep is what
             makes the electrical parameters identifiable at all.

  step       Steps of growing amplitude: the transient the inertia and the
             saturating duty show up in. Small steps stay inside the dead zone
             and show it; large ones saturate and show the torque ceiling.

  chirp      0.2 to 8 Hz. Excites the frequency range the 50 Hz policy will
             actually command, which the steps do not.

  triangle   Slow, through zero, under load. The hysteresis loop at the reversal
             is the backlash, read off directly rather than inferred.

  reversal   Dwell, reverse by a little, dwell. Stiction and the punch register,
             which decide whether small commanded corrections move the joint at
             all — the failure mode that makes a policy's fine control evaporate
             on hardware.

Every file records the control registers the servo had at the time. A fit is only
valid for those, and the robot must then run with the same ones; fit_bam.py
refuses to merge runs whose registers disagree rather than averaging two
different machines.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import runlog                                                        # noqa: E402
from feetech import registers as R                                   # noqa: E402
from feetech.bus import Bus, Servo                                   # noqa: E402


# ------------------------------------------------------------- trajectories
def traj_hold(qmax):
    """Angles held long enough for the transient to be gone. 2 s each."""
    steps = [min(qmax, max(-qmax, q))
             for q in (0.0, 0.3, 0.6, 0.9, 1.2, 0.6, -0.3, -0.6, -0.9, 0.0)]
    return ("hold", 2.0 * len(steps),
            lambda t: steps[min(len(steps) - 1, int(t / 2.0))], True)


def traj_step(qmax):
    amps = [a for a in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.2) if a <= qmax]
    dwell = 1.5

    def f(t):
        k = int(t / dwell)
        return amps[min(len(amps) - 1, k // 2)] if k % 2 else 0.0
    return ("step", dwell * 2 * len(amps), f, True)


def traj_chirp(qmax, f0=0.2, f1=8.0, T=30.0, amp=0.26):
    amp = min(amp, qmax)

    def f(t):
        k = (f1 - f0) / T                 # phase is the integral of the frequency
        return amp * math.sin(2 * math.pi * (f0 * t + 0.5 * k * t * t))
    return ("chirp", T, f, True)


def traj_triangle(qmax, amp=0.5, period=20.0, cycles=3):
    amp = min(amp, qmax)

    def f(t):
        u = (t % period) / period
        return amp * (4 * u - 1 if u < 0.5 else 3 - 4 * u)
    return ("triangle", period * cycles, f, True)


def traj_reversal(qmax, base=0.6):
    """Dwell, then reverse by an increasingly small amount. Finds the smallest
    command that actually moves the joint — dead zone, punch and stiction."""
    base = min(base, qmax * 0.8)
    d = (0.20, 0.10, 0.05, 0.02, 0.01, 0.005, 0.002)
    dwell = 1.5

    def f(t):
        k = int(t / dwell)
        return base if k % 2 == 0 else base - d[min(len(d) - 1, k // 2)]
    return ("reversal", dwell * 2 * len(d), f, True)


def traj_freeswing(qmax, start=1.2):
    """Drive to `start`, release, log the decay. Torque off after the first second."""
    return ("freeswing", 12.0, lambda t: min(start, qmax), False)


TRAJ = {"hold": traj_hold, "step": traj_step, "chirp": traj_chirp,
        "triangle": traj_triangle, "reversal": traj_reversal,
        "freeswing": traj_freeswing}
ORDER = ["freeswing", "hold", "step", "reversal", "triangle", "chirp"]


# ------------------------------------------------------------------ running
def run_one(servo, name, T, fn, torque_all, a, meta):
    """Log on a paced clock, not as fast as the bus allows.

    Two reasons, and the first one is not obvious until it bites: an unpaced loop
    runs at whatever the transport permits, which on a loopback is 80 kHz and
    four million rows for one trajectory. The second is the fit: a uniform sample
    rate means no resampling before differentiating, and no jitter masquerading
    as measurement noise. 200 Hz is already an order of magnitude above anything
    a servo whose no-load speed is 4.7 rad/s can do.
    """
    rows, released, late = [], False, 0
    period = 1.0 / a.rate
    servo.torque(True)
    servo.goal(fn(0.0))
    time.sleep(1.0)                                   # get to the start quietly
    t0 = deadline = time.perf_counter()
    try:
        while True:
            t = time.perf_counter() - t0
            if t > T:
                break
            if not torque_all and not released and t > 1.0:
                servo.torque(False)                   # the release, for freeswing
                released = True
            target = fn(t)
            if torque_all:
                servo.goal(target)
            fb = servo.feedback()
            rows.append(dict(t=t,
                             target_rad=float("nan") if released else target,
                             q_rad=fb["q"], w_rad_s=fb["w"], current_a=fb["current"],
                             volt_v=fb["volt"], temp_c=fb["temp"],
                             load_raw=fb["load"], counts=fb["counts"]))
            if fb["temp"] >= a.temp_limit:
                print(f"  !! {fb['temp']} C at the limit, stopping")
                break
            if fb["current"] >= a.current_limit:
                print(f"  !! {fb['current']:.2f} A at the limit, stopping")
                break
            if abs(fb["q"]) > a.qmax + 0.15:
                print(f"  !! {fb['q']:+.2f} rad outside the bench range, stopping")
                break
            deadline += period
            slack = deadline - time.perf_counter()
            if slack > 0:
                time.sleep(slack)
            else:
                late += 1
                deadline = time.perf_counter()
    finally:
        servo.torque(False)
    if not rows:
        print("  no samples")
        return
    rate = len(rows) / max(1e-9, rows[-1]["t"])
    if late:
        print(f"  {late} of {len(rows)} samples were late — the bus could not keep "
              f"{a.rate:g} Hz; the fit uses the timestamps, so this is a warning, "
              f"not a corruption")
    path = os.path.join(a.out, f"{name}_m{a.mass:g}_r{a.radius:g}_"
                               f"v{a.volts:g}_{time.strftime('%H%M%S')}.csv")
    runlog.write(path, dict(meta, trajectory=name, seconds=T, sample_hz=rate,
                            requested_hz=a.rate, late_samples=late,
                            samples=len(rows)), rows)
    print(f"  {len(rows)} samples at {rate:.0f} Hz -> {os.path.relpath(path)}")


def rock_test(servo, a):
    """The 30-second question, before anything else is worth doing.

    Torque off, then rock the output by hand. If Present Position moves, the
    encoder is after the gearbox and reads the true joint angle — the backlash is
    a hole in the TORQUE path, not in the measurement, and step 4 takes the
    observation from the load side. If it does not move, the encoder is before
    the gearbox, the robot cannot see the play at all, and both the observation
    wiring and the safety layer change.
    """
    servo.torque(False)
    print("\nTorque is off. Rock the horn back and forth against the play for 10 s")
    print("(gently — you are feeling for the backlash, not the end stops).")
    lo = hi = servo.feedback()["counts"]
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 10.0:
        c = servo.feedback()["counts"]
        lo, hi = min(lo, c), max(hi, c)
        print(f"\r  counts {c:5d}   seen {lo}..{hi}  "
              f"({(hi-lo)*360/R.COUNTS_PER_TURN:.2f} deg)   ", end="")
        time.sleep(0.02)
    span = (hi - lo) * 360.0 / R.COUNTS_PER_TURN
    print(f"\n\n  swing seen: {span:.2f} deg over {hi-lo} counts")
    if span > 0.15:
        print("  -> the encoder is AFTER the gearbox: it reads the true joint angle,")
        print("     and this span is the backlash (expect ~0.5 deg). Keep")
        print("     rl/actuator.py's enc_after_backlash=True.")
    else:
        print("  -> Present Position did not move. Either you did not reach the play,")
        print("     or the encoder is BEFORE the gearbox. Try harder; if it still")
        print("     will not move, set enc_after_backlash=False and tell step 4 —")
        print("     the robot then cannot observe the play at all.")
    return span


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--traj", default="all",
                    help="all, rock, or one of: " + ", ".join(ORDER))
    ap.add_argument("--mass", type=float, default=0.0, help="arm tip mass, kg")
    ap.add_argument("--radius", type=float, default=0.0, help="its radius, m")
    ap.add_argument("--arm-inertia", type=float, default=0.0,
                    help="the printed arm's own J about the axis, kg m^2, from CAD")
    ap.add_argument("--volts", type=float, default=12.0, help="PSU setting, V")
    ap.add_argument("--centre", type=int, default=2048, help="counts at q = 0")
    ap.add_argument("--qmax", type=float, default=1.4, help="bench travel limit, rad")
    ap.add_argument("--rate", type=float, default=200.0, help="logging rate, Hz")
    ap.add_argument("--temp-limit", type=float, default=60.0)
    ap.add_argument("--current-limit", type=float, default=2.5)
    ap.add_argument("--out", default=os.path.join(HERE, "data"))
    ap.add_argument("--check", action="store_true", help="preflight only, no motion")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if a.dry_run:
        from feetech.loopback import LoopbackBus
        bus = Bus(transport=LoopbackBus({a.id: {R.PRESENT_POSITION: a.centre,
                                                R.PRESENT_VOLTAGE: 120,
                                                R.PRESENT_TEMPERATURE: 30}}),
                  discard_echo=False)
        print("DRY RUN — loopback registers, no dynamics. This proves the trajectory")
        print("timing, the safety checks and the csv format run; the data is flat.\n")
    else:
        bus = Bus(a.port, a.baud)
    servo = Servo(bus, a.id, centre=a.centre)

    if not a.dry_run and not bus.ping(a.id):
        raise SystemExit(f"servo {a.id} does not answer on {a.port}")
    regs = servo.registers()
    fb = servo.feedback()
    print(f"servo {a.id}: {fb['volt']:.1f} V, {fb['temp']:.0f} C, "
          f"position {fb['counts']:.0f} counts ({fb['q']:+.3f} rad)")
    print("control registers this run is conditional on:")
    for k, v in regs.items():
        print(f"  {k:<20} {v}")
    if not a.dry_run:
        if abs(fb["volt"] - a.volts) > 0.6:
            print(f"\n!! the servo reports {fb['volt']:.1f} V but --volts says "
                  f"{a.volts:g}. The supply setting is part of the fit; fix one.")
        if regs.get("MODE") not in (0, None):
            print(f"\n!! MODE is {regs['MODE']}, not 0 (position). rl/actuator.py "
                  f"models the position loop.")
    if a.check:
        return

    meta = dict(servo_id=a.id, psu_volts=a.volts, mass_kg=a.mass, radius_m=a.radius,
                arm_inertia=a.arm_inertia, centre=a.centre, registers=regs,
                dry_run=a.dry_run, bus={"port": a.port, "baud": a.baud})

    if a.traj == "rock":
        if a.dry_run:
            print("(the rock test needs a real servo and a hand)")
            return
        rock_test(servo, a)
        return

    for name in (ORDER if a.traj == "all" else [a.traj]):
        if name not in TRAJ:
            raise SystemExit(f"unknown trajectory {name!r}")
        tname, T, fn, torque_all = TRAJ[name](a.qmax)
        print(f"\n{tname}: {T:g} s"
              + ("" if torque_all else "   (torque released after 1 s)"))
        if name == "freeswing" and not a.dry_run:
            input("  put the arm where it can swing freely, then press enter ")
        run_one(servo, tname, T, fn, torque_all, a, meta)
        if not a.dry_run:
            time.sleep(3.0)                      # let it cool between trajectories
    print("\nbus:", bus.stats())
    print("\nNow repeat at another supply voltage and another mass. One voltage is"
          "\nnot enough to identify the electrical half — rl/actuator.py says why.")


if __name__ == "__main__":
    main()
