#!/usr/bin/env python3
"""noload_speed.py - what the joint can ACTUALLY turn at, with nothing on it.

    python bench/noload_speed.py --port /dev/cu.usb... --duty-ladder

SERVO_NOLOAD_RADS = 4.71 rad/s (0.222 s / 60 deg) is a VENDOR number and has
never been checked, while its neighbour SERVO_STALL_NM turned out to be wrong by
53 % when a scale was finally put on it (2026-09-10).  It matters more than it
looks: 3d/../ros2/smalldog_walker/gait.py used to derive its slew limit from it,
and `(forcerange - frictionloss)/damping` - the ceiling the gait now uses - has
MJ_DAMPING in the denominator, which was itself fitted assuming this figure.

TAKE THE ARM OFF.  This runs at FULL duty by design, and the whole point is that
the output is free.  With the torque arm still bolted on you are not measuring a
no-load speed, you are swinging a 170 mm lever at 4 rad/s into whatever is in
front of it.

WHY TORQUE_LIMIT HAS TO GO BACK TO 1000
---------------------------------------
Terminal speed is d*U/k_e: the duty sets it, and TORQUE_LIMIT caps the duty.  At
the 350 the torque rig leaves behind you would measure 0.35*12/2.03 = 2.07 rad/s
and believe you had found the ceiling.  This tool refuses to run below --min-cap
for that reason.  Raising it is safe HERE and nowhere else, because a free
output has nothing to push against.

WHY THE SPEED COMES OFF THE POSITION AND NOT OFF PRESENT_SPEED
--------------------------------------------------------------
feetech/registers.py flags SPEED_LSB_COUNTS_PER_S = 1.0 as the vendor's word,
unverified.  Measuring the thing you are trying to measure with an unverified
scale factor is how the torque number went wrong for a year.  POSITION has a
known LSB - 4096 counts/turn - so the speed is differentiated from it, and
PRESENT_SPEED is then compared against that.  The ratio IS the missing LSB
check, and it costs nothing to take while the shaft is already moving.

THE DUTY LADDER IS THE REAL PRIZE
---------------------------------
One point gives the no-load speed.  A ladder over TORQUE_LIMIT gives the SLOPE
of terminal speed against duty*volts, which is 1/k_e - and k_e is still carried
as `spec` in rl/params/domain_rand.json (vendor 2.55 against a fitted 2.03).
Same shape of argument as the torque rig: the slope is the measurement, the
endpoint is not.

MEASURED 2026-09-11, 12.1 V, 21-23 C, span 120 deg, 4 slews per rung
-------------------------------------------------------------------
    cap    d*U    w_pos    w_reg   ratio
   1000  12.00    3.864    3.835   1.008
    800   9.60    3.851    3.835   1.004
    600   7.20    3.093    2.953   1.048
    400   4.80    2.077    1.994   1.041

**No-load speed 3.86 rad/s**, 18 % under the vendor's 4.71; it is
SERVO_NOLOAD_RADS now.  w_pos/w_reg within 1-5 % says SPEED_LSB_COUNTS_PER_S =
1.0 is right, which was the side question.  The main finding is the PLATEAU:
800 and 1000 give the same speed, and PRESENT_SPEED reads a flat 2500 counts/s
at both, while the two rungs below are linear through the origin at 0.43
rad/s/V (k_e = 2.32 V*s/rad, between the 2.03 fit and the 2.55 vendor).  So the
cap stops mattering somewhere around 740, and the first fit this tool printed -
"k_e = 3.92, intercept +1.08" - was two plateau points dragged through a line;
`--duty-ladder` now drops the plateau before it fits.

Two readings of the plateau, and this tool cannot tell them apart on its own:
  A. the position loop's PROFILE caps at 2500 counts/s (a round number in the
     register, GOAL_SPEED = 0 meaning "the firmware's max", not "unlimited").
     Then the motor's own ceiling is 12/2.32 = 5.2 rad/s, the stall extrapolation
     to full duty (4.50 N*m) stands, and what the rl actuator needs is a rate cap
     on the goal, not a duty cap.
  B. the position loop never applies more than ~75 % PWM.  Then the same ceiling
     applies against a block, and 4.50 N*m - a 2.2x extrapolation from rungs
     200/350/450, all under the knee - would be ~3.3.
`--pwm` decides it: MODE 2 drives the bridge open loop at a commanded duty, no
profile in the way.  Same plateau at duty 1000 -> B.  ~5 rad/s -> A.  It was
written after the adapter was unplugged and HAS NOT RUN on hardware yet.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import numpy as np                                     # noqa: E402
from feetech import registers as R                     # noqa: E402
from feetech.bus import Bus, Servo                      # noqa: E402

COUNTS = R.COUNTS_PER_TURN
TAU = 2.0 * np.pi


def one_run(bus, servo, a, span_counts, direction):
    """One saturated slew. Returns (terminal rad/s from position, from PRESENT_SPEED)."""
    start = servo.feedback()["counts"]
    goal = int(np.clip(start + direction * span_counts, 40, COUNTS - 40))
    if abs(goal - start) < span_counts * 0.5:
        return None                                     # too close to a stop
    # Goal first, torque second — `runtime/loop.py:engage` says why in full. Here it
    # is not a nicety: the slews ALTERNATE direction, so the goal left in the
    # register is the far end of the previous one, and enabling torque before
    # writing the new goal flung a free hub back the other way at full duty before
    # the measurement had started. Parking it on `start` means nothing moves until
    # the goal below is written, which is also what makes the slew start from rest.
    bus.write(a.id, R.GOAL_POSITION, start)
    servo.torque(True)
    bus.write(a.id, R.GOAL_POSITION, goal)
    t0 = time.perf_counter()
    ts, cs, ws = [], [], []
    while time.perf_counter() - t0 < a.window:
        fb = servo.feedback()
        ts.append(time.perf_counter() - t0)
        cs.append(fb["counts"])
        ws.append(fb["w"])
        if len(cs) > 4 and abs(fb["counts"] - goal) < 8:
            break
    servo.torque(False)
    t, c, w = np.asarray(ts), np.asarray(cs, float), np.asarray(ws)
    if len(t) < 8:
        return None
    # speed from POSITION, which has a known LSB
    v = np.gradient(c, t) * TAU / COUNTS                 # rad/s
    # the plateau: the fastest sustained stretch, not a single differentiated spike
    k = max(3, len(v) // 12)
    sm = np.convolve(np.abs(v), np.ones(k) / k, mode="valid")
    i = int(np.argmax(sm))
    plateau = float(sm[i])
    w_reg = float(np.median(np.abs(w[i:i + k]))) if len(w) >= i + k else float("nan")
    return plateau, w_reg, abs(goal - start) * 360.0 / COUNTS


def pwm_spin(bus, servo, a, duty):
    """One open-loop burst in MODE 2 at a signed duty (register 44, sign bit 0x400).
    The hub turns continuously, so the position is unwrapped, not clipped."""
    bus.write(a.id, R.GOAL_TIME, abs(duty) | (0x400 if duty < 0 else 0))
    t0 = time.perf_counter()
    ts, cs, ws = [], [], []
    while time.perf_counter() - t0 < a.window:
        fb = servo.feedback()
        ts.append(time.perf_counter() - t0)
        cs.append(fb["counts"])
        ws.append(fb["w"])
    bus.write(a.id, R.GOAL_TIME, 0)
    t = np.asarray(ts)
    c = np.unwrap(np.asarray(cs, float) * TAU / COUNTS)
    v = np.gradient(c, t)
    k = max(3, len(v) // 4)
    sm = np.convolve(np.abs(v), np.ones(k) / k, mode="valid")
    i = int(np.argmax(sm))
    return float(sm[i]), float(np.median(np.abs(np.asarray(ws)[i:i + k])))


def pwm_ladder(bus, servo, a, volts):
    """The discriminating run: the same duty rungs with no position loop between
    the duty and the bridge.  Restores MODE 0 and the TORQUE_LIMIT whatever happens.

    The cap matters as much as the mode. This raises TORQUE_LIMIT to 1000 because a
    capped duty caps the speed and the answer would be the cap — and the servo it is
    run on is the one the torque rig leaves at 350. Putting it back is not tidiness:
    the next thing to touch this servo is a rig with a 170 mm arm on it, and 1000 is
    the difference between a push and a throw.
    """
    print("  MODE 2 (open-loop PWM).  The hub will turn CONTINUOUSLY.\n")
    print(f"  {'duty':>5} {'d*U':>6} {'w_pos':>8} {'w_reg':>8}")
    rows = []
    cap0 = bus.read(a.id, R.TORQUE_LIMIT)
    try:
        servo.torque(False)
        bus.write(a.id, R.TORQUE_LIMIT, 1000)
        bus.write(a.id, R.MODE, 2)
        if bus.read(a.id, R.MODE) != 2:
            print("  !! MODE would not take 2")
            return rows
        servo.torque(True)
        for d in (400, 600, 800, 1000, -1000, -800):
            w, wr = pwm_spin(bus, servo, a, d)
            time.sleep(0.3)
            print(f"  {d:>5} {abs(d) / 1000 * volts:6.2f} {w:8.3f} {wr:8.3f}")
            rows.append((abs(d) / 1000 * volts, w))
    finally:
        bus.write(a.id, R.GOAL_TIME, 0)
        time.sleep(0.3)
        servo.torque(False)
        bus.write(a.id, R.MODE, 0)
        bus.write(a.id, R.TORQUE_LIMIT, cap0)
        print(f"\n  restored MODE = {bus.read(a.id, R.MODE)}, "
              f"TORQUE_LIMIT = {bus.read(a.id, R.TORQUE_LIMIT)} (was {cap0})")
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--span-deg", type=float, default=120.0,
                    help="how far each slew travels. Needs to be long enough to "
                         "spend most of it at terminal speed: the shaft reaches "
                         "terminal in ~0.02 s and ~2 deg")
    ap.add_argument("--reps", type=int, default=4, help="slews per duty, alternating")
    ap.add_argument("--window", type=float, default=1.5, help="s, per slew")
    ap.add_argument("--min-cap", type=int, default=900,
                    help="refuse to run below this TORQUE_LIMIT — a capped duty "
                         "caps the speed and the number would be meaningless")
    ap.add_argument("--duty-ladder", action="store_true",
                    help="sweep TORQUE_LIMIT and fit 1/k_e from the slope")
    ap.add_argument("--pwm", action="store_true",
                    help="MODE 2, open loop: the same rungs with no position loop "
                         "between the duty and the bridge. Decides whether the "
                         "plateau is the profile's (A) or the PWM's (B). UNTESTED "
                         "on hardware as of 2026-09-11.")
    ap.add_argument("--volts", type=float, default=12.0,
                    help="the PSU's dial, as a CROSS-CHECK. Every d*U below is "
                         "computed from what the servo actually reports")
    a = ap.parse_args()

    bus = Bus(a.port, a.baud)
    servo = Servo(bus, a.id)
    if not bus.ping(a.id):
        raise SystemExit(f"servo {a.id} does not answer on {a.port}")

    print("\n  !! THE ARM MUST BE OFF THE HUB. This runs at full duty on a free "
          "output.\n")
    if bus.read(a.id, R.GOAL_SPEED) != 0:
        print(f"  !! GOAL_SPEED is {bus.read(a.id, R.GOAL_SPEED)}, not 0. That is a "
              f"commanded speed cap and it will BE the answer. Set it to 0.")
        return 2
    if bus.read(a.id, R.MODE) != 0:
        print("  !! MODE is not 0 (position).")
        return 2

    caps = [1000, 800, 600, 400] if a.duty_ladder else [1000]
    if caps[0] < a.min_cap:
        # the guard is against measuring a capped duty and calling it the
        # ceiling.  It applies to the rung that CLAIMS to be the ceiling, not to
        # the ladder below it - a --min-cap of 900 used to silently reduce
        # --duty-ladder to its top rung.
        raise SystemExit(f"top rung {caps[0]} is under --min-cap {a.min_cap}")
    span = int(round(a.span_deg * COUNTS / 360.0))
    fb = servo.feedback()
    print(f"  {fb['volt']:.1f} V, {fb['temp']:.0f} C, span {a.span_deg:.0f} deg\n")
    # d*U is the x-axis of the whole ladder and the slope through it IS 1/k_e, so it
    # is taken from the supply the servo REPORTS, not from the dial the operator
    # typed. Reading the voltage and then regressing against a nominal 12.0 is how a
    # bench at 12.4 V reports k_e 3 % low and nobody can see it in the output.
    volts = fb["volt"]
    if abs(volts - a.volts) > 0.3:
        print(f"  !! the servo reads {volts:.1f} V against --volts {a.volts:g}. The "
              f"ladder below uses {volts:.1f}; check the supply if that is wrong.\n")
    if a.pwm:
        rows = pwm_ladder(bus, servo, a, volts)
        report(rows, volts, pwm=True)
        print("\n  Remember to cap TORQUE_LIMIT again before anything with a load "
              "on it.")
        return 0
    print(f"  {'cap':>5} {'d*U':>6} {'w_pos':>8} {'w_reg':>8} {'ratio':>7}  "
          f"(w from position | from PRESENT_SPEED)")
    rows = []
    try:
        for cap in caps:
            bus.write(a.id, R.TORQUE_LIMIT, cap)
            if bus.read(a.id, R.TORQUE_LIMIT) != cap:
                print(f"  !! TORQUE_LIMIT would not take {cap}")
                continue
            vs, rs = [], []
            for i in range(a.reps):
                r = one_run(bus, servo, a, span, +1 if i % 2 == 0 else -1)
                if r:
                    vs.append(r[0]); rs.append(r[1])
                time.sleep(0.25)
            if not vs:
                print(f"  {cap:>5}  no usable slew — too close to an end stop?")
                continue
            w_pos, w_reg = float(np.median(vs)), float(np.median(rs))
            du = cap / 1000.0 * volts
            rows.append((du, w_pos))
            ratio = w_pos / w_reg if w_reg else float("nan")
            print(f"  {cap:>5} {du:6.2f} {w_pos:8.3f} {w_reg:8.3f} {ratio:7.3f}")
    finally:
        servo.torque(False)

    report(rows, volts)
    print("\n  Remember to cap TORQUE_LIMIT again before anything with a load on it.")
    return 0


def report(rows, volts, pwm=False):
    if not rows:
        return
    rows = sorted(rows)
    du = np.array([r[0] for r in rows]); w = np.array([r[1] for r in rows])
    top = float(w[-1])
    print(f"\n  {'open-loop' if pwm else 'no-load'} speed at {volts:.1f} V: "
          f"{top:.3f} rad/s   (measured 3.86 in position mode; vendor 4.71)")
    if len(rows) < 2:
        return
    # the plateau: rungs from the top down that agree within 3 %.  Fitting a
    # line through those is how this printed k_e = 3.92 the first time.
    n = len(rows)
    while n > 1 and w[n - 2] > 0.97 * w[n - 1]:
        n -= 1
    if n < len(rows):
        print(f"  plateau: {len(rows) - n + 1} rungs from d*U = {du[n - 1]:.2f} V up "
              f"all read {top:.2f} rad/s - the duty stopped mattering there, "
              f"{'so it is NOT the position loop (B)' if pwm else 'A or B, see --pwm'}")
    # rows[n-1] is the lowest plateau rung, so the slope is fitted over everything
    # BELOW it. With no plateau there is no rung to drop, and dropping one anyway
    # threw away the top of the ladder — the point with the most leverage on a
    # through-the-origin slope — on exactly the runs where every rung is honest.
    cut = n - 1 if n < len(rows) else n
    lo_du, lo_w = du[:cut], w[:cut]
    n = len(lo_du)
    if n >= 1:
        # through the origin: a free output at duty d sits at d*U/k_e, nothing
        # to intercept.  An intercept fit needs >=3 unsaturated rungs.
        slope = float(np.dot(lo_du, lo_w) / np.dot(lo_du, lo_du))
        print(f"  slope through origin over {n} unsaturated rung(s): "
              f"{slope:.4f} rad/s/V -> k_e = {1.0 / slope:.3f} V*s/rad   "
              f"(2.32 on 2026-09-11; fitted 2.03, vendor 2.55)")
        if n >= 3:
            A = np.vstack([lo_du, np.ones_like(lo_du)]).T
            sl, ic = np.linalg.lstsq(A, lo_w, rcond=None)[0]
            print(f"  with intercept: {sl:.4f} rad/s/V, {ic:+.3f} rad/s - a large "
                  f"intercept means the slews never reached terminal speed")


if __name__ == "__main__":
    sys.exit(main())
