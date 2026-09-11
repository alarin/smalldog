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
    ap.add_argument("--volts", type=float, default=12.0)
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
    span = int(round(a.span_deg * COUNTS / 360.0))
    fb = servo.feedback()
    print(f"  {fb['volt']:.1f} V, {fb['temp']:.0f} C, span {a.span_deg:.0f} deg\n")
    print(f"  {'cap':>5} {'d*U':>6} {'w_pos':>8} {'w_reg':>8} {'ratio':>7}  "
          f"(w from position | from PRESENT_SPEED)")
    rows = []
    try:
        for cap in caps:
            if cap < a.min_cap:
                continue
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
            du = cap / 1000.0 * a.volts
            rows.append((du, w_pos))
            ratio = w_pos / w_reg if w_reg else float("nan")
            print(f"  {cap:>5} {du:6.2f} {w_pos:8.3f} {w_reg:8.3f} {ratio:7.3f}")
    finally:
        servo.torque(False)

    if rows:
        du = np.array([r[0] for r in rows]); w = np.array([r[1] for r in rows])
        print(f"\n  no-load speed at {a.volts:g} V: {w[np.argmax(du)]:.3f} rad/s"
              f"   (vendor SERVO_NOLOAD_RADS = 4.71)")
        if len(rows) > 1:
            A = np.vstack([du, np.ones_like(du)]).T
            slope, icept = np.linalg.lstsq(A, w, rcond=None)[0]
            print(f"  slope d(w)/d(d*U) = {slope:.4f} rad/s/V -> k_e = "
                  f"{1.0/slope:.3f} V*s/rad   (fitted 2.03, vendor 2.55)")
            print(f"  intercept {icept:+.3f} rad/s — should be near zero; a large "
                  f"one means the slews never reached terminal speed")
    print("\n  Remember to cap TORQUE_LIMIT again before anything with a load on it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
