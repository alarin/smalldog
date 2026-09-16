#!/usr/bin/env python
"""
acc_register.py — does the ACCELERATION register (41) move the goal profile?

    python bench/acc_register.py --port /dev/ttyACM0 --id 22
    python bench/acc_register.py --port /dev/ttyACM0 --id 22 --acc 0 254 --amps 0.3

Why: the ST3215 at ACCELERATION 0 profiles its goal at 8 rad/s^2 (a step's peak speed
is 2.8*sqrt(step), bench/chirp_gain.py), which is the whole of the servo's frequency
response and the reason a 5 Hz RL gait rocked in place. Two vendor-side documents
disagree on what 0 means — "instant" (community register reference) against "the
smaller, the slower; max 150" (Waveshare wiki) — and neither reports a floor. This
steps the servo at several register values and prints the fitted a = w_peak^2 / d
for each. If a follows the register (unit said to be 100 counts/s^2 = 0.153 rad/s^2),
the 8 is a setting; if it stays at 8, it is firmware.

The register is volatile SRAM; it is restored to what it was on exit either way.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from feetech import registers as R                         # noqa: E402
from feetech.bus import Bus, Servo                         # noqa: E402

RAD_PER_COUNT = 2 * math.pi / R.COUNTS_PER_TURN
ACC_LSB = 100 * RAD_PER_COUNT          # the documented unit, rad/s^2


def step(servo: Servo, q0: float, q1: float, dwell: float, rate: float):
    """Command q0 -> q1, log (t, q) at `rate` for `dwell` seconds."""
    servo.goal(q0)
    time.sleep(dwell)
    t0 = time.perf_counter()
    servo.goal(q1)
    ts, qs = [], []
    while (t := time.perf_counter() - t0) < dwell:
        qs.append(servo.feedback()["q"])
        ts.append(t)
        time.sleep(max(0.0, 1.0 / rate - 0.002))
    return np.array(ts), np.array(qs)


def peak_speed(t, q, win=5):
    """Peak |dq/dt| after a `win`-sample moving mean (the encoder is 12-bit)."""
    if len(q) < 2 * win + 2:
        return float("nan"), float("nan")
    k = np.ones(win) / win
    qs = np.convolve(q, k, mode="valid")
    ts = t[win // 2: win // 2 + len(qs)]
    w = np.gradient(qs, ts)
    i = int(np.argmax(np.abs(w)))
    # the time the goal takes to arrive within 2 % of the step
    d = q[-1] - q[0]
    arrived = np.nonzero(np.abs(q - q[-1]) < 0.02 * abs(d) + 2 * RAD_PER_COUNT)[0]
    t_arr = t[arrived[0]] if len(arrived) else float("nan")
    return float(abs(w[i])), t_arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--id", type=int, default=22)
    ap.add_argument("--acc", type=int, nargs="+", default=[0, 10, 50, 150, 254])
    ap.add_argument("--amps", type=float, nargs="+", default=[0.1, 0.3, 0.8],
                    help="step sizes, rad")
    ap.add_argument("--centre", type=int, default=None,
                    help="counts at q = 0; default: where the servo is now")
    ap.add_argument("--dwell", type=float, default=1.5)
    ap.add_argument("--rate", type=float, default=200.0)
    a = ap.parse_args()

    bus = Bus(a.port, a.baud)
    if not bus.ping(a.id):
        sys.exit(f"no servo at id {a.id} on {a.port}")
    servo = Servo(bus, a.id)
    if a.centre is None:
        servo.centre = servo.feedback()["counts"]
    print(f"id {a.id}: centre {servo.centre} counts, "
          f"volt {servo.feedback()['volt']:.1f}, temp {servo.feedback()['temp']}")
    regs = servo.registers()
    print("registers:", {k: regs[k] for k in ("P_COEF", "D_COEF", "I_COEF", "ACCELERATION",
                                              "GOAL_SPEED", "MODE", "TORQUE_LIMIT")})
    acc0 = regs["ACCELERATION"]
    servo.torque(True)
    try:
        print(f"\n{'ACC':>4} {'unit*ACC':>9} {'step':>6} {'w_peak':>7} {'a_fit':>7} "
              f"{'t_arr':>6}   law 2.8*sqrt(d)")
        for acc in a.acc:
            bus.write(a.id, R.ACCELERATION, acc)
            got = bus.read(a.id, R.ACCELERATION)
            if got != acc:
                print(f"!! ACCELERATION wrote {acc}, reads {got}")
            for amp in a.amps:
                # out and back, both directions, take the mean
                res = []
                for q0, q1 in ((0.0, amp), (amp, 0.0)):
                    t, q = step(servo, q0, q1, a.dwell, a.rate)
                    res.append(peak_speed(t, q))
                w = float(np.mean([r[0] for r in res]))
                t_arr = float(np.nanmean([r[1] for r in res]))
                a_fit = w * w / amp
                print(f"{acc:>4} {acc * ACC_LSB:>9.2f} {amp:>6.2f} {w:>7.2f} {a_fit:>7.1f} "
                      f"{t_arr:>6.2f}   {2.8 * math.sqrt(amp):.2f}")
            sys.stdout.flush()
    finally:
        servo.goal(0.0)
        time.sleep(a.dwell)
        bus.write(a.id, R.ACCELERATION, acc0 if acc0 is not None else 0)
        servo.torque(False)
        print(f"\nACCELERATION restored to {acc0}, torque off")


if __name__ == "__main__":
    main()
