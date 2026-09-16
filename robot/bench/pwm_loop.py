#!/usr/bin/env python
"""
pwm_loop.py — is there a servo under the firmware?  MODE 2 with the position loop on the host.

    python bench/pwm_loop.py --port /dev/ttyACM0 --id 22
    python bench/pwm_loop.py --port /dev/ttyACM0 --id 22 --kp 6000 --kd 80

Why: in position mode the ST3215 profiles its goal at a firmware-capped 7.7 rad/s^2
(bench/acc_register.py), which is the whole of its frequency response — 7 % of a ±15°
sine at 5 Hz.  MODE 2 drives the bridge open loop at a duty (register 44, sign bit
0x400).  If the ramp lives in the position loop, a host loop here — duty = kp*err -
kd*w, one servo, as fast as the bus goes — should step and follow sines the firmware
cannot.  Three runs, free horn:

  A. duty steps from rest: the free acceleration, tau/J.  ~100 rad/s^2 says no ramp.
  B. host P-loop steps 0.1 / 0.3 / 0.8 rad: peak speed and arrival against position
     mode's 2.8*sqrt(d) and 0.23 / 0.38 / 0.60 s.
  C. host P-loop sines ±0.26 rad at 1 / 2 / 3 / 5 Hz: gain against 0.74 / 0.25 / - / 0.07.

MODE is EEPROM; it is put back to 0 whatever happens, duty 0 and torque off first.
Runaway guard: |q| past --qmax from the start zeroes the duty and stops the run.
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


def duty_write(bus, dev_id, d):
    d = int(max(-1000, min(1000, round(d))))
    bus.write(dev_id, R.GOAL_TIME, abs(d) | (0x400 if d < 0 else 0))


class Loop:
    """One host-side position loop over one servo, logging everything."""

    def __init__(self, bus, servo, kp, kd, dmax, qmax):
        self.bus, self.s, self.kp, self.kd, self.dmax, self.qmax = bus, servo, kp, kd, dmax, qmax
        self.trip = None

    def run(self, target, T, open_loop=None):
        """Follow target(t) for T seconds; open_loop(t) instead gives the duty directly."""
        t0 = time.perf_counter()
        log = []
        while (t := time.perf_counter() - t0) < T:
            fb = self.s.feedback()
            q, w = fb["q"], fb["w"]
            if abs(q) > self.qmax:
                duty_write(self.bus, self.s.id, 0)
                self.trip = f"runaway guard: q = {q:.2f} rad"
                break
            if open_loop is not None:
                d = open_loop(t)
                tg = float("nan")
            else:
                tg = target(t)
                # the bridge's sign is opposite to the encoder's: +duty turns q negative
                d = -(self.kp * (tg - q) - self.kd * w)
            d = max(-self.dmax, min(self.dmax, d))
            duty_write(self.bus, self.s.id, d)
            log.append((t, tg, q, w, d, fb["current"]))
        duty_write(self.bus, self.s.id, 0)
        return np.array(log)


def smooth_rate(t, q, win=5):
    k = np.ones(win) / win
    qs = np.convolve(q, k, mode="valid")
    ts = t[win // 2: win // 2 + len(qs)]
    return ts, np.gradient(qs, ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--id", type=int, required=True, help="a FREE servo — this spins horns")
    ap.add_argument("--kp", type=float, default=3000.0, help="duty units per rad")
    ap.add_argument("--kd", type=float, default=60.0, help="duty units per rad/s")
    ap.add_argument("--dmax", type=int, default=800, help="duty clamp, of 1000")
    ap.add_argument("--qmax", type=float, default=1.6, help="runaway guard, rad from start")
    ap.add_argument("--amp", type=float, default=0.26, help="sine amplitude, rad")
    ap.add_argument("--freqs", type=float, nargs="+", default=[1.0, 2.0, 3.0, 5.0])
    ap.add_argument("--steps", type=float, nargs="+", default=[0.1, 0.3, 0.8])
    ap.add_argument("--skip-open", action="store_true")
    ap.add_argument("--trace", action="store_true",
                    help="A only: print the open-loop trajectory at 10 ms")
    a = ap.parse_args()

    bus = Bus(a.port, a.baud)
    if not bus.ping(a.id):
        sys.exit(f"no servo at id {a.id}")
    s = Servo(bus, a.id)
    fb = s.feedback()
    s.centre = fb["counts"]
    print(f"id {a.id}: centre {s.centre}, {fb['volt']:.1f} V, {fb['temp']} C, "
          f"MODE {bus.read(a.id, R.MODE)}, TORQUE_LIMIT {bus.read(a.id, R.TORQUE_LIMIT)}")
    mode0 = bus.read(a.id, R.MODE)
    loop = Loop(bus, s, a.kp, a.kd, a.dmax, a.qmax)
    try:
        s.torque(False)
        bus.write(a.id, R.MODE, 2)
        if bus.read(a.id, R.MODE) != 2:
            sys.exit("!! MODE would not take 2")
        duty_write(bus, a.id, 0)
        s.torque(True)

        # loop rate
        t0 = time.perf_counter()
        for _ in range(100):
            s.feedback(); duty_write(bus, a.id, 0)
        hz = 100 / (time.perf_counter() - t0)
        print(f"host loop: {hz:.0f} Hz (one feedback read + one duty write)")

        if not a.skip_open:
            print("\nA. open-loop duty steps from rest, 0.25 s, free horn")
            print(f"  {'duty':>5} {'U':>5} {'acc0':>7} {'w@0.25':>7} {'I_pk':>5}")
            for d in ((300, 500, -300, -500) if not a.trace else (500, -500)):
                s.torque(False); time.sleep(0.2)
                s.centre = s.feedback()["counts"]; s.torque(True)
                L = loop.run(None, 0.25 if not a.trace else 0.5, open_loop=lambda t, d=d: d)
                if loop.trip:
                    break
                ts, w = smooth_rate(L[:, 0], L[:, 2])
                if a.trace:
                    print(f"  duty {d}:  t_ms  q_deg  w  I")
                    last = -1
                    for i in range(len(L)):
                        if int(L[i, 0] * 100) != last:
                            last = int(L[i, 0] * 100)
                            print(f"    {L[i, 0] * 1000:5.0f} {math.degrees(L[i, 2]):6.1f} "
                                  f"{L[i, 3]:6.2f} {L[i, 5]:5.2f}")
                    continue
                m = ts < 0.08
                acc0 = np.polyfit(ts[m], w[m], 1)[0] if m.sum() > 3 else float("nan")
                print(f"  {d:>5} {abs(d) / 1000 * fb['volt']:5.1f} {acc0:7.1f} "
                      f"{w[-1]:7.2f} {L[:, 5].max():5.2f}")
                time.sleep(0.5)
            s.torque(False); time.sleep(0.3)
            s.centre = s.feedback()["counts"]; s.torque(True)

        if loop.trip:
            print("!!", loop.trip)
            return

        print(f"\nB. host loop steps, kp {a.kp:.0f} kd {a.kd:.0f} dmax {a.dmax}"
              f"   (position mode: w_peak 2.8*sqrt(d), t_arr 0.23/0.38/0.60)")
        print(f"  {'step':>5} {'w_peak':>7} {'a_fit':>6} {'t_arr':>6} {'over':>6} {'err_ss':>7} {'I_pk':>5}")
        for d in a.steps:
            res = []
            for q0, q1 in ((0.0, d), (d, 0.0)):
                loop.run(lambda t: q0, 0.6)
                L = loop.run(lambda t: q1, 1.0)
                if loop.trip:
                    break
                ts, w = smooth_rate(L[:, 0], L[:, 2])
                wpk = float(np.abs(w).max())
                arr = np.nonzero(np.abs(L[:, 2] - q1) < 0.02 * abs(d))[0]
                over = float(np.max((L[:, 2] - q1) * np.sign(q1 - q0)))
                res.append((wpk, L[arr[0], 0] if len(arr) else float("nan"), over,
                            abs(L[-10:, 2].mean() - q1), L[:, 5].max()))
            if loop.trip:
                break
            r = np.nanmean(np.array(res), axis=0)
            print(f"  {d:>5.2f} {r[0]:7.2f} {r[0] ** 2 / d:6.1f} {r[1]:6.2f} {math.degrees(r[2]):5.1f}° "
                  f"{math.degrees(r[3]):6.2f}° {r[4]:5.2f}")

        if loop.trip:
            print("!!", loop.trip)
            return

        print(f"\nC. host loop sines ±{math.degrees(a.amp):.0f}°   (position mode: 0.74 @1, 0.25 @2, 0.07 @5 Hz)")
        print(f"  {'Hz':>4} {'gain':>5} {'lag':>6} {'w_rms':>6} {'I_rms':>5} {'temp':>4}")
        for f in a.freqs:
            T = max(3.0, 6 / f)
            L = loop.run(lambda t: a.amp * math.sin(2 * math.pi * f * t), T)
            if loop.trip:
                break
            t, tg, q = L[:, 0], L[:, 1], L[:, 2]
            m = t > 1.0 / f                       # drop the first cycle
            g = math.sqrt(np.mean((q[m] - q[m].mean()) ** 2) / np.mean((tg[m] - tg[m].mean()) ** 2))
            # lag by cross-correlation
            x, y = tg[m] - tg[m].mean(), q[m] - q[m].mean()
            dt = float(np.median(np.diff(t[m])))
            nmax = max(1, int(0.5 / f / dt))
            c = [np.dot(x[max(0, -k):len(x) - max(0, k)], y[max(0, k):len(y) - max(0, -k)])
                 for k in range(0, nmax + 1)]
            lag = -int(np.argmax(c)) * dt * f * 360
            print(f"  {f:>4.1f} {g:5.2f} {lag:5.0f}° {np.sqrt(np.mean(L[m, 3] ** 2)):6.2f} "
                  f"{np.sqrt(np.mean(L[m, 5] ** 2)):5.2f} {s.feedback()['temp']:>4}")
            time.sleep(0.5)
        if loop.trip:
            print("!!", loop.trip)
    finally:
        duty_write(bus, a.id, 0)
        time.sleep(0.3)
        s.torque(False)
        bus.write(a.id, R.MODE, mode0)
        print(f"\nrestored MODE {bus.read(a.id, R.MODE)}, torque off, "
              f"{s.feedback()['temp']} C")


if __name__ == "__main__":
    main()
