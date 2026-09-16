#!/usr/bin/env python
"""
mode2_protect.py — do the firmware's protections still act in MODE 2 (open-loop PWM)?

    python bench/mode2_protect.py --port /dev/ttyACM0 --id 1            # free spin, 5 s at duty 1000
    python bench/mode2_protect.py --port /dev/ttyACM0 --id 1 --stall    # horn BLOCKED by hand/clamp

Position mode has two protections: OVERLOAD_TORQUE (load above N % for PROTECTION_TIME
cuts drive to PROTECTIVE_TORQUE %) and PROTECTION_CURRENT (a current above it for
OVERCURRENT_TIME).  A host loop in MODE 2 slams the duty to 1000 from rest and holds it
against a stalled joint, which is exactly what they exist for.  Free spin at duty 1000
exercises the load-based one (load reads 100 %); --stall the current-based one, and it
needs the horn held — a clamp, not fingers: 3 N·m on a 25 mm horn is 120 N.
Logs speed, load, current, SERVO_STATUS every 100 ms; a trip shows as the speed or the
current collapsing while the duty stays.  Duty 0, torque off, MODE 0 whatever happens.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from feetech import registers as R                         # noqa: E402
from feetech.bus import Bus, Servo                         # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--id", type=int, required=True, help="a FREE servo")
    ap.add_argument("--duty", type=int, default=1000)
    ap.add_argument("--seconds", type=float, default=5.0)
    ap.add_argument("--stall", action="store_true", help="horn blocked: current test, 3 s max")
    a = ap.parse_args()
    if a.stall:
        a.seconds = min(a.seconds, 3.0)

    bus = Bus(a.port, 1_000_000)
    if not bus.ping(a.id):
        sys.exit(f"no servo at id {a.id}")
    s = Servo(bus, a.id)
    regs = {k: bus.read(a.id, getattr(R, k)) for k in
            ("PROTECTION_CURRENT", "OVERCURRENT_TIME", "OVERLOAD_TORQUE", "PROTECTION_TIME",
             "PROTECTIVE_TORQUE", "TORQUE_LIMIT", "MAX_TORQUE")}
    print("protection registers:", regs)
    print(f"start: {s.feedback()['volt']:.1f} V, {s.feedback()['temp']} C, "
          f"status {bus.read(a.id, R.SERVO_STATUS)}")
    mode0 = bus.read(a.id, R.MODE)
    try:
        s.torque(False)
        bus.write(a.id, R.MODE, 2)
        bus.write(a.id, R.GOAL_TIME, 0)
        s.torque(True)
        print(f"\nduty {a.duty} for {a.seconds:.0f} s{' — STALLED' if a.stall else ', free'}")
        print(f"  {'t':>4} {'w':>6} {'load':>5} {'I':>5} {'status':>6} {'torque_en':>9} {'temp':>4}")
        bus.write(a.id, R.GOAL_TIME, a.duty)
        t0 = time.perf_counter()
        nxt = 0.0
        while (t := time.perf_counter() - t0) < a.seconds:
            if t >= nxt:
                fb = s.feedback()
                st = bus.read(a.id, R.SERVO_STATUS)
                te = bus.read(a.id, R.TORQUE_ENABLE)
                print(f"  {t:4.1f} {fb['w']:6.2f} {fb['load']:5d} {fb['current']:5.2f} "
                      f"{st:6d} {te:9d} {fb['temp']:4d}")
                if a.stall and fb["temp"] > 55:
                    print("  hot, stopping"); break
                nxt += 0.1
    finally:
        bus.write(a.id, R.GOAL_TIME, 0)
        time.sleep(0.3)
        s.torque(False)
        bus.write(a.id, R.MODE, mode0)
        fb = s.feedback()
        print(f"\nrestored MODE {bus.read(a.id, R.MODE)}, torque off, {fb['temp']} C, "
              f"status {bus.read(a.id, R.SERVO_STATUS)}")


if __name__ == "__main__":
    main()
