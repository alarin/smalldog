#!/usr/bin/env python3
"""torque_hold.py - one long, steady push, so a filtered scale can settle.

    python bench/torque_hold.py --port /dev/cu.usb... --seconds 5 --reps 2

`sweep.py --traj stall` pushes for 0.8 s and rests for 1.7, which is right for
identifying the ELECTRICAL side: bursts keep a locked rotor cool and the rest
sits at zero position error.  It is wrong for reading a SCALE.  A kitchen or
coffee scale filters hard - it is built to reject a wobbling cup - so 0.8 s is
not enough for the display to settle, and every reading taken that way is biased
LOW by however far the filter got.

So this holds instead.  The thermal argument that justified bursts does not
apply at a capped duty: at TORQUE_LIMIT 350 the motor sees d*U*I = 4.13 V x
0.36 A = 1.5 W, against the ~32 W of an uncapped locked rotor.  The abort below
is what keeps that true if the cap is ever raised.

Direction is NOT guessed.  `--push-counts` is signed and the caller passes the
direction that presses into the scale, established by lifting the arm by hand
and watching the encoder.  Getting it wrong drives the arm off the scale into
free air, so there is no default that is safe on both sides of the rig.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from feetech import registers as R                     # noqa: E402
from feetech.bus import Bus, Servo                     # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=5.0, help="length of each push")
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--rest", type=float, default=3.0, help="rest between pushes")
    ap.add_argument("--lead-in", type=float, default=2.0,
                    help="quiet time before the first push, so the operator can look")
    ap.add_argument("--push-counts", type=int, default=+326,
                    help="signed commanded overshoot; +326 counts = 0.5 rad. The SIGN "
                         "must be the direction that presses into the scale")
    ap.add_argument("--limit-max", type=int, default=500,
                    help="refuse to run if TORQUE_LIMIT is above this")
    # SUPPLY amps, like every other current limit here: PRESENT_CURRENT is d^2*U/R
    # (feetech/registers.py, CURRENT_LSB_A). At the TORQUE_LIMIT 350 this tool
    # insists on, the duty cannot exceed 0.35 and the register cannot exceed
    # 0.35^2*12/4.35 = 0.34 A, so this abort is dark by construction while the cap
    # holds — which is the point: it is the backstop for a cap that was raised, and
    # there 1.5 A of supply is d ~ 0.74 and ~2.0 A through the motor.
    ap.add_argument("--current-limit", type=float, default=1.5,
                    help="abort above this many amps of PRESENT_CURRENT, which is "
                         "SUPPLY current (default %(default)s)")
    ap.add_argument("--temp-limit", type=float, default=55.0)
    ap.add_argument("--rate", type=float, default=50.0)
    a = ap.parse_args()

    bus = Bus(a.port, a.baud)
    servo = Servo(bus, a.id)
    if not bus.ping(a.id):
        raise SystemExit(f"servo {a.id} does not answer on {a.port}")

    limit = bus.read(a.id, R.TORQUE_LIMIT)
    if limit > a.limit_max:
        raise SystemExit(f"TORQUE_LIMIT is {limit}, above --limit-max {a.limit_max}. "
                         f"Cap it first with torque_limit.py.")
    if bus.read(a.id, R.TORQUE_ENABLE):
        raise SystemExit("torque is already enabled - turn it off before holding")

    contact = servo.feedback()["counts"]
    target = contact + a.push_counts
    if not 0 <= target <= 4095:
        raise SystemExit(f"push target {target} is off the encoder range")

    print(f"  TORQUE_LIMIT {limit}   contact {contact} counts   "
          f"push to {target} ({a.push_counts:+d})")
    print(f"  schedule: {a.lead_in:.0f} s quiet, then {a.reps} x "
          f"({a.seconds:.0f} s PUSH + {a.rest:.0f} s rest)")
    for i in range(a.reps):
        t = a.lead_in + i*(a.seconds + a.rest)
        print(f"    push {i+1}: t = {t:.0f}..{t+a.seconds:.0f} s")

    dt = 1.0 / a.rate
    t0 = time.perf_counter()
    hot = 0
    try:
        time.sleep(a.lead_in)
        # Goal first, torque second — `runtime/loop.py:engage` has the argument in
        # full. `contact` is where the arm already is, read above, so nothing moves
        # when torque arrives; without this the servo drove to whatever
        # GOAL_POSITION it was last left holding, which after a `sweep.py` run is
        # the end of a trajectory and is pressed against a scale.
        bus.write(a.id, R.GOAL_POSITION, contact)
        servo.torque(True)
        for i in range(a.reps):
            for phase, goal, dur in (("push", target, a.seconds),
                                     ("rest", contact, a.rest)):
                bus.write(a.id, R.GOAL_POSITION, goal)
                samples = []
                tp = time.perf_counter()
                while time.perf_counter() - tp < dur:
                    fb = servo.feedback()
                    samples.append(fb)
                    if fb["current"] > a.current_limit or fb["temp"] > a.temp_limit:
                        hot += 1
                        if hot >= 3:
                            raise RuntimeError(
                                f"ABORT: {fb['current']:.2f} A, {fb['temp']:.0f} C")
                    else:
                        hot = 0
                    time.sleep(dt)
                if phase != "push":
                    continue
                # per-second profile: does the SERVO decay, or only the scale?
                print(f"  push {i+1} profile:")
                print(f"    {'t s':>5} {'A':>7} {'V':>6} {'counts':>7} {'temp':>5}")
                n = len(samples)
                for sec in range(int(dur)):
                    lo = int(n*sec/dur); hi = int(n*(sec+1)/dur)
                    w = samples[lo:hi]
                    if not w:
                        continue
                    print(f"    {sec:5d} {sum(x['current'] for x in w)/len(w):7.3f} "
                          f"{sum(x['volt'] for x in w)/len(w):6.2f} "
                          f"{sum(x['counts'] for x in w)/len(w):7.1f} "
                          f"{max(x['temp'] for x in w):5.0f}")
                # report the SETTLED part: the last 40 % of the hold
                tail = samples[int(len(samples)*0.6):]
                head = samples[int(len(samples)*0.1):int(len(samples)*0.3)]
                c_head = sum(s["counts"] for s in head)/len(head)
                c_tail = sum(s["counts"] for s in tail)/len(tail)
                amps = sorted(s["current"] for s in tail)
                print(f"  push {i+1}: settled current "
                      f"{amps[len(amps)//2]:.3f} A   "
                      f"volt {sum(s['volt'] for s in tail)/len(tail):.2f} V   "
                      f"temp {max(s['temp'] for s in tail):.0f} C")
                print(f"           position {c_tail-contact:+.1f} counts from contact, "
                      f"creep over the hold {c_tail-c_head:+.1f} counts "
                      f"({'steady' if abs(c_tail-c_head) < 1.5 else 'MOVING'})")
    finally:
        servo.torque(False)
        print(f"  torque off after {time.perf_counter()-t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
