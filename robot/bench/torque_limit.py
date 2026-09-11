#!/usr/bin/env python3
"""torque_limit.py — cap what the servo may push with, and read the cap back.

    python bench/torque_limit.py --port /dev/cu.usbmodem... --show
    python bench/torque_limit.py --port /dev/cu.usbmodem... --set 350

`sweep.py` stamps TORQUE_LIMIT into every csv but has never been able to SET it,
so the cap that `3d/torque_rig.py`'s protocol step 0 asks for had no tool. This
is that tool and nothing else: one register, written, then read back off the
servo rather than assumed.

TORQUE_LIMIT IS VOLATILE
------------------------
Register 48 is in SRAM, not EPROM (`feetech/registers.py` draws the line at 40).
The servo reloads it from MAX_TORQUE — register 16, persistent — on every
power-up, so a cap set now is gone the next time the supply is switched off, and
what comes back is whatever MAX_TORQUE says, normally 1000. Set it AFTER
power-up and BEFORE torque is enabled, every session. That is why this prints
MAX_TORQUE beside it: that number is what the cap decays to.

The other three ceilings
------------------------
TORQUE_LIMIT is not the only thing that can stop a push, and if one of the
others trips first you have measured the protection rather than the servo —
`3d/torque_rig.py`, "four ways to get a wrong number". PROTECTION_CURRENT,
OVERLOAD_TORQUE and PROTECTIVE_TORQUE are printed here for the same reason.
"""
from __future__ import annotations

import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from feetech import registers as R            # noqa: E402
from feetech.bus import Bus, Servo            # noqa: E402

#: What the ladder in 3d/torque_rig.py prints for the worst open candidate.
#: Not a default — the operator passes the number — but it is the one this
#: repository has a documented reason for, so --set says so when it differs.
RIG_CAP = 350

CEILINGS = ["TORQUE_LIMIT", "MAX_TORQUE", "PROTECTION_CURRENT",
            "OVERLOAD_TORQUE", "PROTECTIVE_TORQUE"]


def show(servo, note=""):
    vals = {}
    for name in CEILINGS:
        vals[name] = servo.bus.read(servo.id, getattr(R, name))
    print(f"\n  servo {servo.id}{note}")
    for name in CEILINGS:
        tag = "  <- volatile, reloads from MAX_TORQUE at power-up" \
              if name == "TORQUE_LIMIT" else ""
        print(f"    {name:<20}{vals[name]:>6}{tag}")
    return vals


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--set", type=int, default=None, metavar="N",
                    help=f"write TORQUE_LIMIT, 0..1000 (the torque rig asks for "
                         f"{RIG_CAP})")
    ap.add_argument("--show", action="store_true", help="read only, write nothing")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        return selftest()
    if a.set is None and not a.show:
        ap.error("say --show or --set N; this tool never picks a number for you")
    if a.set is not None and not 0 <= a.set <= 1000:
        ap.error("--set takes 0..1000 (the register's own units, not N*m)")

    bus = Bus(a.port, a.baud)
    servo = Servo(bus, a.id)
    if not bus.ping(a.id):
        raise SystemExit(f"servo {a.id} does not answer on {a.port}")

    show(servo, "  — before")            # for the printing; nothing reads it back
    if a.set is None:
        return 0

    if bus.read(a.id, R.TORQUE_ENABLE):
        print("\n  !! torque is ENABLED. Turn it off before changing the cap —")
        print("     a limit written under load is a step change in what the")
        print("     output is allowed to push.")
        return 2

    bus.write(a.id, R.TORQUE_LIMIT, a.set)
    after = show(servo, "  — after")

    got = after["TORQUE_LIMIT"]
    if got != a.set:
        print(f"\n  !! wrote {a.set}, read back {got}. The cap is NOT set. Do not push.")
        return 1
    print(f"\n  TORQUE_LIMIT = {got}, confirmed by read-back"
          f"{'' if a.set == RIG_CAP else f'  (the torque rig asks for {RIG_CAP})'}")
    if after["MAX_TORQUE"] > got:
        print(f"  Remember: power-cycling reloads {after['MAX_TORQUE']} from "
              f"MAX_TORQUE. Re-run this after every power-up.")
    return 0


def selftest() -> int:
    from feetech.loopback import LoopbackBus
    ok = True
    lb = LoopbackBus({1: {R.MAX_TORQUE: 1000, R.TORQUE_LIMIT: 1000,
                          R.TORQUE_ENABLE: 0}})
    bus = Bus(transport=lb, discard_echo=False)
    servo = Servo(bus, 1)

    bus.write(1, R.TORQUE_LIMIT, RIG_CAP)
    got = bus.read(1, R.TORQUE_LIMIT)
    ok &= got == RIG_CAP
    print(f"  write/read-back {RIG_CAP:>5} -> {got:<5} "
          f"{'ok' if got == RIG_CAP else '!! FAIL'}")

    # the two-byte path is the one that can silently truncate
    for v in (0, 1, 255, 256, 1000):
        bus.write(1, R.TORQUE_LIMIT, v)
        r = bus.read(1, R.TORQUE_LIMIT)
        ok &= r == v
        print(f"  write/read-back {v:>5} -> {r:<5} {'ok' if r == v else '!! FAIL'}")

    vals = show(servo, "  — loopback")
    ok &= set(vals) == set(CEILINGS)
    print(f"\n  {'selftest ok' if ok else '!! SELFTEST FAILED'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
