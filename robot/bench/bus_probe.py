"""
bus_probe.py — what the bus actually costs, on this machine, with this adapter.

    python bench/bus_probe.py --port /dev/ttyUSB0 --ids 1
    python bench/bus_probe.py --port /dev/ttyUSB0 --ids 1,2,3,4,5,6,7,8,9,10,11,12
    python bench/bus_probe.py --dry-run            # no hardware, exercises the code

Run this BEFORE step 4, not after. The command delay it measures is an input to
the training randomisation, and a guessed delay trains a policy against a robot
that does not exist. It is also the first thing to look at when the 50 Hz loop
misses its deadline.

The wire is not the constraint and never was. At 1 Mbit a byte is 10 us:

    SyncWrite, 12 goal positions      44 bytes    0.44 ms
    SyncRead request, 12 servos       20 bytes    0.20 ms
    SyncRead replies, 15 bytes each   21 x 12     2.52 ms
                                                  ------
    one full 12-servo exchange                    3.2 ms of wire in a 20 ms tick

Everything above that is the host: USB frame scheduling, the driver's latency
timer, and the servo's own Return Delay register. An FTDI adapter defaults to
latency_timer = 16 ms and will single-handedly eat the tick —

    cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer   # set it to 1

— and CH340, which is what an FE-URT-1 is likely to carry, has no such knob and
its own behaviour. Which one you have changes the answer, so measure it rather
than reading a number off a forum.

The report separates the two: wire time is arithmetic, p50/p99 are what the host
actually delivered, and the gap between them is the host's contribution.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from feetech import registers as R                                  # noqa: E402
from feetech.bus import Bus, BusError, Servo, pack                  # noqa: E402

BIT_US = 10.0                     # 10 bits per byte at 1 Mbit


def wire_us(n_bytes: int) -> float:
    return n_bytes * BIT_US


def budget(n_servos: int) -> dict:
    sw = len(pack(R.BROADCAST_ID, R.SYNC_WRITE, bytes([R.GOAL_POSITION, 2])
                  + b"".join(bytes([i]) + b"\0\0" for i in range(n_servos))))
    sr_req = len(pack(R.BROADCAST_ID, R.SYNC_READ,
                      bytes([R.FEEDBACK_START, R.FEEDBACK_LEN])
                      + bytes(range(1, n_servos + 1))))
    sr_rep = (6 + R.FEEDBACK_LEN) * n_servos
    return {"sync_write_bytes": sw, "sync_read_request_bytes": sr_req,
            "sync_read_reply_bytes": sr_rep,
            "wire_us_total": wire_us(sw + sr_req + sr_rep)}


def timed(fn, n: int) -> list[float]:
    out = []
    for _ in range(n):
        t0 = time.perf_counter()
        try:
            fn()
        except BusError:
            pass
        out.append(time.perf_counter() - t0)
    return out


def report(name: str, ts: list[float]) -> dict:
    s = sorted(ts)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]
    d = {"n": len(s), "mean_ms": 1e3 * statistics.fmean(s), "p50_ms": 1e3 * q(.5),
         "p95_ms": 1e3 * q(.95), "p99_ms": 1e3 * q(.99), "max_ms": 1e3 * s[-1]}
    print(f"  {name:<36} p50 {d['p50_ms']:6.2f}  p95 {d['p95_ms']:6.2f}  "
          f"p99 {d['p99_ms']:6.2f}  max {d['max_ms']:6.2f} ms")
    return d


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--ids", default="1")
    ap.add_argument("--n", type=int, default=500, help="samples per measurement")
    ap.add_argument("--loop-hz", type=float, default=50.0)
    ap.add_argument("--loop-s", type=float, default=10.0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "..", "rl",
                                                  "params", "bus_timing.json"))
    a = ap.parse_args()
    ids = [int(x) for x in a.ids.split(",") if x]

    if a.dry_run:
        from feetech.loopback import LoopbackBus
        bus = Bus(transport=LoopbackBus({i: {} for i in ids}), discard_echo=False)
        print("DRY RUN — a loopback with no wire and no servo. The numbers below are")
        print("this machine's Python overhead only; they say the code runs, not what")
        print("the bus costs.\n")
    else:
        bus = Bus(a.port, a.baud)

    b = budget(len(ids))
    print(f"wire time, arithmetic, {len(ids)} servos at {a.baud/1e6:g} Mbit:")
    print(f"  SyncWrite {b['sync_write_bytes']} B, SyncRead {b['sync_read_request_bytes']}"
          f" B out + {b['sync_read_reply_bytes']} B back"
          f"  ->  {b['wire_us_total']/1000:.2f} ms\n")
    print("measured round trips:")

    out = {"ids": ids, "budget": b, "dry_run": a.dry_run}
    servo = Servo(bus, ids[0])
    out["ping"] = report("ping, one servo", timed(lambda: bus.ping(ids[0]), a.n))
    out["read_pos"] = report("read position, one servo",
                             timed(lambda: bus.read(ids[0], R.PRESENT_POSITION), a.n))
    out["feedback"] = report(f"read {R.FEEDBACK_LEN} B feedback, one servo",
                             timed(servo.feedback, a.n))
    out["sync_write"] = report(f"SyncWrite goal, {len(ids)} servos",
                               timed(lambda: bus.sync_write(
                                   R.GOAL_POSITION, {i: 2048 for i in ids}), a.n))
    out["sync_read"] = report(f"SyncRead feedback, {len(ids)} servos",
                              timed(lambda: bus.sync_read(
                                  R.FEEDBACK_START, R.FEEDBACK_LEN, ids), a.n))
    out["sync_read_supported"] = bus._sync_read_ok

    # The number that matters: a whole control tick, paced the way the runtime
    # paces it. Jitter here is what the policy experiences as a varying delay.
    print(f"\nfull tick at {a.loop_hz:g} Hz for {a.loop_s:g} s "
          f"(SyncRead then SyncWrite, absolute deadlines):")
    period = 1.0 / a.loop_hz
    work, late, start = [], 0, time.perf_counter()
    deadline = start
    while time.perf_counter() - start < a.loop_s:
        deadline += period
        t0 = time.perf_counter()
        try:
            bus.sync_read(R.FEEDBACK_START, R.FEEDBACK_LEN, ids)
            bus.sync_write(R.GOAL_POSITION, {i: 2048 for i in ids})
        except BusError:
            pass
        work.append(time.perf_counter() - t0)
        slack = deadline - time.perf_counter()
        if slack < 0:
            late += 1
            deadline = time.perf_counter()
        else:
            time.sleep(slack)
    out["tick_work"] = report("work per tick", work)
    out["ticks_late"] = late
    out["tick_budget_ms"] = 1e3 * period
    print(f"  missed deadlines: {late} of {len(work)}"
          f"   ({100.0*late/max(1,len(work)):.1f} %)")
    out["errors"] = {"timeouts": bus.n_timeout, "checksum": bus.n_checksum}
    print(f"  bus errors: {out['errors']}")

    print("\nfor step 4's randomisation: sample the command delay from the measured"
          f"\n  tick work — p50 {out['tick_work']['p50_ms']:.2f} ms, "
          f"p99 {out['tick_work']['p99_ms']:.2f} ms — plus one tick of transport"
          "\n  latency, and widen the range rather than centring it. A policy robust"
          "\n  to more delay than it will see costs a little performance; one robust"
          "\n  to less costs the robot.")
    if not a.dry_run:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(out, open(a.out, "w"), indent=2)
        print(f"\nwrote {os.path.relpath(a.out)}")
    else:
        print("\n(dry run: nothing written)")


if __name__ == "__main__":
    main()
