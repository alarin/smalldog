"""
loop.py — the 50 Hz tick: read feedback, ask a controller, write goals, hold time.

    python runtime/loop.py --selftest        # 5 s of loop against a loopback bus

`Runtime` does not know what a trot is. It takes

    source(dt, feedback) -> 12 joint angles in radians, in `calib.joints` order

and runs it against the bus, which is why the analytic gait (today) and the ONNX
policy out of `rl/` (step 6) can share a safety layer, a calibration and a timing
report instead of each growing their own.

The tick, in order, and the order is a decision
-----------------------------------------------
    read (SyncRead, 12 x 15 B)  ->  guard  ->  source  ->  write (SyncWrite)

The alternative — write first, then read — puts the write at a fixed offset from
the tick boundary and so has less jitter on the actuation, at the cost of acting
on feedback that is a whole tick old. That trade is worth making when the compute
between them is expensive and variable. It is not here: `robot/README.md`'s
arithmetic puts the whole bus exchange at 3.2 ms of a 20 ms tick and the trot's IK
is tens of microseconds of pure Python, so the write lands within a few hundred
microseconds of the same place every time and the feedback is as fresh as the bus
can make it. Revisit this the day the source is a neural network with a variable
compile.

`dt` is measured, not assumed
-----------------------------
The gait integrates its phase with the `dt` it is handed, so handing it the
nominal 20 ms when the tick actually took 31 ms makes the gait run slow in the
world while believing it is on time — and the error accumulates silently. So the
loop passes the *measured* interval, clamped to [0.5, 2] x nominal: a clamp is
needed because a single scheduler hiccup would otherwise ask the gait to advance
half a stride in one step, and half a stride in one step is a leg thrown at the
floor. An overrun is counted and reported rather than hidden.

Torque has exactly one owner
----------------------------
This file, and within it `__exit__`. `safety.Guard` raises, `Runtime` cuts torque,
and `run()` itself deliberately catches nothing: a trip, a bus error, Ctrl-C, a
source that raises and a run that simply ends all leave by the same door. Use
`Runtime` as a context manager — `with rt: ...`, the way `bench/sweep.py` uses its
`finally` — and there is no path that leaves twelve servos powered.
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from feetech import registers as R                                   # noqa: E402
from feetech.bus import Bus, BusError                                # noqa: E402
from feetech.loopback import LoopbackBus                             # noqa: E402
from runtime.calib import Calibration                                # noqa: E402
from runtime.safety import Guard, Limits, Tripped                    # noqa: E402

CTRL_HZ = 50.0                     # robot/README.md, "The 50 Hz budget"


def latency_timer(port) -> int | None:
    """The FTDI/CH340 driver's own buffering, in ms. None if it cannot be read.

    Ships at 16 ms on FTDI, which is most of a 20 ms tick on its own and is the
    single most common reason a loop that is fine on paper misses every deadline.
    `robot/README.md` has the `echo 1 | sudo tee` for it.
    """
    try:
        base = os.path.basename(str(port))
        with open(f"/sys/bus/usb-serial/devices/{base}/latency_timer") as f:
            return int(f.read().strip())
    except (OSError, ValueError):
        return None


class Runtime:
    def __init__(self, bus: Bus, calib: Calibration, hz=CTRL_HZ,
                 limits: Limits | None = None, log=print):
        self.bus, self.calib, self.log = bus, calib, log
        self.hz = float(hz)
        self.dt = 1.0 / self.hz
        self.servos = calib.attach(bus)
        self.guard = Guard(calib.joints, limits, log=log)
        self.goal = {n: 0.0 for n in calib.joints}
        self.torque_on = False
        self.ticks = self.overruns = self.bus_errors = 0
        self._late = []

    # ------------------------------------------------------------- lifecycle
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cut()
        return False

    def cut(self):
        """Torque off, now, on every servo. Safe to call twice, and it must be."""
        try:
            self.bus.sync_write(R.TORQUE_ENABLE, {i: 0 for i in self.calib.ids})
        except BusError as e:
            self.log(f"!! could not cut torque over the bus: {e}")
            for n in self.calib.joints:               # one at a time, best effort
                try:
                    self.servos[n].torque(False)
                except BusError:
                    pass
        self.torque_on = False

    # ------------------------------------------------------------------- io
    def read(self) -> dict:
        """One SyncRead of the whole robot -> {joint: decoded dict or None}.

        A SyncRead is one broadcast and twelve replies, so a single servo that
        answers late or not at all malforms the *stream* and the whole read raises.
        Taking that at face value would turn one marginal connector into a robot
        that thinks its entire bus is dead — and `Guard` would cut torque for it. So
        a failed bulk read falls back to reading each servo on its own, which costs
        a tick's deadline but localises the fault to the joint that actually has it.
        The overrun is counted; going blind is not an option worth the microseconds.
        """
        try:
            raw = self.bus.sync_read(R.FEEDBACK_START, R.FEEDBACK_LEN, self.calib.ids)
        except BusError:
            self.bus_errors += 1
            raw = {}
            for n in self.calib.joints:
                try:
                    raw[self.calib.id[n]] = self.bus.read(
                        self.calib.id[n], R.FEEDBACK_START, R.FEEDBACK_LEN)
                except BusError:
                    pass
        out = {}
        for n in self.calib.joints:
            r = raw.get(self.calib.id[n])
            out[n] = self.servos[n].decode(r) if r and len(r) >= R.FEEDBACK_LEN else None
        return out

    def send(self, q) -> list:
        """Clamp to the soft limits and write all twelve goals in one packet.

        The clamp is here and not only in the gait because the runtime is the last
        line: it must hold for any source, including one that has just been handed
        a NaN or a policy trained against limits that have since moved.
        """
        # A NaN out of a controller must not become a MOVE. Zero is the mechanical
        # zero — legs straight down — so "clamp it to zero" is a lunge under 2.5 kg;
        # holding the last goal is the only choice that does nothing.
        q = [self.goal[n] if (v is None or not math.isfinite(v)) else float(v)
             for n, v in zip(self.calib.joints, q)]
        q = self.calib.clamp(q)
        counts = {}
        for n, v in zip(self.calib.joints, q):
            self.goal[n] = v
            counts[self.calib.id[n]] = self.servos[n].to_counts(v)
        self.bus.sync_write(R.GOAL_POSITION, counts)
        return q

    # -------------------------------------------------------------- preflight
    def preflight(self, port=None) -> dict:
        """Ping everything, read the control registers, check the host. No motion."""
        out = {"missing": [], "latency_timer": latency_timer(port) if port else None}
        for n in self.calib.joints:
            if not self.bus.ping(self.calib.id[n]):
                out["missing"].append(n)
        if out["missing"]:
            self.log(f"!! no answer from: {', '.join(out['missing'])}")
        else:
            self.log(f"all {len(self.calib.joints)} servos answered")

        lt = out["latency_timer"]
        if lt is None:
            self.log("latency_timer: unreadable (not a USB serial port, or not Linux)")
        elif lt > 2:
            self.log(f"!! latency_timer is {lt} ms of a {1e3*self.dt:.0f} ms tick — "
                     f"set it to 1, see robot/README.md")
        else:
            self.log(f"latency_timer: {lt} ms")

        # ACCELERATION and GOAL_SPEED shape every setpoint this loop streams. They are
        # reported, never written: the servo's own settings are what any future fit is
        # conditional on, and this file has no business moving them silently.
        regs, odd = {}, []
        for n in self.calib.joints:
            try:
                regs[n] = {k: self.servos[n].registers().get(k)
                           for k in ("ACCELERATION", "GOAL_SPEED", "MODE", "P_COEF")}
            except BusError as e:
                regs[n] = {"error": str(e)}
                continue
            r = regs[n]
            if r.get("MODE") not in (0, None):
                odd.append(f"{n}: MODE={r['MODE']}, not position control")
            if r.get("ACCELERATION"):
                odd.append(f"{n}: ACCELERATION={r['ACCELERATION']}, "
                           f"a streamed setpoint wants 0")
        out["registers"] = regs
        for line in odd:
            self.log("!! " + line)

        fb = self.read()
        volts = [f["volt"] for f in fb.values() if f]
        temps = [f["temp"] for f in fb.values() if f]
        if volts:
            self.log(f"bus: {min(volts):.1f}..{max(volts):.1f} V, "
                     f"{min(temps):.0f}..{max(temps):.0f} C")
            out["volt"] = (min(volts), max(volts))
        out["ok"] = not out["missing"]
        return out

    # ----------------------------------------------------------------- engage
    def engage(self, q_target, ramp_s=2.0, max_rate=None):
        """Take the weight: goals to where the joints already are, torque on, ramp.

        The order is the whole point. Enabling torque with a stale Goal Position is
        how a robot throws itself across the bench — the servo has been holding some
        number since it was last powered and will drive to it the instant it can. So
        the present position is written as the goal *first*, with torque still off,
        and only then does torque come on. Nothing moves at that moment by
        construction.

        The ramp after it is a cosine ease over `ramp_s`, rate-limited on top: two
        seconds to stand up looks slow and is the difference between a servo taking
        up 2.5 kg and a servo hitting it.
        """
        rate = max_rate or self.calib.params.get("joint_velocity_limit", 4.7) * 0.5
        fb = self.read()
        blind = [n for n in self.calib.joints if fb[n] is None]
        if blind:
            raise Tripped(f"cannot read {', '.join(blind)} — refusing to take the weight "
                          f"from a pose that is a guess")
        q0 = [fb[n]["q"] for n in self.calib.joints]
        self.send(q0)                                    # no jump when torque arrives
        self.bus.sync_write(R.TORQUE_ENABLE, {i: 1 for i in self.calib.ids})
        self.torque_on = True
        self.log(f"torque on at the measured pose; ramping over {ramp_s:.1f} s")

        q_target = self.calib.clamp(q_target)
        n = max(1, int(round(ramp_s / self.dt)))
        q_prev = list(q0)
        t0 = time.perf_counter()
        for k in range(1, n + 1):
            a = 0.5 - 0.5 * math.cos(math.pi * k / n)    # 0 -> 1, zero slope at both ends
            want = [b + (t - b) * a for b, t in zip(q0, q_target)]
            step = rate * self.dt
            q_prev = [p + max(-step, min(step, v - p)) for v, p in zip(want, q_prev)]
            self.send(q_prev)
            self.guard.update(self.dt, self.read(), self.goal)
            self._sleep_until(t0 + k * self.dt)
        return q_prev

    def relax(self, q=None, ramp_s=1.5):
        """Optionally ramp to a pose, then cut torque. The polite way to finish."""
        if q is not None and self.torque_on:
            try:
                self.engage_ramp_to(q, ramp_s)
            except (BusError, Tripped) as e:
                self.log(f"!! could not sit down ({e}); cutting torque where it is")
        self.cut()
        self.log("torque off")

    def engage_ramp_to(self, q_target, ramp_s=1.5, max_rate=None):
        """Ramp from the current goal to another pose, torque already on."""
        rate = max_rate or self.calib.params.get("joint_velocity_limit", 4.7) * 0.5
        q0 = [self.goal[n] for n in self.calib.joints]
        q_target = self.calib.clamp(q_target)
        n = max(1, int(round(ramp_s / self.dt)))
        q_prev, t0 = list(q0), time.perf_counter()
        for k in range(1, n + 1):
            a = 0.5 - 0.5 * math.cos(math.pi * k / n)
            want = [b + (t - b) * a for b, t in zip(q0, q_target)]
            step = rate * self.dt
            q_prev = [p + max(-step, min(step, v - p)) for v, p in zip(want, q_prev)]
            self.send(q_prev)
            self._sleep_until(t0 + k * self.dt)
        return q_prev

    # -------------------------------------------------------------- the loop
    def run(self, source, seconds=None, on_tick=None) -> dict:
        """Tick `source` at `hz` until `seconds` elapse, it raises StopIteration, or
        something trips. Returns the timing report."""
        t0 = time.perf_counter()
        prev = t0
        k = 0
        while seconds is None or (time.perf_counter() - t0) < seconds:
            now = time.perf_counter()
            dt = min(2.0 * self.dt, max(0.5 * self.dt, now - prev))
            prev = now

            fb = self.read()
            self.guard.update(dt, fb, self.goal)
            try:
                q = source(dt, fb)
            except StopIteration:
                break
            self.send(q)
            if on_tick:
                on_tick(k, dt, fb)

            self.ticks = k = k + 1
            self._sleep_until(t0 + k * self.dt)
        return self.report()

    def _sleep_until(self, when):
        slack = when - time.perf_counter()
        if slack > 0:
            time.sleep(slack)
        else:
            self.overruns += 1
            self._late.append(-slack)

    def report(self) -> dict:
        late = sorted(self._late)
        out = dict(ticks=self.ticks, overruns=self.overruns, bus_errors=self.bus_errors,
                   hz=self.hz, **self.guard.summary())
        if late:
            out["late_p50_ms"] = 1e3 * late[len(late) // 2]
            out["late_max_ms"] = 1e3 * late[-1]
        out["bus"] = self.bus.stats()
        return out

    def report_lines(self) -> str:
        r = self.report()
        pct = 100.0 * r["overruns"] / max(1, r["ticks"])
        s = [f"{r['ticks']} ticks at {r['hz']:.0f} Hz, {r['overruns']} late ({pct:.1f} %)"]
        if "late_max_ms" in r:
            s.append(f"  late by p50 {r['late_p50_ms']:.1f} ms, max {r['late_max_ms']:.1f} ms")
        b = r["bus"]
        if b.get("n"):
            s.append(f"  bus p50 {b['p50_ms']:.2f} ms, p99 {b['p99_ms']:.2f} ms, "
                     f"{b['timeouts']} timeouts, {b['checksum_errors']} checksum, "
                     f"sync_read={b['sync_read']}")
        s.append("  " + self.guard.report())
        return "\n".join(s)


# ------------------------------------------------------- a bus for the self-test
class FollowingLoopback(LoopbackBus):
    """`feetech.loopback.LoopbackBus` plus one rule: present position = goal.

    This is NOT a servo model and must never grow into one — `rl/actuator.py` is
    the one copy of the ST3215's law, and a second, cruder copy hiding in the test
    harness is exactly how a loop that "works in dry-run" stops working on the
    bench. The identity map exists so that the *loop's own* bookkeeping — the ramp,
    the clamp, the tracking check, the timing — is exercised end to end with no
    hardware. Anything that needs the servo to lag, saturate or draw current
    belongs in the simulator.
    """

    def __init__(self, ids, volt=12.0, temp=30.0):
        centre = R.COUNTS_PER_TURN // 2
        super().__init__({i: {R.PRESENT_VOLTAGE: int(round(volt / R.VOLTAGE_LSB_V)),
                              R.PRESENT_TEMPERATURE: int(temp),
                              R.PRESENT_POSITION: centre,
                              R.GOAL_POSITION: centre} for i in ids})

    def _handle(self, frame):
        super()._handle(frame)
        for i in self.mem:
            self.set(i, R.PRESENT_POSITION, self.get(i, R.GOAL_POSITION))


# ============================================================== self-test
def _selftest(seconds=2.0) -> int:
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {name}{extra}")

    calib = Calibration.default()
    bus = Bus(transport=FollowingLoopback(calib.ids), discard_echo=False)
    rt = Runtime(bus, calib, log=lambda *_: None)

    pre = rt.preflight()
    chk("preflight finds every servo", pre["ok"])

    # a source that asks for a slow sine on every joint, well inside the soft limits
    def sine(dt, fb, state={"t": 0.0}):
        state["t"] += dt
        return [0.3 * math.sin(2 * math.pi * 0.5 * state["t"])] * 12

    with rt:
        q = rt.engage([0.0] * 12, ramp_s=0.2)
        chk("engage leaves torque on", rt.torque_on)
        chk("engage reaches the target", max(abs(v) for v in q) < 1e-6)
        chk("torque enable reached the servos", bus.io.get(1, R.TORQUE_ENABLE) == 1)
        rt.run(sine, seconds=seconds)
    chk("the context manager cuts torque", bus.io.get(1, R.TORQUE_ENABLE) == 0)
    chk("torque_on is false after the exit", rt.torque_on is False)

    r = rt.report()
    want = int(seconds * CTRL_HZ)
    chk(f"~{want} ticks in {seconds:.0f} s", abs(r["ticks"] - want) <= 3,
        f" (got {r['ticks']})")
    chk("no bus errors", r["bus_errors"] == 0)
    chk("the guard saw no tracking error", r["q_err"] < 0.05, f" ({r['q_err']:.3f} rad)")
    chk("late ticks are rare", r["overruns"] <= max(2, 0.05 * r["ticks"]),
        f" ({r['overruns']} of {r['ticks']})")

    # the clamp is the last line: a source that asks for the moon gets the soft limit
    rt2 = Runtime(Bus(transport=FollowingLoopback(calib.ids), discard_echo=False),
                  calib, log=lambda *_: None)
    q = rt2.send([99.0] * 12)
    chk("send clamps to the soft limits",
        abs(q[2] - calib.soft["fl_knee"]) < 1e-9, f" (knee -> {q[2]:.2f} rad)")
    held = rt2.send([0.4] * 12)
    q = rt2.send([float("nan")] * 12)
    chk("a NaN holds the last goal rather than moving", q == held)

    # one servo off the bus: the read localises it instead of going blind, and
    # engage refuses rather than ramping twelve joints from an eleven-joint guess
    partial = FollowingLoopback(calib.ids[:-1])
    rt4 = Runtime(Bus(transport=partial, discard_echo=False), calib, log=lambda *_: None)
    fb = rt4.read()
    chk("a missing servo reads as None", fb["rr_knee"] is None)
    chk("... and the other eleven still read", sum(v is not None for v in fb.values()) == 11)
    chk("... and it was counted as a bus error", rt4.bus_errors == 1)
    refused = False
    try:
        rt4.engage([0.0] * 12, ramp_s=0.1)
    except Tripped:
        refused = True
    chk("engage refuses a pose it cannot read", refused)
    chk("... without having enabled torque", partial.get(1, R.TORQUE_ENABLE) == 0)

    # a trip must cut torque and propagate
    hot = FollowingLoopback(calib.ids, temp=90.0)
    rt3 = Runtime(Bus(transport=hot, discard_echo=False), calib, log=lambda *_: None)
    tripped = False
    try:
        with rt3:
            rt3.engage([0.0] * 12, ramp_s=0.1)
    except Tripped:
        tripped = True
    chk("an over-temperature servo trips the engage", tripped)
    chk("... and torque is off afterwards", hot.get(1, R.TORQUE_ENABLE) == 0)

    print("loop:", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--seconds", type=float, default=2.0)
    a = ap.parse_args()
    raise SystemExit(_selftest(a.seconds))
