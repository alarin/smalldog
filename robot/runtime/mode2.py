"""
mode2.py — the runtime with the position loop on the host: MODE 2 (open-loop PWM).

    python runtime/mode2.py --selftest                         # no hardware
    python runtime/mode2.py --port /dev/ttyACM0 --id 1 --sine  # one free servo, a bench check

Why
---
The ST3215's own position loop profiles every goal at a firmware-capped 7.7 rad/s²
(`ideas/FAST_SERVOS.md`): a ±15° sine gets through at 74 % at 1 Hz and 7 % at 5 Hz,
and every RL policy that asked for a 5 Hz trot rocked in place. MODE 2 drives the
bridge at a duty with no profile, and a PD on the host — measured on one servo at
250 Hz — passes the same sine at 99 / 97 / 95 % at 1 / 2 / 3 Hz.

`Mode2Runtime` is `loop.Runtime` with the servo's loop replaced by ours. The 50 Hz
tick is unchanged — read, guard, source, send — and the source still hands over
joint angles. What changes is that `send` does not write a goal: it stores the
target, and the time the base loop would have SLEPT until the next tick is spent
running the PD as fast as the bus goes:

    duty = kp·(target − q) − kd·ω + kff·ω_target        (joint frame, then the servo's)

with the gains from `bench/pwm_loop.py`'s tuned run and scaled by 11.3 V / pack
volts so the stiffness (≈ 41 N·m/rad, the firmware's own) does not drift with the
pack.

Three things the firmware did for us that we now do ourselves
-------------------------------------------------------------
**The centre.** `PRESENT_POSITION` is raw in MODE 2 — `OFFSET` (the servo's own
middle-position calibration) is not applied, so `calib.json`'s centres, taken in
position mode, are off by exactly that. The switch measures it: read counts in
MODE 0, switch, read again, shift the centre by the difference. The position is
then decoded through the 0/4095 wrap, because a raw centre can sit anywhere.

**Current.** The firmware still cuts torque at `PROTECTION_CURRENT` (2 A) after
`OVERCURRENT_TIME` (2 s) in MODE 2 — measured stalled, `bench/mode2_protect.py`
— and a servo cut that way is dead until torque is cycled: a leg going limp
mid-stride. So the duty is folded back on the register current before the
firmware can act: full duty below `i_soft`, down to `i_floor` of it at `i_hard`,
per servo, and the same fold on the whole bus against `i_sum`, which is what
stops twelve joints pulling stall current together off one pack. `safety.Guard`
still trips at its own `current_a` for `current_hold_s`; the fold is what keeps
it from having to.

**Runaway.** The loop closes in the encoder's frame, so a sign wrong in
`calib.json` moves a joint the wrong way but still tracks — the same fault it is
in position mode. What does run away is a servo whose bridge turns the count the
other way from the one measured (+duty, count down: one unit, one firmware), or
a joint driven off its target by a jam. Positive feedback at 9000 duty/rad hits
full duty in a few counts, and the guard's tracking check needs 0.3 s — 1.4 rad
of leg at 4.7 rad/s. So the host loop has its own: an error past `runaway_rad`
on any sub-tick zeroes every duty and trips at once.

**A torque ceiling.** In position mode the firmware never let a joint hold full
duty: `OVERLOAD_TORQUE` 80 cut anything above 80 % to 20 % after 2 s, and
`PROTECTION_CURRENT` bounded a held stall to ~2.3 N·m. In MODE 2 both are ours to
provide, and the fold above only sees a stall once the current register has
climbed. `duty_cap` is the per-joint ceiling that stands in for them: a joint
capped at 600 gets 7 V of drive at the pack's 12 V, which the torque rig read as
2.3 N·m — the firmware's own sustained bound — against ~3.3 N·m at full duty on a
fresh pack. It exists because the front hip brackets broke under the pitch
servos' full duty on the first day the feet gripped (siped soles, 2026-09-17):
`3d/fea.py`'s `stall` case reads SF 2.2 for `hip_bracket_A` at 3.2 N·m, and a
hop's landing is on top of that. The cost is speed: no-load speed scales with
drive, so a capped joint is slower than the 4.7 rad/s ceiling the gait is fitted
to. Which joints, and how much, is **verify** — `--duty-cap pitch=600` on
`walk.py` / `policy.py`, `duty_cap:=` on `robot.launch.py`.

MODE is EEPROM and `cut()` puts 0 back, duty 0 and torque off first. A process
killed with SIGKILL cannot: the servos keep their last duty until power is cycled
or the next run's preflight finds MODE 2 and restores it.
"""
from __future__ import annotations

import argparse
import collections
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from feetech import registers as R                                   # noqa: E402
from feetech.bus import Bus, BusError, Servo                         # noqa: E402
from feetech.loopback import LoopbackBus                             # noqa: E402
from runtime.calib import Calibration                                # noqa: E402
from runtime.loop import CTRL_HZ, Runtime                            # noqa: E402
from runtime.safety import Limits, Tripped                           # noqa: E402

#: Measured on one servo at 11.3 V, 250 Hz, `bench/pwm_loop.py --kp 9000 --kd 150
#: --kff 200`: 0.99 / 0.97 / 0.95 / 0.62 of a ±15° sine at 1 / 2 / 3 / 5 Hz.
KP, KD, KFF = 9000.0, 150.0, 200.0
VOLT_REF = 11.3
DUTY_MAX = 1000


def parse_duty_caps(spec, joints) -> dict:
    """`"pitch=600,fl_roll=800"` -> {joint: cap}. A key is a joint name or a joint
    kind (`roll`, `pitch`, `knee`: every joint whose name ends in it); an empty
    spec is no caps. Unknown keys and caps outside 1..DUTY_MAX are errors, not
    silence — a typo here is a bracket."""
    caps = {}
    for item in (spec or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"duty cap {item!r}: want NAME=DUTY")
        key, val = (x.strip() for x in item.split("=", 1))
        cap = int(val)
        if not 1 <= cap <= DUTY_MAX:
            raise ValueError(f"duty cap {item!r}: {cap} is outside 1..{DUTY_MAX}")
        hit = [n for n in joints if n == key or n.endswith("_" + key)]
        if not hit:
            raise ValueError(f"duty cap {item!r}: no joint called or ending in {key!r}")
        for n in hit:
            caps[n] = min(cap, caps.get(n, DUTY_MAX))
    return caps


def duty_word(d: int) -> int:
    """Register 44 in MODE 2: magnitude in the low bits, direction in bit 10."""
    d = int(max(-DUTY_MAX, min(DUTY_MAX, d)))
    return abs(d) | (0x400 if d < 0 else 0)


class RawServo(Servo):
    """A `Servo` whose centre is in the raw frame, decoded through the wrap.

    In MODE 2 the encoder count is not offset, so a centre that position mode put
    at 2048 can be anywhere in 0..4095 and a joint at ±1 rad can cross 0. The
    shortest way round the circle is the joint angle; `Servo.to_rad` would hand
    back ±2π of it.
    """

    def to_rad(self, counts: int) -> float:
        half = R.COUNTS_PER_TURN // 2
        d = (counts - self.centre + half) % R.COUNTS_PER_TURN - half
        return self.sign * d * 2 * math.pi / R.COUNTS_PER_TURN


class Mode2Runtime(Runtime):
    def __init__(self, bus: Bus, calib: Calibration, hz=CTRL_HZ,
                 limits: Limits | None = None, log=print,
                 kp=KP, kd=KD, kff=KFF, duty_max=DUTY_MAX,
                 i_soft=1.4, i_hard=2.0, i_floor=0.25, i_sum=10.0,
                 runaway_rad=1.2, sub_hz=0.0, smooth=True, duty_cap=None):
        super().__init__(bus, calib, hz, limits, log)
        #: {joint: duty}, the per-joint ceiling under `duty_max` (the docstring, "A
        #: torque ceiling"). `parse_duty_caps` turns the CLI's "pitch=600" into it.
        self.duty_cap = {n: int(c) for n, c in (duty_cap or {}).items()
                         if n in calib.joints and c < duty_max}
        self.peak_duty_joint = {n: 0 for n in calib.joints}
        #: The source hands over a target every tick, so the host loop sees a 50 Hz
        #: staircase — and at kp 9000 each 20 ms step is a ~600-duty kick that the
        #: joint answers and overshoots (the load flapped ±500 tick to tick in stance
        #: on the first air trot). Smoothing slides the target linearly from the
        #: previous setpoint to the new one over the tick: one tick of lag, no kicks.
        #: Off for a policy trained against a PD that saw its target at once.
        self.smooth = smooth
        self._goal_prev = dict(self.goal)
        self._t_goal = time.perf_counter()
        self.kp, self.kd, self.kff, self.duty_max = kp, kd, kff, duty_max
        self.i_soft, self.i_hard, self.i_floor, self.i_sum = i_soft, i_hard, i_floor, i_sum
        self.runaway_rad = runaway_rad
        self.sub_dt = 1.0 / sub_hz if sub_hz else 0.0
        self.v_ff = {n: 0.0 for n in calib.joints}
        self.duty = {n: 0 for n in calib.joints}
        self._fb = None
        self._mode0: dict[int, int] = {}
        self.in_mode2 = False
        self.sub_ticks = 0
        self.sub_time = 0.0                # wall seconds spent in the sub-tick loop
        self.folds = 0                     # sub-ticks on which some duty was folded
        self.peak_duty = 0
        self._trace = collections.deque(maxlen=40)   # (t, joint, q, goal, w, duty, I)

    # ------------------------------------------------------------- the mode
    def enter_mode2(self):
        """Torque off, measure the raw-frame centre, switch every servo. No motion.

        One servo at a time and verified by reading back: MODE is EEPROM and a
        write that did not take leaves that joint under the firmware's loop
        while the host drives the other eleven — which is worse than refusing.
        """
        if self.in_mode2:
            return
        self.bus.sync_write(R.TORQUE_ENABLE, {i: 0 for i in self.calib.ids})
        time.sleep(0.05)
        raw = {}
        for n in self.calib.joints:
            i = self.calib.id[n]
            self._mode0[i] = self.bus.read(i, R.MODE)
            c0 = self.bus.read(i, R.PRESENT_POSITION)
            self.bus.write(i, R.MODE, 2)
            if self.bus.read(i, R.MODE) != 2:
                self.leave_mode2()
                raise Tripped(f"{n}: MODE would not take 2")
            # The raw frame appears some tens of ms AFTER the write: a read straight
            # after it still returns the offset count (measured: 2018, 2018, then 100
            # from 50 ms on). Wait, then insist on two identical reads — the servo is
            # not driving, so anything else is the switch still in progress.
            time.sleep(0.08)
            c2, prev = self.bus.read(i, R.PRESENT_POSITION), None
            for _ in range(20):
                if c2 == prev:
                    break
                time.sleep(0.02)
                prev, c2 = c2, self.bus.read(i, R.PRESENT_POSITION)
            half = R.COUNTS_PER_TURN // 2
            shift = (c2 - c0 + half) % R.COUNTS_PER_TURN - half
            # ... and it must be the servo's own OFFSET (raw = offset count + OFFSET;
            # 85 -> +86, -1918 -> -1918 on the bench). A disagreement is a frame that
            # has not settled or a joint that moved, and either one is a centre that
            # is wrong by the difference for the whole run.
            off = self.bus.read(i, R.OFFSET)
            if abs(shift - off) > 3:
                # once more from the top: a horn still coasting from the last run
                # moves between the two reads (15 counts on the bench)
                time.sleep(0.3)
                c0 = None
                self.bus.write(i, R.MODE, 0)
                time.sleep(0.1)
                c0 = self.bus.read(i, R.PRESENT_POSITION)
                self.bus.write(i, R.MODE, 2)
                time.sleep(0.1)
                c2 = self.bus.read(i, R.PRESENT_POSITION)
                shift = (c2 - c0 + half) % R.COUNTS_PER_TURN - half
            if abs(shift - off) > 3:
                self.leave_mode2()
                raise Tripped(f"{n}: raw centre shift {shift:+d} counts disagrees with "
                              f"OFFSET {off:+d} — the frame did not settle, or it moved")
            raw[n] = (self.calib.centre[n] + shift) % R.COUNTS_PER_TURN
            # A duty write in MODE 2 turns torque ON by itself (TORQUE_ENABLE reads 1
            # after it). Duty 0 holds nothing, so this is harmless, but it is why
            # `leave_mode2` writes duty 0 BEFORE torque 0 and never the other way.
            self.bus.write(i, R.GOAL_TIME, 0)
        self.servos = {n: RawServo(self.bus, self.calib.id[n], centre=raw[n],
                                   sign=self.calib.sign[n]) for n in self.calib.joints}
        self.in_mode2 = True
        # The goal starts AT the pose, so the first `send` sees no velocity: a goal
        # left at zero from construction would put kff x (pose / dt) of duty on the
        # very write that precedes torque-on.
        fb = self.read()
        for n in self.calib.joints:
            if fb[n] is not None:
                self.goal[n] = self._goal_prev[n] = fb[n]["q"]
            self.v_ff[n] = 0.0
        shifts = {n: (raw[n] - self.calib.centre[n]) for n in self.calib.joints}
        self.log("MODE 2 on all servos; raw centre shift: " + " ".join(
            f"{n}{v:+d}" for n, v in shifts.items() if v) or "MODE 2 on all servos")

    def leave_mode2(self):
        """Duty 0, torque off, MODE back to what it was. Best effort, every servo."""
        try:
            self.bus.sync_write(R.GOAL_TIME, {i: 0 for i in self.calib.ids})
            self.bus.sync_write(R.TORQUE_ENABLE, {i: 0 for i in self.calib.ids})
        except BusError as e:
            self.log(f"!! could not zero the duties over the bus: {e}")
        for n in self.calib.joints:
            i = self.calib.id[n]
            try:
                self.bus.write(i, R.GOAL_TIME, 0)
                self.bus.write(i, R.TORQUE_ENABLE, 0)
                self.bus.write(i, R.MODE, self._mode0.get(i, 0))
            except BusError as e:
                self.log(f"!! {n}: could not restore MODE ({e})")
        self.servos = self.calib.attach(self.bus)
        self.in_mode2 = False
        self.duty = {n: 0 for n in self.calib.joints}

    def __enter__(self):
        self.enter_mode2()
        return self

    def cut(self):
        if self.in_mode2:
            self.leave_mode2()
        super().cut()

    # ------------------------------------------------------------------- io
    def read(self) -> dict:
        self._fb = super().read()
        return self._fb

    def send(self, q) -> list:
        """Store the target and its velocity; drive one PD step off the last read."""
        q = [self.goal[n] if (v is None or not math.isfinite(v)) else float(v)
             for n, v in zip(self.calib.joints, q)]
        q = self.calib.clamp(q)
        for n, v in zip(self.calib.joints, q):
            self.v_ff[n] = (v - self.goal[n]) / self.dt
            self._goal_prev[n] = self.goal[n]
            self.goal[n] = v
        self._t_goal = time.perf_counter()
        if self._fb is not None:
            self._pd(self._fb)
        return q

    def _pd(self, fb: dict):
        """One duty per joint from one feedback frame, folded, written in one packet."""
        duties, cur, folded = {}, {}, False
        a = min(1.0, (time.perf_counter() - self._t_goal) / self.dt) if self.smooth else 1.0
        for n in self.calib.joints:
            f = fb.get(n)
            if f is None:
                duties[n] = 0                    # a joint we cannot hear gets no drive
                continue
            target = self._goal_prev[n] + (self.goal[n] - self._goal_prev[n]) * a
            err = target - f["q"]
            if abs(err) > self.runaway_rad:
                self._zero_all()
                self.log(f"!! trip trace, last sub-ticks (t, joint, q, goal, w, duty, I):")
                for row in self._trace:
                    if row[1] == n:
                        self.log("     %.3f %s q %+.3f goal %+.3f w %+.2f d %+5d I %.2f" % row)
                raise Tripped("host loop runaway — the loop is unstable on this joint, or "
                              "it is jammed",
                              n, abs(err), self.runaway_rad)
            scale = VOLT_REF / max(f["volt"], 6.0)
            d = scale * (self.kp * err - self.kd * f["w"] + self.kff * self.v_ff[n])
            # the bridge's sign is opposite to the encoder's: +duty turns the count down
            d = -self.calib.sign[n] * d
            fold = self._fold(f["current"], self.i_soft, self.i_hard)
            lim = self.duty_cap.get(n, self.duty_max) * fold
            if fold < 1.0 and abs(d) > lim:
                folded = True
            duties[n] = max(-lim, min(lim, d))
            cur[n] = f["current"]
        total = self._fold(sum(cur.values()), self.i_sum * self.i_soft / self.i_hard,
                           self.i_sum) if cur else 1.0
        if folded or total < 1.0:
            self.folds += 1
        words = {}
        now = time.perf_counter()
        for n in self.calib.joints:
            d = int(round(duties[n] * total))
            self.duty[n] = d
            self.peak_duty = max(self.peak_duty, abs(d))
            self.peak_duty_joint[n] = max(self.peak_duty_joint[n], abs(d))
            words[self.calib.id[n]] = duty_word(d)
            f = fb.get(n)
            if f is not None and abs(self.goal[n] - f["q"]) > 0.5 * self.runaway_rad:
                self._trace.append((now, n, f["q"], self.goal[n], f["w"], d, f["current"]))
        self.bus.sync_write(R.GOAL_TIME, words)

    def _fold(self, i, soft, hard) -> float:
        """1 below `soft`, `i_floor` at `hard`, linear between."""
        if i <= soft:
            return 1.0
        if i >= hard:
            return self.i_floor
        return 1.0 - (1.0 - self.i_floor) * (i - soft) / (hard - soft)

    def _zero_all(self):
        try:
            self.bus.sync_write(R.GOAL_TIME, {i: 0 for i in self.calib.ids})
        except BusError:
            pass
        self.duty = {n: 0 for n in self.calib.joints}

    # ------------------------------------------------------------ the loop
    def _sleep_until(self, when) -> float:
        """The base loop sleeps here; this one runs the PD until the deadline.

        Returns the slack the way the base does — negative when the tick was
        already late — so the 50 Hz schedule, its overrun count and its re-basing
        are untouched. The sub-ticks are the time the tick had to spare.
        """
        slack = when - time.perf_counter()
        if slack <= 0:
            self.overruns += 1
            self._late.append(-slack)
            return slack
        if not self.in_mode2 or not self.torque_on:
            time.sleep(slack)
            return slack
        while True:
            t = time.perf_counter()
            if when - t < 0.0015:            # not enough left for a read and a write
                break
            fb = self.read()
            self._pd(fb)
            self.sub_ticks += 1
            self.sub_time += time.perf_counter() - t
            if self.sub_dt:
                nxt = t + self.sub_dt
                while time.perf_counter() < min(nxt, when):
                    pass
        rest = when - time.perf_counter()
        if rest > 0:
            time.sleep(rest)
        return slack

    def report(self) -> dict:
        out = super().report()
        out["sub_ticks"] = self.sub_ticks
        out["sub_hz"] = self.sub_ticks / self.sub_time if self.sub_time else 0.0
        out["folds"] = self.folds
        out["peak_duty"] = self.peak_duty
        out["peak_duty_joint"] = dict(self.peak_duty_joint)
        out["duty_cap"] = dict(self.duty_cap)
        return out

    def report_lines(self) -> str:
        r = self.report()
        return (super().report_lines()
                + f"\n  host loop: {r['sub_hz']:.0f} Hz over the bus (kp {self.kp:.0f} kd "
                  f"{self.kd:.0f} kff {self.kff:.0f}), peak duty "
                  f"{r['peak_duty']} of {self.duty_max}, current fold on {r['folds']} "
                  f"of {r['sub_ticks']} sub-ticks"
                + (("\n  duty caps (peak/cap): " + ", ".join(
                    f"{n} {r['peak_duty_joint'][n]}/{c}" for n, c in r["duty_cap"].items()))
                   if r["duty_cap"] else ""))


# ------------------------------------------------------- a bus for the self-test
class DutyLoopback(LoopbackBus):
    """`LoopbackBus` plus the one rule MODE 2 needs: the count moves at a rate set
    by the duty, in the bridge's direction (+duty, count down).

    Not a servo model — `rl/actuator.py` is the one copy of that — but the host
    loop cannot be exercised at all against a count that never moves, and a
    velocity proportional to duty is the least a bridge on a free horn does.
    `OFFSET` is honoured only in MODE 0, which is the raw-frame rule the switch
    has to get right.
    """

    def __init__(self, ids, volt=12.0, temp=30.0, offset=300, rate=0.005, reversed_ids=()):
        self.offset, self.rate, self.raw = offset, rate, {}
        self.reversed = set(reversed_ids)          # a bridge that turns the count UP
        self._t = time.perf_counter()
        super().__init__({i: {R.PRESENT_VOLTAGE: int(round(volt / R.VOLTAGE_LSB_V)),
                              R.PRESENT_TEMPERATURE: int(temp), R.MODE: 0,
                              R.GOAL_TIME: 0,
                              # raw = offset count + OFFSET, so OFFSET is -offset here
                              R.OFFSET: (abs(offset) | (0x800 if offset > 0 else 0))}
                         for i in ids})
        for i in ids:
            self.raw[i] = 2048 - offset
            self._show(i)

    def _show(self, i):
        c = self.raw[i] if self.get(i, R.MODE) == 2 else self.raw[i] + self.offset
        self.set(i, R.PRESENT_POSITION, int(round(c)) % R.COUNTS_PER_TURN)

    def _handle(self, frame):
        now = time.perf_counter()
        dt, self._t = now - self._t, now
        for i in self.mem:
            if self.get(i, R.MODE) == 2 and self.get(i, R.TORQUE_ENABLE):
                w = self.get(i, R.GOAL_TIME)
                d = -(w & 0x3FF) if w & 0x400 else (w & 0x3FF)
                self.raw[i] -= (-d if i in self.reversed else d) * self.rate * dt * 1000
                self.set(i, R.PRESENT_SPEED, 0)
            self._show(i)
        super()._handle(frame)


# ============================================================== self-test
def _selftest(seconds=1.0) -> int:
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {name}{extra}")

    chk("duty word: +300 is 300", duty_word(300) == 300)
    chk("duty word: -300 sets bit 10", duty_word(-300) == (0x400 | 300))
    chk("duty word clamps", duty_word(5000) == 1000 and duty_word(-5000) == 0x400 | 1000)

    s = RawServo(None, 1, centre=10, sign=+1)
    chk("raw decode goes the short way round the wrap",
        abs(s.to_rad(4090) + 16 * 2 * math.pi / 4096) < 1e-9)

    calib = Calibration.default()
    io = DutyLoopback(calib.ids)
    bus = Bus(transport=io, discard_echo=False)
    rt = Mode2Runtime(bus, calib, log=lambda *_: None, sub_hz=400)
    rt.preflight()
    fb0 = rt.read()
    chk("in MODE 0 the loopback shows the offset centre", abs(fb0["fl_roll"]["q"]) < 1e-9)

    with rt:
        chk("enter switches every servo to MODE 2",
            all(io.get(i, R.MODE) == 2 for i in calib.ids))
        fb = rt.read()
        chk("... and the centre shift makes the raw count read as the same angle",
            abs(fb["fl_roll"]["q"]) < 1e-3, f" ({fb['fl_roll']['q']:.4f} rad)")
        rt.engage([0.0] * 12, ramp_s=0.1)
        chk("engage leaves torque on", rt.torque_on)
        target = [0.3] * 12
        rt.run(lambda dt, fb: target, seconds=seconds)
        fb = rt.read()
        err = max(abs(fb[n]["q"] - 0.3) for n in calib.joints)
        chk("the host loop drives the joints to the target", err < 0.02, f" ({err:.3f} rad)")
        chk("... on both signs of joint", abs(fb["fl_knee"]["q"] - 0.3) < 0.02
            and abs(fb["rr_knee"]["q"] - 0.3) < 0.02)
        chk("sub-ticks ran between the 50 Hz ticks", rt.sub_ticks > rt.ticks,
            f" ({rt.sub_ticks} in {rt.ticks} ticks)")
    chk("exit restores MODE 0", all(io.get(i, R.MODE) == 0 for i in calib.ids))
    chk("... duty 0", all(io.get(i, R.GOAL_TIME) == 0 for i in calib.ids))
    chk("... torque off", all(io.get(i, R.TORQUE_ENABLE) == 0 for i in calib.ids))
    chk("... and the position-mode servos are back", not isinstance(rt.servos["fl_roll"], RawServo))

    # the current fold: full duty below soft, the floor at hard
    rt2 = Mode2Runtime(Bus(transport=DutyLoopback(calib.ids), discard_echo=False), calib,
                       log=lambda *_: None)
    chk("fold is 1 below i_soft", rt2._fold(1.0, 1.4, 2.0) == 1.0)
    chk("fold is the floor at i_hard", rt2._fold(2.5, 1.4, 2.0) == rt2.i_floor)
    chk("fold is linear between", abs(rt2._fold(1.7, 1.4, 2.0) - (1 + rt2.i_floor) / 2) < 1e-9)

    # a calib sign only changes the joint frame: the loop still closes, both ways
    flipped = Calibration.default()
    flipped.sign["fl_knee"] = -1
    io4 = DutyLoopback(flipped.ids)
    rt4 = Mode2Runtime(Bus(transport=io4, discard_echo=False), flipped, log=lambda *_: None,
                       sub_hz=400)
    with rt4:
        rt4.engage([0.0] * 12, ramp_s=0.05)
        rt4.run(lambda dt, fb: [0.2] * 12, seconds=0.3)
        fb = rt4.read()
    chk("a flipped calib sign still tracks (it is a frame, not a runaway)",
        abs(fb["fl_knee"]["q"] - 0.2) < 0.02, f" ({fb['fl_knee']['q']:.3f} rad)")

    # a bridge that turns the count the other way IS a runaway, and it trips on the
    # sub-tick, not after the guard's 0.3 s hold
    calib3 = Calibration.default()
    io3 = DutyLoopback(calib3.ids, reversed_ids=[calib3.id["fl_knee"]])
    rt3 = Mode2Runtime(Bus(transport=io3, discard_echo=False), calib3, log=lambda *_: None,
                       sub_hz=400)
    tripped = None
    try:
        with rt3:
            rt3.engage([0.0] * 12, ramp_s=0.05)
            rt3.run(lambda dt, fb: [0.2] * 12, seconds=2.0)
    except Tripped as e:
        tripped = e
    chk("a reversed bridge trips as a runaway", tripped is not None and "runaway" in str(tripped))
    chk("... naming the joint", tripped is not None and tripped.joint == "fl_knee")
    chk("... in well under the guard's hold", rt3.ticks < 0.3 * CTRL_HZ, f" ({rt3.ticks} ticks)")
    chk("... and everything is off afterwards",
        all(io3.get(i, R.TORQUE_ENABLE) == 0 and io3.get(i, R.MODE) == 0 for i in calib3.ids))

    # the per-joint duty cap: a capped joint never exceeds it, an uncapped one still
    # reaches full duty on the same step, and the fold still applies under the cap
    joints = list(calib.joints)
    caps = parse_duty_caps("pitch=600,fl_roll=300", joints)
    chk("cap spec: a kind names every joint of that kind",
        all(caps[n] == 600 for n in joints if n.endswith("_pitch")) and len(caps) == 5)
    chk("... and a joint name names one", caps["fl_roll"] == 300)
    chk("... the tighter of two wins", parse_duty_caps("roll=500,fl_roll=800", joints)["fl_roll"] == 500)
    for bad in ("pitch", "hip=600", "pitch=0", "pitch=1001"):
        try:
            parse_duty_caps(bad, joints)
            chk(f"cap spec {bad!r} is refused", False)
        except ValueError:
            chk(f"cap spec {bad!r} is refused", True)
    chk("an empty spec is no caps", parse_duty_caps("", joints) == {})
    io5 = DutyLoopback(calib.ids)
    rt5 = Mode2Runtime(Bus(transport=io5, discard_echo=False), calib, log=lambda *_: None,
                       sub_hz=400, duty_cap=caps)
    with rt5:
        rt5.engage([0.0] * 12, ramp_s=0.05)
        rt5.run(lambda dt, fb: [0.5] * 12, seconds=0.3)       # a 0.5 rad step: full duty asked
    pk = rt5.peak_duty_joint
    chk("a capped joint never exceeds its cap", pk["fl_pitch"] <= 600 and pk["fl_roll"] <= 300,
        f" (fl_pitch {pk['fl_pitch']}, fl_roll {pk['fl_roll']})")
    chk("... and reaches it", pk["fl_pitch"] == 600 and pk["fl_roll"] == 300)
    chk("an uncapped joint still reaches full duty", pk["fl_knee"] == DUTY_MAX, f" ({pk['fl_knee']})")
    chk("the report names the caps", "duty caps" in rt5.report_lines() and "fl_pitch 600/600" in rt5.report_lines())
    chk("no caps, no line", "duty caps" not in rt.report_lines())

    print("mode2:", "ok" if ok else "FAILED")
    return 0 if ok else 1


# ============================================================== one servo, a bench
def _bench(a) -> int:
    """One free servo through the runtime: a sine, then the report. The gains and the
    bus rate against `bench/pwm_loop.py`, but through the real tick, the real fold and
    the real exit path."""
    bus = Bus(a.port, a.baud)
    if not bus.ping(a.id):
        print(f"no servo at id {a.id}")
        return 1
    s = Servo(bus, a.id)
    fb = s.feedback()
    calib = Calibration(["j"], ids={"j": a.id}, centre={"j": fb["counts"]},
                        sign={"j": +1}, soft={"j": a.amp + 0.3})
    rt = Mode2Runtime(bus, calib, hz=a.hz, kp=a.kp, kd=a.kd, kff=a.kff)
    print(f"id {a.id}: centre {fb['counts']}, {fb['volt']:.1f} V, {fb['temp']} C")
    log = []
    t = [0.0]

    def sine(dt, fb):
        t[0] += dt
        return [a.amp * math.sin(2 * math.pi * a.freq * t[0])]

    def tick(k, dt, fb):
        log.append((t[0], rt.goal["j"], fb["j"]["q"], fb["j"]["current"], rt.duty["j"]))

    try:
        with rt:
            rt.engage([0.0], ramp_s=0.5)
            rt.run(sine, seconds=a.seconds, on_tick=tick)
            rt.relax([0.0], ramp_s=0.5)
    except Tripped as e:
        print(f"!! TRIPPED: {e}")
    print(rt.report_lines())
    tt = [r for r in log if r[0] > 1.0 / a.freq]
    if tt:
        g = math.sqrt(sum(r[2] ** 2 for r in tt) / max(1e-9, sum(r[1] ** 2 for r in tt)))
        ipk = max(r[3] for r in tt)
        print(f"sine ±{math.degrees(a.amp):.0f}° at {a.freq:g} Hz: gain {g:.2f}, "
              f"peak current {ipk:.2f} A, peak duty {max(abs(r[4]) for r in tt)}")
    print(f"MODE now {bus.read(a.id, R.MODE)}, torque {bus.read(a.id, R.TORQUE_ENABLE)}")
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--sine", action="store_true", help="one FREE servo: a sine through the runtime")
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument("--id", type=int, default=1)
    ap.add_argument("--hz", type=float, default=CTRL_HZ)
    ap.add_argument("--kp", type=float, default=KP)
    ap.add_argument("--kd", type=float, default=KD)
    ap.add_argument("--kff", type=float, default=KFF)
    ap.add_argument("--amp", type=float, default=0.26, help="rad")
    ap.add_argument("--freq", type=float, default=2.0, help="Hz")
    ap.add_argument("--seconds", type=float, default=4.0)
    a = ap.parse_args()
    if a.sine:
        raise SystemExit(_bench(a))
    raise SystemExit(_selftest())
