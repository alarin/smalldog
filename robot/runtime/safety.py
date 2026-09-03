"""
safety.py — the limits, and the one place that decides to cut torque.

    python runtime/safety.py --selftest

Twelve ST3215 carrying 2.5 kg have four ways to hurt themselves, and the bus
reports all four in the same 15-byte read the loop already does for free
(`registers.FEEDBACK_LEN` — position, speed, load, voltage, temperature, current
in one round trip). So the guard costs nothing to run and there is no excuse for
a loop that does not have one.

  **Temperature.** The slow one, and the one that actually kills servos: a joint
  holding a stance at 2 kg drifts up over minutes with nothing dramatic to see.
  Tripped immediately when it is over — the signal is already an average, so
  waiting for it to persist is waiting twice.

  **Current.** The fast one. A stalled ST3215 draws 2.7 A (`robot/README.md`), and
  a leg that lands on the edge of a table gets there in one tick. But so does a
  hard footfall, briefly, which is the gait working — so this one is held: over
  the limit for `current_hold_s` continuously, not once.

  **Voltage.** Two different faults at two different speeds. Over `volt_max` is a
  supply set wrong and is tripped at once, before anything is asked to move.
  Under `volt_min` is a pack running out, and it sags at every footfall, so it is
  held. On a 3S pack 9.9 V is the bottom of the bench's own voltage sweep and the
  default sits just above it.

  **Tracking error.** Commanded minus measured, per joint. This is the one that
  catches what the other three miss: a leg jammed against the chassis, a hub that
  came loose, a sign that is wrong in `calib.json`. Held, because the servo is
  always behind a moving target — at 4 rad/s and 20 ms of loop, 0.08 rad of lag is
  simply the loop, and the limit has to sit well clear of it.

Plus the bus itself: `bus_fail` consecutive ticks with no usable feedback is a
disconnected cable or a dead adapter, and a controller running open loop into
twelve servos it can no longer hear is exactly the thing to stop.

What the guard does NOT do is decide how to stop. It raises `Tripped`; `loop.py`
owns the torque, because there is one owner or there are races.
"""
from __future__ import annotations

import argparse
import dataclasses
import math


class Tripped(RuntimeError):
    """A limit was exceeded. `loop.Runtime` turns this into torque off."""

    def __init__(self, reason, joint=None, value=None, limit=None):
        self.reason, self.joint, self.value, self.limit = reason, joint, value, limit
        where = f" on {joint}" if joint else ""
        num = "" if value is None else f": {value:.2f} against a limit of {limit:.2f}"
        super().__init__(f"{reason}{where}{num}")


@dataclasses.dataclass
class Limits:
    """Defaults are conservative and none of them is measured on this robot yet.

    The three that will want revisiting once the bench has run: `current_a` against
    what a real footfall draws, `q_err_rad` against the servo's actual lag at the
    gait's joint rates, and `volt_min` against the pack's sag under twelve servos.
    """
    temp_c: float = 65.0            # the servo's own MAX_TEMPERATURE default is ~70
    temp_warn_c: float = 55.0
    current_a: float = 2.0          # stall is 2.7 A at 12 V
    current_hold_s: float = 0.30
    volt_min: float = 9.5           # 3S nearly empty; the bench's lowest point is 9.9
    volt_max: float = 13.2          # 3S full is 12.6; higher is a supply set wrong
    volt_hold_s: float = 0.30
    q_err_rad: float = 0.35         # ~20 deg; the loop's own lag is a quarter of that
    q_err_hold_s: float = 0.30
    bus_fail: int = 5               # consecutive ticks with no usable feedback


class Guard:
    """Per-tick limit checking with per-joint accumulators.

    `update()` raises `Tripped` on the first tick a held condition has been true
    for long enough. It is deliberately stateful: every held limit needs to know
    how long it has been held, and putting that anywhere else means two objects
    that disagree about whether the robot is in trouble.
    """

    def __init__(self, joints, limits: Limits | None = None, log=print):
        self.joints = list(joints)
        self.lim = limits or Limits()
        self.log = log
        self.reset()

    def reset(self):
        self._hot = {n: 0.0 for n in self.joints}
        self._err = {n: 0.0 for n in self.joints}
        self._low_v = 0.0
        self._miss = 0
        self._warned = set()
        self.peak = dict(temp=0.0, current=0.0, q_err=0.0,
                         volt_min=math.inf, volt_max=0.0)

    # ------------------------------------------------------------------ warn
    def _warn(self, key, msg):
        """Say it once. A warning repeated at 50 Hz is a warning nobody reads."""
        if key not in self._warned:
            self._warned.add(key)
            self.log(f"!! {msg}")

    # ---------------------------------------------------------------- update
    def update(self, dt, feedback: dict, goal: dict | None = None):
        """One tick. `feedback` is {joint: decoded dict or None}, `goal` {joint: rad}."""
        lim, live = self.lim, 0
        for n in self.joints:
            fb = feedback.get(n)
            if fb is None:
                continue
            live += 1

            t = fb["temp"]
            self.peak["temp"] = max(self.peak["temp"], t)
            if t >= lim.temp_c:
                raise Tripped("over temperature", n, t, lim.temp_c)
            if t >= lim.temp_warn_c:
                self._warn(f"hot:{n}", f"{n} is at {t:.0f} C, {lim.temp_c:.0f} trips")

            i = fb["current"]
            self.peak["current"] = max(self.peak["current"], i)
            self._hot[n] = self._hot[n] + dt if i >= lim.current_a else 0.0
            if self._hot[n] >= lim.current_hold_s:
                raise Tripped(f"over current for {self._hot[n]:.2f} s", n, i, lim.current_a)

            v = fb["volt"]
            self.peak["volt_min"] = min(self.peak["volt_min"], v)
            self.peak["volt_max"] = max(self.peak["volt_max"], v)
            if v >= lim.volt_max:
                raise Tripped("over voltage", n, v, lim.volt_max)

            if goal is not None and n in goal:
                e = abs(goal[n] - fb["q"])
                self.peak["q_err"] = max(self.peak["q_err"], e)
                self._err[n] = self._err[n] + dt if e >= lim.q_err_rad else 0.0
                if self._err[n] >= lim.q_err_hold_s:
                    raise Tripped(f"not tracking for {self._err[n]:.2f} s — jammed, or a "
                                  f"sign is wrong in calib.json", n, e, lim.q_err_rad)

        # Undervoltage is a property of the pack, not of one servo: hold it on the
        # lowest reading of the tick so a single noisy frame does not start the clock.
        volts = [fb["volt"] for fb in feedback.values() if fb is not None]
        if volts:
            lo = min(volts)
            self._low_v = self._low_v + dt if lo <= lim.volt_min else 0.0
            if self._low_v >= lim.volt_hold_s:
                raise Tripped(f"under voltage for {self._low_v:.2f} s — the pack is done",
                              None, lo, lim.volt_min)

        self._miss = 0 if live else self._miss + 1
        if self._miss >= lim.bus_fail:
            raise Tripped(f"no feedback from any servo for {self._miss} ticks — "
                          f"check the bus, the adapter and the power")
        if live and live < len(self.joints):
            self._warn(f"partial:{len(self.joints) - live}",
                       f"only {live} of {len(self.joints)} servos answered")

    # ---------------------------------------------------------------- report
    def summary(self) -> dict:
        p = dict(self.peak)
        if p["volt_min"] is math.inf:
            p["volt_min"] = float("nan")
        return p

    def report(self) -> str:
        p = self.summary()
        return (f"peaks: {p['temp']:.0f} C, {p['current']:.2f} A, "
                f"{p['q_err']*57.3:.1f} deg tracking error, "
                f"{p['volt_min']:.1f}..{p['volt_max']:.1f} V")


# ============================================================== self-test
def _selftest() -> int:
    joints = ["a", "b"]
    ok = True

    def chk(name, cond):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {name}")

    def frame(**kw):
        f = dict(q=0.0, counts=2048, w=0.0, load=0, volt=12.0, temp=30.0, current=0.1)
        f.update(kw)
        return f

    def feed(g, n, dt=0.02, goal=None, **kw):
        """n ticks of the same frame on every joint; returns the Tripped or None."""
        try:
            for _ in range(n):
                g.update(dt, {j: frame(**kw) for j in joints}, goal)
        except Tripped as e:
            return e
        return None

    quiet = lambda *_: None

    g = Guard(joints, log=quiet)
    chk("nominal does not trip", feed(g, 500) is None)

    g = Guard(joints, log=quiet)
    chk("over temperature trips at once", feed(g, 1, temp=70.0) is not None)

    g = Guard(joints, log=quiet)
    chk("hot but not over does not trip", feed(g, 500, temp=60.0) is None)

    # current is held: 0.3 s at 50 Hz is 15 ticks, so 10 is not enough and 20 is
    g = Guard(joints, log=quiet)
    chk("a current spike does not trip", feed(g, 10, current=2.5) is None)
    chk("... but a sustained one does", feed(g, 10, current=2.5) is not None)

    g = Guard(joints, log=quiet)
    e = feed(g, 500, current=2.5)
    chk("current trip names the joint", e is not None and e.joint in joints)

    g = Guard(joints, log=quiet)
    chk("current resets when it drops",
        feed(g, 10, current=2.5) is None and feed(g, 10, current=0.1) is None
        and feed(g, 10, current=2.5) is None)

    g = Guard(joints, log=quiet)
    chk("over voltage trips at once", feed(g, 1, volt=16.0) is not None)

    g = Guard(joints, log=quiet)
    chk("a voltage sag does not trip", feed(g, 10, volt=9.0) is None)
    chk("... but a flat pack does", feed(g, 10, volt=9.0) is not None)

    g = Guard(joints, log=quiet)
    goal = {j: 0.0 for j in joints}
    chk("tracking within the band is fine", feed(g, 500, goal=goal, q=0.1) is None)
    g = Guard(joints, log=quiet)
    chk("a brief lag is fine", feed(g, 10, goal=goal, q=0.5) is None)
    chk("... a jam is not", feed(g, 10, goal=goal, q=0.5) is not None)

    g = Guard(joints, log=quiet)
    chk("no goal means no tracking check", feed(g, 500, q=3.0) is None)

    # a dead bus: every joint reports None
    g = Guard(joints, log=quiet)
    dead = {j: None for j in joints}
    try:
        for _ in range(4):
            g.update(0.02, dead)
        four = True
    except Tripped:
        four = False
    chk("four dead ticks are survivable", four)
    try:
        g.update(0.02, dead)
        five = False
    except Tripped:
        five = True
    chk("five are not", five)

    g = Guard(joints, log=quiet)
    for _ in range(4):
        g.update(0.02, dead)
    g.update(0.02, {j: frame() for j in joints})
    chk("one good tick resets the bus counter", feed(g, 4, ) is None)

    # partial answers warn once, not at 50 Hz
    said = []
    g = Guard(joints, log=said.append)
    for _ in range(100):
        g.update(0.02, {"a": frame(), "b": None})
    chk("a partial bus warns exactly once", len(said) == 1)

    g = Guard(joints, log=quiet)
    feed(g, 5, temp=40.0, current=1.0, volt=11.5, goal=goal, q=0.05)
    p = g.summary()
    chk("peaks are recorded", p["temp"] == 40.0 and abs(p["current"] - 1.0) < 1e-9
        and abs(p["volt_min"] - 11.5) < 1e-9 and abs(p["q_err"] - 0.05) < 1e-9)

    print("safety:", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selftest", action="store_true")
    ap.parse_args()
    raise SystemExit(_selftest())
