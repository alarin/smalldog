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
import collections
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
    #: Temperature is HELD and MEDIAN-FILTERED, and that is not caution, it is a
    #: measurement. `PRESENT_TEMPERATURE` is clean to +-1 C while a servo is still
    #: and unreliable the moment its motor drives: on 2026-09-07, fl_pitch sitting
    #: at a true 30 C returned isolated samples of 41, 48, 54, 56 and once 98 while
    #: it moved, on both the SyncRead and the single-read path, with the voltage
    #: byte beside it correct in every frame. It is PWM noise reaching the ADC.
    #: This tripped the very first --stand: `over temperature on fl_pitch: 65.00`
    #: on a servo that a direct read showed at 30 C a second later. Temperature is
    #: the slowest thing the guard watches — a 55 g rotor cannot move 30 C in 20 ms
    #: — so a single sample over the line is evidence of noise, not of heat, and
    #: every other held limit here already knew that.
    temp_hold_s: float = 0.50
    #: Odd, so the median is a sample and not a mean. NINE, not five, and the
    #: difference is measured: over a 20 s trot on all twelve servos, 12000
    #: samples carried 102 elevated stretches and **every one of them was a single
    #: sample long** — the noise is isolated spikes, not sustained error. A median
    #: of five is fooled only when three spikes land inside one five-window, which
    #: at that rate is roughly a one-in-ten event over a 30 s run; it duly happened,
    #: and the first --baseline reported a 44 C peak on servos a rest reading put at
    #: 30. Nine needs five spikes in nine samples, which is not going to happen, and
    #: costs 0.18 s of lag on a quantity that moves over tens of seconds.
    temp_median_n: int = 9
    temp_spike_c: float = 8.0       # raw minus median above this is counted, not acted on
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
        self._hot_t = {n: 0.0 for n in self.joints}
        self._temps = {n: collections.deque(maxlen=self.lim.temp_median_n)
                       for n in self.joints}
        self._err = {n: 0.0 for n in self.joints}
        self._low_v = 0.0
        self._miss = 0
        self._warned = set()
        #: `temp` is the filtered figure — the robot's actual temperature. `temp_raw`
        #: is the highest single byte any servo returned, spikes included, so the
        #: noise stays visible instead of being quietly smoothed out of the report.
        self.temp_spikes = 0
        self.peak = dict(temp=0.0, temp_raw=0.0, current=0.0, q_err=0.0,
                         volt_min=math.inf, volt_max=0.0)

    # ------------------------------------------------------------------ warn
    def _warn(self, key, msg):
        """Say it once. A warning repeated at 50 Hz is a warning nobody reads."""
        if key not in self._warned:
            self._warned.add(key)
            self.log(f"!! {msg}")

    # ------------------------------------------------------------- at rest
    def check_at_rest(self, feedback: dict):
        """Temperature, judged on a single reading, before torque comes on.

        `update` filters and holds temperature because PRESENT_TEMPERATURE is only
        noisy while a motor drives — see `Limits.temp_hold_s`. Nothing is driving
        here, and a still servo's byte is good to +-1 C: thousands of samples on a
        stationary ST3215 produced not one outlier, while the same servo moving
        produced bytes as high as 150. So this is the one place a single reading is
        allowed to refuse. A servo that is genuinely hot must not be asked to take
        the weight, and half a second of filtering to establish what a stationary
        robot could say immediately is half a second of putting torque into it.
        """
        for n in self.joints:
            fb = feedback.get(n)
            if fb is not None and fb["temp"] >= self.lim.temp_c:
                raise Tripped("over temperature before torque",
                              n, fb["temp"], self.lim.temp_c)

    # ---------------------------------------------------------------- update
    def update(self, dt, feedback: dict, goal: dict | None = None):
        """One tick. `feedback` is {joint: decoded dict or None}, `goal` {joint: rad}."""
        lim, live = self.lim, 0
        for n in self.joints:
            fb = feedback.get(n)
            if fb is None:
                continue
            live += 1

            # Median of the last few samples, then held: see Limits.temp_hold_s for
            # why one hot byte is noise rather than news.
            raw_t = fb["temp"]
            self.peak["temp_raw"] = max(self.peak["temp_raw"], raw_t)
            d = self._temps[n]
            d.append(raw_t)
            # Nothing is judged until the window is full. A partly-filled window has
            # no median worth the name — on the first tick it IS whatever single
            # sample arrived, spike and all, which is precisely how a startup spike
            # would walk past the filter it was built to stop. Four ticks is 80 ms
            # of not looking, against a hold of 500 ms, and temperature is the one
            # quantity slow enough that the wait cannot hide anything.
            if len(d) == d.maxlen:
                t = sorted(d)[len(d) // 2]
                if raw_t - t >= lim.temp_spike_c:
                    self.temp_spikes += 1
                self.peak["temp"] = max(self.peak["temp"], t)
                self._hot_t[n] = self._hot_t[n] + dt if t >= lim.temp_c else 0.0
                if self._hot_t[n] >= lim.temp_hold_s:
                    raise Tripped(f"over temperature for {self._hot_t[n]:.2f} s",
                                  n, t, lim.temp_c)
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
        s = (f"peaks: {p['temp']:.0f} C, {p['current']:.2f} A, "
             f"{p['q_err']*57.3:.1f} deg tracking error, "
             f"{p['volt_min']:.1f}..{p['volt_max']:.1f} V")
        if self.temp_spikes:
            s += (f"\n  {self.temp_spikes} temperature spikes discarded "
                  f"(raw max {p['temp_raw']:.0f} C) — see Limits.temp_hold_s")
        return s


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

    # Temperature is held (0.5 s = 25 ticks at 50 Hz) and median-filtered, because
    # PRESENT_TEMPERATURE is noisy while the motor drives. These four cases are the
    # real 2026-09-07 failure, not a hypothetical: a servo at a true 30 C returned
    # isolated bytes as high as 98 and tripped --stand at 65.00.
    g = Guard(joints, log=quiet)
    chk("a single hot sample does not trip", feed(g, 1, temp=98.0) is None)

    # ... but before torque, on a robot that is not driving, one reading is enough
    g = Guard(joints, log=quiet)
    hot = {j: frame(temp=70.0) for j in joints}
    try:
        g.check_at_rest(hot)
        chk("at rest, one hot reading does refuse", False)
    except Tripped:
        chk("at rest, one hot reading does refuse", True)
    g = Guard(joints, log=quiet)
    g.check_at_rest({j: frame(temp=30.0) for j in joints})
    chk("at rest, a cold robot is fine", True)

    g = Guard(joints, log=quiet)
    chk("... nor a burst shorter than the hold", feed(g, 10, temp=70.0) is None)

    g = Guard(joints, log=quiet)
    chk("sustained over temperature still trips", feed(g, 40, temp=70.0) is not None)

    g = Guard(joints, log=quiet)
    trip = None
    for i in range(500):                       # one 98 C spike every 17 ticks, 30 C otherwise
        try:
            g.update(0.02, {j: frame(temp=98.0 if i % 17 == 0 else 30.0) for j in joints})
        except Tripped as e:
            trip = e
            break
    chk("isolated spikes never trip", trip is None)
    chk("... the filtered peak stays honest", g.summary()["temp"] == 30.0)
    chk("... the raw spike is still reported", g.summary()["temp_raw"] == 98.0)
    chk("... and the spikes are counted", g.temp_spikes > 0)

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
    # more ticks than temp_median_n: the temperature peak is only recorded once the
    # median window is full, so a run shorter than the warm-up reports no peak at all
    feed(g, 12, temp=40.0, current=1.0, volt=11.5, goal=goal, q=0.05)
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
