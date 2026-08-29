"""Foot contact from the servos, with no foot sensor in the robot.

The ST3215 reports Position, Load, Speed and Voltage over its bus, so every leg already
has a load signal.  Raw, it says almost nothing about contact — over a trot the knee's
load is dominated by the gait phase, not by what is under the foot (measured: AUC 0.37 to
0.57 against the true contact, which is chance).  Subtract what the *same leg at the same
phase* reads with nothing under it and the signal appears: AUC 0.86, and driving the
gait's terrain feedback from it is as good as a perfect foot sensor.

    12 terrain seeds, 5 s trot at 0.20 m/s, flat-ground trot for reference

    contact source                        flat     terrain     tilt med/worst   fell
    none (open loop)                      793 mm   625+-56 mm    13.5 / 180.0    1/12
    ideal foot sensor (MuJoCo touch)      758 mm   618+-53 mm     7.2 /  13.6    0/12
    knee load - baseline, 0.5 Nm          758 mm   647+-36 mm     6.0 /  12.7    0/12

This works because the gait wants the *touchdown edge* inside a swing, debounced over a
few ms — not a calibrated force.  That is a much weaker thing to ask of a noisy signal.

Two things this module cannot tell you, both measured in MuJoCo where the "load" is a
clean kp*err - kv*qvel:

  * Feetech's Present Load is a PWM-duty proxy, not a torque: quantised, signed, with a
    deadband and stiction, and in its own units.  The result above says the information
    is there in principle; it does not say the number the servo reports carries it.
    Re-find the threshold on hardware, in whatever units the bus gives you.
  * The bus is half duplex.  Reading four knee loads per 100 Hz tick costs four round
    trips on top of the position writes, and a late reading is worse than none: a 20 ms
    delay is a fifth of the swing.  Measure that before building on this.

The baseline is specific to the gait that produced it — period, swing height, body height
and commanded speed all shape the in-air load curve — so `Baseline` stamps those and
`mismatch()` reports what has drifted since.
"""
import json
import math

LEGS = ("fl", "fr", "rl", "rr")
NBIN = 60                      # phase bins over one gait cycle; 60 is ~7.5 ms at 0.45 s


class Baseline:
    """Per-leg, per-phase load of a leg swinging with nothing under the foot.

    Fill it by feeding samples from a robot that is *hanging* — lift it, let the gait run,
    call `add()` every tick — then `save()`. `residual()` is what the estimator thresholds.
    """

    def __init__(self, nbin=NBIN, legs=LEGS, gait=None):
        self.nbin = int(nbin)
        self.legs = tuple(legs)
        self.gait = dict(gait or {})          # the gait settings this was recorded under
        self._sum = {l: [0.0] * self.nbin for l in self.legs}
        self._n = {l: [0] * self.nbin for l in self.legs}
        self.curve = {l: [0.0] * self.nbin for l in self.legs}

    # ---------------------------------------------------------------- recording
    def add(self, leg, phase, load):
        b = self._bin(phase)
        self._sum[leg][b] += float(load)
        self._n[leg][b] += 1

    def finish(self):
        """average the samples into `curve`, filling any empty bin from its neighbours"""
        for l in self.legs:
            got = [i for i in range(self.nbin) if self._n[l][i]]
            if not got:
                raise ValueError(f"no samples for {l}: was the gait running?")
            for i in range(self.nbin):
                if self._n[l][i]:
                    self.curve[l][i] = self._sum[l][i] / self._n[l][i]
                else:                          # nearest recorded bin, wrapping
                    j = min(got, key=lambda g: min(abs(g - i), self.nbin - abs(g - i)))
                    self.curve[l][i] = self._sum[l][j] / self._n[l][j]
        return self

    def coverage(self):
        """fraction of bins that got a real sample, per leg — a thin run shows up here"""
        return {l: sum(1 for v in self._n[l] if v) / self.nbin for l in self.legs}

    # ---------------------------------------------------------------- use
    def _bin(self, phase):
        return min(self.nbin - 1, max(0, int((phase % 1.0) * self.nbin)))

    def residual(self, leg, phase, load):
        """load minus what this leg reads at this phase in free air"""
        return float(load) - self.curve[leg][self._bin(phase)]

    def mismatch(self, gait):
        """which recorded gait settings the live gait no longer agrees with.

        The curve is the load of a specific trajectory. Change the period, the swing, the
        body height or the speed it was recorded at and it is measuring something else.
        """
        out = {}
        for k, was in self.gait.items():
            now = gait.get(k)
            if now is None or not isinstance(was, (int, float)):
                continue
            if abs(now - was) > 1e-6 + 0.02 * abs(was):
                out[k] = (was, now)
        return out

    # ---------------------------------------------------------------- io
    def save(self, path):
        with open(path, "w") as f:
            json.dump(dict(nbin=self.nbin, legs=list(self.legs), gait=self.gait,
                           curve=self.curve), f, indent=1)
        return path

    @classmethod
    def load(cls, path):
        with open(path) as f:
            d = json.load(f)
        b = cls(nbin=d["nbin"], legs=d["legs"], gait=d.get("gait"))
        b.curve = {l: list(v) for l, v in d["curve"].items()}
        return b


class ServoContact:
    """Baseline + threshold + debounce -> {leg: bool}, ready for TrotGait.feedback().

    `sign` is which way contact pushes the residual: -1 for a knee load that *drops* when
    the foot is loaded (what the ST3215 knee reads, and what the numbers above are for),
    +1 for a sensor that rises, e.g. an FSR in the foot.
    """

    def __init__(self, baseline, threshold=0.5, sign=-1, hold=0.004, legs=LEGS):
        self.baseline = baseline
        self.threshold = float(threshold)
        self.sign = -1 if sign < 0 else 1
        self.hold = float(hold)                # s the residual must stay over threshold
        self.legs = tuple(legs)
        self._over = {l: 0.0 for l in self.legs}
        self._state = {l: False for l in self.legs}

    def update(self, dt, phase, load):
        """phase and load are {leg: value}; returns {leg: bool}.

        The debounce is one-sided on purpose: it delays a rising edge by `hold` and lets
        the falling edge through at once. A late touchdown costs the gait a few mm of
        latch height; a touchdown that never arrives costs it the whole footfall.
        """
        for l in self.legs:
            if l not in load or l not in phase:
                continue
            r = self.baseline.residual(l, phase[l], load[l]) * self.sign
            if r > self.threshold:
                self._over[l] += dt
                if self._over[l] >= self.hold:
                    self._state[l] = True
            else:
                self._over[l] = 0.0
                self._state[l] = False
        return dict(self._state)

    def residuals(self, phase, load):
        """the raw residuals, sign applied — publish these and let the far end threshold"""
        return {l: self.baseline.residual(l, phase[l], load[l]) * self.sign
                for l in self.legs if l in load and l in phase}
