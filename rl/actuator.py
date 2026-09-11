"""
actuator.py — the ST3215 as a voltage-driven machine, not an ideal PD.

The model, and why each term is in it
-------------------------------------
A position-mode bus servo is a closed loop we do not own: it reads its own
encoder, computes a PWM duty from its own registers, and drives a brushed motor
through a 1:345 gearbox. Everything the robot experiences — the torque ceiling
falling as the pack drains, the speed limit, the stiction near a reversal, the
dead angle at zero crossing — comes out of that chain. An ideal PD in MuJoCo
reproduces none of it, and a policy trained against one learns to spend torque
and speed the servo does not have.

So the chain is modelled as it is, in five stages:

  1. inner loop     duty = kp*err + kd*(-w), on the QUANTISED encoder reading,
                    with the servo's own dead zone and minimum-startup-force
                    ("punch") registers, saturating at +-1.
  2. voltage        U = duty * U_bat.  This is the whole point: U_bat is a 3S
                    pack under load, not a constant, so torque and no-load speed
                    fall together as it drains.
  3. electrical     i = (U - k_e*w) / R.  The back-EMF term is what makes the
                    servo's speed limit emerge instead of being clamped on.
  4. mechanical     tau = k_u*U - k_w*w - friction, driving the reflected rotor
                    inertia J_m, which for 1:345 is ~151x the knee link's own
                    inertia (measured: rl/checks/check_model.py prints the
                    ratio; it read 73x against an earlier armature). The friction
                    is (tau_c + mu_load*|tau_transmitted|)*sign(w) + b_v*w, and
                    the middle term is not a refinement: on this gearbox it is
                    0.28 N*m per N*m carried against a 0.19 N*m floor, so at the
                    bench arm's 0.88 N*m it is half the load again. Measured
                    2026-09-09; see friction() for how, and for why there is no
                    forward/back-driving asymmetry beside it.
  5. transmission   a deadzone spring of width theta_bl between motor and output.
                    Below it no torque crosses at all.

Identifiability
---------------
k_u, k_w, k_e and R are not independent — k_u = eta*k_e/R and k_w = k_u*k_e —
and from position data alone at a single voltage, the back-EMF damping k_w and
the viscous friction b_v are the same column of the regressor. They separate on
two things and only those two: measured CURRENT (which sees k_e and R but not
b_v) and runs at SEVERAL SUPPLY VOLTAGES. That is why robot/bench/sweep.py
insists on both, and why a fit from one voltage with no ammeter is refused
rather than reported with wide error bars.

**The b_v / k_e half of that is now known to be a dead end on this bench, and the
voltage sweep does not rescue it.** Both terms cost a motor voltage proportional
to omega and NEITHER depends on supply, so eight volts says exactly what twelve
does. What the speed ladder measures cleanly is their SUM, 1.37 N*m*s/rad to
5 % over three voltages, of which back-EMF is 1.31 — leaving 0.06 +- 0.07 for
b_v, i.e. consistent with zero. Use the sum. Splitting it needs a current
measurement finer than PRESENT_CURRENT's 6.5 mA LSB divided by a duty.

**The friction terms are immune to the register-scale problem below, and that is
worth knowing before trusting them.** tau_c and mu_load are measured as ratios
between two quantities in the same PRESENT_LOAD units, converted to N*m through
a gravity anchor that is known exactly, so any constant mis-scaling of that
register cancels out of both. k_u is not so lucky: friction-cancelled holds put
the torque constant at 2.39 N*m/A through the current channel and 2.38 through
the duty channel — agreeing to 0.3 % with each other and disagreeing by 2.2x
with the vendor's 1.09. See PLAN.md step 2c; it is not friction, because
removing friction makes it worse rather than better.

Where the encoder sits
----------------------
`enc_after_backlash` decides whether the servo's control loop — and the position
we read back over the bus — sees the output shaft (after the gearbox play) or
the motor side. The vendor wiki in 3d/ref/st3215_wiki.html says "360 degree
magnetic encoder ... 360/4096", i.e. one absolute turn of the OUTPUT, which only
works with the sensor on the output shaft.

**CONFIRMED on the bench, 2026-09-08: the encoder is AFTER the gearbox.** Torque
off, rocking the horn against the play moves Present Position. So the servo reads
the true joint angle, and the backlash is a hole in the TORQUE path, not in the
measurement — the robot can observe its own play. `True` below is now measured,
not inherited from a wiki. The answer changes the observation wiring in
step 4, so it is a parameter and not an assumption baked into the equations.

Units are SI at the joint: rad, rad/s, N*m, V, A.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os

import numpy as np

ENC_STEPS = 4096                      # counts per output turn, vendor spec
ENC_STEP_RAD = 2.0 * math.pi / ENC_STEPS


@dataclasses.dataclass
class Params:
    """One servo's parameters. Everything here is meant to be FITTED.

    The defaults are priors derived from the vendor sheet, not measurements, and
    `fitted=False` marks them as such — `load()` says so out loud, because a
    policy trained on vendor priors and deployed on a real servo is exactly the
    failure this whole file exists to avoid.
    """
    # --- electrical -------------------------------------------------------
    # The three vendor numbers pin these exactly, and are worth doing out loud
    # because they also expose how lossy this gearbox is:
    #   locked rotor 2.7 A @ 12 V  ->  R   = 12 / 2.7   = 4.44 ohm
    #   no load 4.71 rad/s @ 12 V  ->  k_e = 12 / 4.71  = 2.55 V*s/rad
    #       (MEASURED 2026-09-11: 3.86 on the free hub, and it is a PLATEAU
    #        at TORQUE_LIMIT >= ~740, not d*U/k_e - the rungs below it give
    #        k_e = 2.32.  The law has no such plateau; PLAN.md 3c says what
    #        would add one and what decides its form.)
    #   stall 2.94 N*m @ 12 V      ->  k_u = 2.94 / 12  = 0.245 N*m/V
    # and then eta = k_u*R/k_e = 0.43. A 43 %-efficient drivetrain is low even
    # for a 1:345 stack of spur gears; either the gearbox really is that lossy
    # or one of the three specs is optimistic. Both possibilities are reasons to
    # fit rather than compute, which is what this file is for.
    R: float = 4.44           # ohm, motor + wiring, at the terminals
    k_e: float = 2.55         # V*s/rad at the JOINT (= n*kt)
    k_u: float = 0.245        # N*m per volt at zero speed (= eta*n*kt/R)
    # --- mechanical -------------------------------------------------------
    J_m: float = 0.008        # kg*m^2, reflected rotor+gearbox inertia at the joint
    J_l: float = 0.0          # kg*m^2, load-side inertia the servo carries itself
    tau_c: float = 0.05       # N*m, Coulomb friction at zero load
    b_v: float = 0.010        # N*m*s/rad, viscous friction
    # Friction that GROWS WITH THE TORQUE BEING TRANSMITTED, as gear-tooth
    # normal forces do.  N*m of friction per N*m through the gearbox.  Unlike
    # its neighbours this default is a MEASUREMENT, not a vendor prior: the
    # bidirectional hold ladder (robot/bench/sweep.py --traj holdbi, read by
    # bench/hysteresis.py) puts it at 0.25-0.30 across 8/10/12 V.  Zero here
    # would silently restore the defect this term exists to fix, so the measured
    # value is the default and `fitted` still says whether a fit has run.
    mu_load: float = 0.28
    # --- inner loop (registers 21/22, 26/27, 24) --------------------------
    kp: float = 32.0          # duty per rad of error, after the register scaling
    kd: float = 0.0           # duty per rad/s
    deadband: float = 2 * ENC_STEP_RAD    # CW/CCW dead zone registers
    punch: float = 0.0        # minimum startup duty once outside the dead zone
    duty_max: float = 1.0
    loop_hz: float = 1000.0   # the servo's own loop rate
    # --- transmission -----------------------------------------------------
    theta_bl: float = math.radians(0.5)   # total backlash, rad at the output
    k_bl: float = 3000.0      # N*m/rad once engaged; stiff, not identified
    c_bl: float = 1.0         # N*m*s/rad, engaged damping; numerical, not physics
    enc_after_backlash: bool = True
    # --- numerics ---------------------------------------------------------
    v_eps: float = 0.02       # rad/s, tanh width standing in for sign(); a
    #                           numerical device, never a physical parameter
    # --- provenance -------------------------------------------------------
    fitted: bool = False
    source: str = "vendor spec (2.94 N*m, 4.71 rad/s, 2.7 A stall, all @ 12 V); NOT fitted"
    servo_ids: tuple = ()
    rms_pos_deg: float = float("nan")
    rms_current_a: float = float("nan")

    # -- derived, for reporting -------------------------------------------
    @property
    def k_w(self) -> float:
        """Back-EMF torque coefficient, N*m*s/rad. Not free: k_u * k_e."""
        return self.k_u * self.k_e

    def stall_torque(self, u_bat: float) -> float:
        return self.k_u * u_bat

    def no_load_speed(self, u_bat: float) -> float:
        return u_bat / self.k_e

    def to_json(self, path: str) -> None:
        d = dataclasses.asdict(self)
        d["servo_ids"] = list(self.servo_ids)
        d["_derived"] = {
            "k_w_Nm_s_per_rad": self.k_w,
            "stall_Nm_at_12V": self.stall_torque(12.0),
            "stall_Nm_at_9.9V": self.stall_torque(9.9),
            "no_load_rad_s_at_12V": self.no_load_speed(12.0),
            "backlash_deg": math.degrees(self.theta_bl),
        }
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Params":
        d = {k: v for k, v in d.items() if not k.startswith("_")}
        if "servo_ids" in d:
            d["servo_ids"] = tuple(d["servo_ids"])
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})


# A Params of scalars is a container of Python floats; a Params whose fields are
# arrays is a batch of servos. The second only survives a jax transform if jax
# knows which fields are data and which are provenance — without this, `vmap`
# over a Params is a silent no-op that hands every environment the SAME servo,
# and domain randomisation looks like it is working while doing nothing.
#
# Registered behind a try/except on purpose: the mac's check runs and the Orange
# Pi import this file with no jax installed (rl/CLAUDE.md), and that has to keep
# working. source/servo_ids/fitted are str/tuple/bool and cannot be traced;
# rms_* are how well a fit went, not state.
try:
    import jax as _jax

    _jax.tree_util.register_dataclass(
        Params,
        data_fields=["R", "k_e", "k_u", "J_m", "J_l", "tau_c", "b_v", "mu_load",
                     "kp", "kd", "deadband", "punch", "duty_max", "loop_hz",
                     "theta_bl", "k_bl", "c_bl", "v_eps"],
        meta_fields=["enc_after_backlash", "fitted", "source", "servo_ids",
                     "rms_pos_deg", "rms_current_a"])
except ImportError:                       # numpy-only machine; nothing to register
    pass


DEFAULT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "params", "st3215.json")


def load(path: str = DEFAULT_PATH, quiet: bool = False) -> Params:
    with open(path) as f:
        p = Params.from_dict(json.load(f))
    if not p.fitted and not quiet:
        print(f"!! {os.path.relpath(path)} is NOT a fit: {p.source}\n"
              f"!! Training against this trains against the datasheet. Run\n"
              f"!! robot/bench/sweep.py and robot/bench/fit_bam.py first.")
    return p


# ======================================================================= law
def _sign(w, v_eps, xp=np):
    """tanh in place of sign(): least_squares needs a derivative, and a true
    sign() makes the Coulomb term a wall the optimiser cannot see across."""
    return xp.tanh(w / v_eps)


def duty(p: Params, err: float, w: float, xp=np) -> float:
    """The servo's inner loop, on a quantised error, in [-duty_max, duty_max].

    The dead zone and the punch are real registers (26/27 and 24) and they are
    what makes a bus servo buzz around its target instead of settling: inside the
    dead zone there is no drive at all, and the first drive outside it is not
    infinitesimal but `punch`.
    """
    # NOTE: round() has zero gradient almost everywhere. PPO never
    # differentiates through the sim so this costs nothing here, but it does
    # make this law unusable for anything that wants dynamics gradients.
    e = xp.round(err / ENC_STEP_RAD) * ENC_STEP_RAD          # the encoder's view
    e = xp.where(xp.abs(e) <= p.deadband, 0.0, e)
    u = p.kp * e - p.kd * w
    u = xp.where((u != 0.0) & (xp.abs(u) < p.punch), xp.sign(u) * p.punch, u)
    return xp.clip(u, -p.duty_max, p.duty_max)


def current(p: Params, u_volt: float, w: float, driven=True, xp=np) -> float:
    """Motor current, A. i = (U - back-EMF) / R — the channel that separates the
    electrical damping from the mechanical, and the only reason the fit closes.

    `driven=False` is Torque Enable = 0: the bridge is off, the winding is open,
    and there is no current at all whatever the shaft is doing.
    """
    return xp.where(driven, (u_volt - p.k_e * w) / p.R, 0.0)


def friction(p: Params, w: float, tau_t: float, xp=np) -> float:
    """Gearbox friction: a constant, a term that grows with LOAD, and a viscous one.

    The middle term is the one most models leave out, and leaving it out is not a
    small approximation on a 1:345 spur stack. Gear-tooth friction is proportional
    to the normal force between teeth, which is proportional to the torque being
    transmitted, so friction is not a property of the gearbox alone — it is a
    property of the gearbox and its present load. Measured on this servo
    (robot/bench/sweep.py --traj holdbi, three voltages): 0.19 N*m at no load,
    growing by 0.28 N*m per N*m carried, so at the 0.88 N*m the bench arm asks
    for, friction is 0.41 N*m — half the load again.

    Until this term existed the fit had nowhere to put that and put it in `k_u`,
    which is why the identified stall was 4.23 N*m against a spec 2.94, and why
    `b_v` sat pinned on its lower bound however much data it was given.

    **There is deliberately no forward/back-driving asymmetry here**, and that is
    a conclusion rather than an omission. The bench reports ~28 % of motor torque
    lost forward-driving against ~62 % back-driving, which reads as a strongly
    asymmetric gearbox — but almost all of that gap is `tau_c` being a FIXED
    offset measured against two very different load levels. Run the symmetric law
    above at each: forward, 0.19 + 0.28*0.88 = 0.41 N*m out of the 1.29 the motor
    supplies, i.e. 32 %; back-driving over a free swing's mean gravity torque,
    56 %. Both land on the reported numbers without a second coefficient, and a
    second coefficient fitted to the residual of the first would not be
    identifiable from this bench anyway.
    """
    return ((p.tau_c + p.mu_load * xp.abs(tau_t)) * _sign(w, p.v_eps, xp)
            + p.b_v * w)


def motor_torque(p: Params, u_volt: float, w: float, driven=True, tau_t=None,
                 xp=np) -> float:
    """Torque at the joint before the transmission.

    The back-EMF term is gated by `driven`, and the distinction is not
    bookkeeping. With the bridge enabled and zero duty the winding is shorted
    through the low-side switches, so the motor brakes: k_w applies, and at
    k_w = k_u*k_e = 0.62 N*m*s/rad it dominates everything mechanical. With
    Torque Enable = 0 the bridge is off, the winding is open, no current flows
    and the shaft coasts against friction alone.

    That is precisely why the free-swing run identifies what it does: releasing
    the torque removes the electrical damping from the equation and leaves
    J, tau_c and b_v alone in it. Model the release as a braked motor and the
    fit reads the electrical damping as mechanical friction — thirty times too
    much of it — and every later trajectory inherits the error.

    `tau_t` is the torque actually crossing the gearbox, which is what the
    load-dependent friction scales with. `simulate()` below has it as a state and
    passes it. Callers that do not — rl/env/walk.py and rl/eval.py, where MuJoCo
    owns the transmission — leave it None and get the PROXY below, and the
    conversion in it is exact rather than a fudge: forward-driving at steady
    state the motor supplies load + friction, so tau_drive = tau_t*(1 + mu_load),
    and dividing it back out recovers the transmitted torque. Where the proxy is
    wrong is where those two disagree in SIGN — the load driving the motor with
    the bridge off, i.e. a free swing, where tau_drive is zero and the gearbox is
    still carrying the whole load. That case only arises on the bench, and on the
    bench `simulate()` passes the real value.
    """
    if tau_t is None:
        tau_t = p.k_u * u_volt / (1.0 + p.mu_load)
    return (p.k_u * u_volt - xp.where(driven, p.k_w * w, 0.0)
            - friction(p, w, tau_t, xp))


def bus_torque(p: Params, err, w, u_bat, sag, xp=np):
    """The whole chain a ROBOT joint sees: inner loop, pack sag, motor torque.

    One statement of it, because it was written out three times — rl/env/walk.py
    inside the physics scan, rl/eval.py's vanilla-MuJoCo pass, and
    rl/checks/check_model.py's probe that every randomised servo parameter is
    actually consumed. The probe is the reason it matters that they are the same
    arithmetic: a copy that drops `R` reads `R` as unconsumed and says so.

    `err`, `w` are per joint and may be arrays; `u_bat` and `sag` are per ROBOT,
    because there is one pack and one harness. That is why the sag is applied to
    the SUMMED current and not to each joint's own — see simulate()'s `sag` note
    for the factor of twelve between the two conventions.

    The clamp on `volt` is not cosmetic, and the arithmetic is worth writing down
    because it is reachable rather than hypothetical: at duty 1 and a joint
    running backwards at 10 rad/s, i = (12 + 3.06*10)/3.33 = 12.8 A per servo,
    and twelve of those into the original 0.18 ohm range is a 27.7 V sag on a
    12 V pack. `volt` would go NEGATIVE, k_u*duty*volt would flip sign, and the
    torque would drive the joint harder in the direction it was already going —
    positive feedback. A discharged pack delivers less voltage; it never delivers
    negative voltage. (The range has since been narrowed to 0-0.06 ohm as well,
    which makes the clamp unreachable in normal operation. Both, not either: a
    floor that is only satisfied by accident is not a floor.)
    """
    d = duty(p, err, w, xp=xp)
    i = current(p, d * u_bat, w, xp=xp)
    volt = xp.clip(u_bat - sag * xp.sum(xp.abs(i)), 0.0, u_bat)
    return motor_torque(p, d * volt, w, xp=xp)


def transmitted(p: Params, delta: float, dw: float, xp=np) -> float:
    """Torque across the gearbox play. Zero inside +-theta_bl/2, spring outside.

    This is what makes the encoder useless as a backlash sensor in the direction
    people expect: with the sensor on the output the servo measures the true
    joint angle, but there is a band in which its motor can move and the joint
    cannot be made to follow. The information is missing from the TORQUE path,
    not from the measurement.
    """
    half = 0.5 * p.theta_bl
    engaged = xp.abs(delta) > half
    x = xp.where(delta > half, delta - half,
                 xp.where(delta < -half, delta + half, 0.0))
    return p.k_bl * x + xp.where(engaged, p.c_bl * dw, 0.0)


# ================================================================= dynamics
def simulate(p: Params, target, dt, q0=0.0, w0=0.0, u_bat=12.0, load_torque=None,
             sag=0.0, dt_int=1e-4, torque_on=None):
    """Run the servo against a target trajectory. Returns the bench observables.

    target        array of joint targets, one per sample of length dt
    load_torque   f(q) -> N*m acting ON the load, e.g. a pendulum arm's gravity
    u_bat         supply volts, scalar or per-sample
    sag           ohms of supply resistance: U_bat drops by sag*|i| under load,
                  which is the pack sagging and is a real randomisation axis.
                  NOTE the convention, and the factor between the two uses of
                  it. Here |i| is THIS servo's current, because this function
                  simulates one servo. rl/env/walk.py applies the same law to
                  the SUM over all twelve, because there the load case is one
                  pack and one harness feeding the whole robot. Both are ohms,
                  but they are not the same ohms: the same numeric value drops
                  TWELVE TIMES more voltage in walk.py than it does here, so a
                  value fitted against this function must be divided by the
                  number of servos drawing at once before it means anything
                  there, and vice versa. rl/params/domain_rand.json's 0-0.06 is
                  the summed-current convention. Do not copy a number across
                  without doing that arithmetic.
    torque_on     per-sample bool; False drives the duty to zero AND opens the
                  winding, which is what Torque Enable = 0 does. The bench's
                  free-swing run is exactly this, and it is the only run that
                  sees the inertia and the friction with no motor torque in the
                  way.

    Two inertias, coupled through the backlash: the motor side carries J_m (the
    reflected rotor, which dominates) and the load side J_l plus whatever the arm
    adds. With theta_bl = 0 they lock and it degenerates to one state.

    The inner loop below is written in scalar Python rather than by calling
    duty()/current()/motor_torque(). It is the same arithmetic — `_selftest()`
    asserts the two agree — but a fit evaluates this hundreds of times over
    hundreds of thousands of 0.1 ms steps, and numpy's scalar dispatch costs more
    than the physics does. The array versions above stay the readable statement
    of the law and are what step 4 will port to MJX.
    """
    target = np.atleast_1d(np.asarray(target, float))
    n = len(target)
    u_arr = np.broadcast_to(np.asarray(u_bat, float), (n,))
    if load_torque is None:
        load_torque = lambda q: 0.0
    on = (np.ones(n, bool) if torque_on is None
          else np.broadcast_to(np.asarray(torque_on, bool), (n,)))

    sub = max(1, int(round(dt / dt_int)))
    h = dt / sub
    ctrl_every = max(1, int(round(1.0 / (p.loop_hz * h))))

    # hoisted, because they are read once per inner step
    kp, kd, dead, punch, dmax = p.kp, p.kd, p.deadband, p.punch, p.duty_max
    k_u, k_e, k_w, R_, tau_c, b_v = p.k_u, p.k_e, p.k_w, p.R, p.tau_c, p.b_v
    mu_l = p.mu_load
    J_m, J_l = p.J_m, max(p.J_l, 1e-9)
    half, k_bl, c_bl = 0.5 * p.theta_bl, p.k_bl, p.c_bl
    v_eps, enc = p.v_eps, ENC_STEP_RAD
    after = p.enc_after_backlash

    th_m = th_l = float(q0)
    w_m = w_l = float(w0)
    q_o = np.empty(n); w_o = np.empty(n); i_o = np.empty(n)
    u_o = np.empty(n); t_o = np.empty(n); m_o = np.empty(n)

    k = 0
    d = 0.0
    for s in range(n):
        tgt, volt0, driven = target[s], u_arr[s], bool(on[s])
        for _ in range(sub):
            if k % ctrl_every == 0:
                if not driven:
                    d = 0.0
                else:
                    q_fb = th_l if after else th_m
                    w_fb = w_l if after else w_m
                    e = round((tgt - q_fb) / enc) * enc
                    if abs(e) <= dead:
                        e = 0.0
                    u = kp * e - kd * w_fb
                    if u != 0.0 and abs(u) < punch:
                        u = punch if u > 0 else -punch
                    d = dmax if u > dmax else (-dmax if u < -dmax else u)
            volt = volt0
            i = (d * volt - k_e * w_m) / R_ if driven else 0.0
            if sag and driven:
                # Floored at zero. A pack under load delivers LESS voltage; it
                # never delivers negative voltage, and if it did the torque term
                # k_u*d*volt would flip sign and drive the shaft harder in the
                # direction it was already turning — positive feedback that
                # destroys the integration rather than modelling anything.
                # Unreachable at bench currents (one servo off a lab PSU cannot
                # sag its own supply past zero), which is exactly why the floor
                # has to be written down instead of relied on: rl/env/walk.py
                # applies the same law to the SUMMED current of twelve servos,
                # where it is reachable, and did reach it.
                volt = max(volt0 - sag * abs(i), 0.0)
                i = (d * volt - k_e * w_m) / R_
            # The transmitted torque is computed FIRST now, because the
            # friction depends on it. No circularity: it is a function of the
            # two positions and velocities only, all of them state.
            delta = th_m - th_l
            if delta > half:
                tau_t = k_bl * (delta - half) + c_bl * (w_m - w_l)
            elif delta < -half:
                tau_t = k_bl * (delta + half) + c_bl * (w_m - w_l)
            else:
                tau_t = 0.0
            # KARNOPP, not tanh: below v_eps the joint STICKS.  tanh(w/v_eps)
            # is exactly zero at w = 0, so the old law had a servo that could
            # not hold anything at rest while the real one breaks away at
            # 0.23-0.35 N*m (ST3215_STS3215_measured_parameters.md).  The
            # consequence was not cosmetic: a hold approached from below and
            # from above settled at the SAME duty, half-difference 0.00-0.03 V
            # against the real servo's 0.29, and `mu_load` was therefore not
            # recoverable from this model's own output (fit_bam --selftest
            # said so, about the model rather than the experiment).
            #
            # The law is unchanged above the stick band - it is the same
            # (tau_c + mu_load*|tau_t|) ceiling, and b_v*w is outside it
            # because viscous drag does not stick.  What changes is only what
            # happens inside |w| < v_eps, where friction now opposes the NET
            # applied torque up to that ceiling instead of fading to nothing.
            tau_app = k_u * d * volt - b_v * w_m - tau_t
            if driven:
                tau_app -= k_w * w_m
            f_max = tau_c + mu_l * abs(tau_t)
            if abs(w_m) < v_eps and abs(tau_app) <= f_max:
                # stuck: friction takes up exactly the applied torque, and the
                # motor is held.  Setting w_m rather than integrating it is the
                # point of Karnopp - an ODE cannot express "does not move".
                w_m = 0.0
                tau_m = tau_t                      # so tau_m - tau_t == 0 below
            else:
                tau_m = (k_u * d * volt
                         - f_max * (math.tanh(w_m / v_eps) if abs(w_m) < v_eps
                                    else (1.0 if w_m > 0 else -1.0))
                         - b_v * w_m)
                if driven:
                    tau_m -= k_w * w_m
            w_m += h * (tau_m - tau_t) / J_m
            w_l += h * (tau_t + load_torque(th_l)) / J_l
            th_m += h * w_m
            th_l += h * w_l
            k += 1
        q_o[s] = th_l; w_o[s] = w_l; m_o[s] = th_m
        u_o[s] = d * u_arr[s]
        i_o[s] = (u_o[s] - k_e * w_m) / R_ if driven else 0.0
        t_o[s] = tau_t
    # what the bus would report: the encoder is quantised, and so is the ammeter
    return dict(q=q_o, w=w_o, i=i_o, u=u_o, tau=t_o, q_motor=m_o,
                q_meas=np.round(q_o / ENC_STEP_RAD) * ENC_STEP_RAD)


def pendulum_load(mass_kg, radius_m, q_zero_is_down=True):
    """Gravity torque of the bench arm — the known load the fit is anchored to.

    q measured from hanging-down, so the torque is -m*g*r*sin(q): maximal with
    the arm horizontal, zero and stable hanging. Angle-dependent on purpose, so
    a single sweep covers a range of load torques.
    """
    mgr = mass_kg * 9.80665 * radius_m
    if q_zero_is_down:
        return lambda q: -mgr * np.sin(q)
    return lambda q: -mgr * np.cos(q)


# ============================================================== self-test
def _selftest() -> int:
    """Physics checks on the law, runnable with no data and no hardware.

    These are the properties the fit and the training environment both rely on,
    and each one of them was a bug at some point while this file was written.
    """
    ok = True

    def check(name, got, want=True, tol=None):
        nonlocal ok
        good = (abs(got - want) <= tol) if tol is not None else (got == want)
        ok &= bool(good)
        print(f"  {'ok  ' if good else 'FAIL'} {name}: {got}"
              + ("" if good else f"  != {want}" + (f" +-{tol}" if tol else "")))

    p = Params()

    # the inner loop's registers do what the servo's registers do
    check("dead zone swallows a small error",
          float(duty(p, 0.5 * p.deadband, 0.0)), 0.0)
    check("outside it, proportional", float(duty(p, 0.02, 0.0)),
          p.kp * (round(0.02 / ENC_STEP_RAD) * ENC_STEP_RAD), tol=1e-9)
    check("duty saturates", float(duty(p, 10.0, 0.0)), 1.0, tol=1e-12)
    # just outside the dead zone, where kp*e is smaller than the punch. Note the
    # error has to survive quantisation first: 1.2 dead zones rounds back to one
    # the encoder cannot distinguish from the edge, and reads as no error at all.
    pp = Params(punch=0.25)
    e = 2.0 * pp.deadband
    assert pp.kp * e < pp.punch
    check("punch is the smallest drive there is",
          float(duty(pp, e, 0.0)), 0.25, tol=1e-12)

    # torque off is an OPEN winding, not a brake. Getting this wrong makes the
    # free-swing run read electrical damping as mechanical friction.
    check("no current with the bridge off", float(current(p, 12.0, 3.0, False)), 0.0)
    check("back-EMF only when driven",
          float(motor_torque(p, 0.0, 1.0, False) - motor_torque(p, 0.0, 1.0, True)),
          p.k_w, tol=1e-12)

    # --- load-dependent friction ------------------------------------------
    # It must GROW with the transmitted torque, at exactly mu_load per N*m, and
    # it must still oppose MOTION rather than the load: a gearbox carrying a
    # torque one way while turning the other loses to friction just the same.
    check("friction grows with load at mu_load",
          float(friction(p, 1.0, 1.0) - friction(p, 1.0, 0.0)),
          p.mu_load, tol=1e-9)
    check("load-dependent friction opposes motion, not load",
          float(np.sign(friction(p, -1.0, 1.0))), -1.0)
    check("its magnitude ignores the load's sign",
          float(friction(p, 1.0, -0.7) - friction(p, 1.0, 0.7)), 0.0, tol=1e-12)

    # --- static friction: the joint must be able to HOLD -------------------
    # PLAN.md step 2b.  These are the checks that were missing while
    # simulate() used tanh(w/v_eps), which is exactly zero at rest: a servo
    # with no stiction creeps under any load at all, however small, and a hold
    # approached from two sides settles at one duty instead of two.  Both
    # probes run with the bridge OFF, where there is no motor torque to hide
    # behind and the only thing that can hold the shaft is friction.
    import dataclasses as _dc
    _ps = _dc.replace(Params(), J_l=1e-3)
    _n, _dt = 400, 0.002

    def _final_speed(load_nm):
        """|w| at the end. Position is the WRONG discriminator here: the load
        swings through the backlash band before the spring engages, so it moves
        a little whatever friction does. Coming to REST is the question."""
        r = simulate(_ps, np.zeros(_n), _dt, q0=0.0, w0=0.0,
                     load_torque=lambda q: load_nm,
                     torque_on=np.zeros(_n, bool))
        return abs(float(r["w"][-1]))

    # Breakaway is tau_c + mu_load*|tau_t|, so the load it can hold solves
    # |tau| <= tau_c + mu_load*|tau| — i.e. tau_c/(1 - mu_load).
    _break = _ps.tau_c / (1.0 - _ps.mu_load)
    check("a sub-breakaway load comes to rest", _final_speed(0.5 * _break) < 1e-3)
    check("a load past breakaway keeps moving", _final_speed(3.0 * _break) > 1e-2)

    # mu_load = 0 must reproduce the law exactly as it was before this term
    # existed. A new term that quietly changes the old answer is not an
    # extension, it is a different model wearing the same name.
    p0 = Params(mu_load=0.0)
    check("mu_load = 0 is the old law",
          float(motor_torque(p0, 6.0, 1.0) -
                (p0.k_u * 6.0 - p0.k_w * 1.0
                 - p0.tau_c * _sign(1.0, p0.v_eps) - p0.b_v * 1.0)),
          0.0, tol=1e-12)

    # The proxy motor_torque() uses when no transmitted torque is available is
    # claimed to be EXACT for forward driving at steady state, where the motor
    # supplies load plus friction. Check the algebra rather than trusting it:
    # feed the proxy's own implied tau_t back in and the drive must reappear.
    U = 6.0
    tau_t = p.k_u * U / (1.0 + p.mu_load)
    check("the proxy inverts the steady-state balance",
          float(tau_t + p.mu_load * abs(tau_t)), p.k_u * U, tol=1e-12)

    # The two figures the docstring uses to argue there is no forward/back
    # asymmetry. Both are the SAME symmetric law evaluated at the two load
    # levels the bench actually measured, and both have to land near the
    # reported 28 % forward / 62 % back-driving for that argument to hold.
    bench = Params(tau_c=0.186, mu_load=0.251)          # holdbi at 12 V
    fwd_load = 0.877                                    # arm horizontal-ish
    f_fwd = bench.tau_c + bench.mu_load * fwd_load
    check("forward loss lands near the reported 28 %",
          round(f_fwd / (fwd_load + f_fwd), 2), 0.32, tol=0.05)
    bwd_load = 0.941 * 2.0 / math.pi                    # mean over a free swing
    f_bwd = bench.tau_c + bench.mu_load * bwd_load
    check("back-driving loss lands near the reported 62 %",
          round(f_bwd / bwd_load, 2), 0.56, tol=0.08)

    # the transmission is dead inside the play and continuous at its edge
    check("no torque inside the backlash",
          float(transmitted(p, 0.4 * p.theta_bl, 0.0)), 0.0)
    check("continuous at the edge",
          float(transmitted(p, 0.5 * p.theta_bl + 1e-9, 0.0)), 0.0, tol=1e-5)

    # vendor consistency: the free-running speed is U / k_e, less friction
    # target far enough away that it never arrives: what is wanted is the
    # terminal speed, which is where the back-EMF cancels the applied voltage
    free = simulate(Params(J_l=1e-4, theta_bl=0.0, tau_c=0.0, b_v=0.0,
                           mu_load=0.0),
                    np.full(1200, 60.0), 0.005, u_bat=12.0)
    check("terminal speed is U / k_e",
          round(float(np.max(free["w"])), 3), round(p.no_load_speed(12.0), 3),
          tol=0.05)

    # a load at the stall torque must stop it
    stalled = simulate(Params(J_l=1e-3, theta_bl=0.0), np.full(600, 1.0), 0.005,
                       u_bat=12.0, load_torque=lambda q: -3.2)
    check("a load past stall does not lift", float(stalled["q"][-1]) < 0.1)

    # supply sag reduces the ceiling, which is the whole reason for the voltage law
    hi = simulate(Params(J_l=1e-3), np.full(400, 1.0), 0.005, u_bat=12.6,
                  load_torque=lambda q: -1.5)
    lo = simulate(Params(J_l=1e-3), np.full(400, 1.0), 0.005, u_bat=9.9,
                  load_torque=lambda q: -1.5)
    check("a flatter pack holds less", float(lo["q"][-1]) < float(hi["q"][-1]) - 1e-3)

    print("  " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_selftest())
