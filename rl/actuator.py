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
  4. mechanical     tau = k_u*U - k_w*w - tau_c*sign(w) - b_v*w, driving the
                    reflected rotor inertia J_m, which for 1:345 is ~73x the
                    knee link's own inertia (measured: rl/checks/check_model.py).
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

Where the encoder sits
----------------------
`enc_after_backlash` decides whether the servo's control loop — and the position
we read back over the bus — sees the output shaft (after the gearbox play) or
the motor side. The vendor wiki in 3d/ref/st3215_wiki.html says "360 degree
magnetic encoder ... 360/4096", i.e. one absolute turn of the OUTPUT, which only
works with the sensor on the output shaft; that is the default here. It is a
30-second bench test to confirm (torque off, rock the horn, watch Present
Position move by the backlash) and the answer changes the observation wiring in
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
    tau_c: float = 0.05       # N*m, Coulomb friction
    b_v: float = 0.010        # N*m*s/rad, viscous friction
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
        data_fields=["R", "k_e", "k_u", "J_m", "J_l", "tau_c", "b_v", "kp", "kd",
                     "deadband", "punch", "duty_max", "loop_hz",
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


def motor_torque(p: Params, u_volt: float, w: float, driven=True, xp=np) -> float:
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
    """
    return (p.k_u * u_volt - xp.where(driven, p.k_w * w, 0.0)
            - p.tau_c * _sign(w, p.v_eps, xp) - p.b_v * w)


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
            tau_m = k_u * d * volt - tau_c * math.tanh(w_m / v_eps) - b_v * w_m
            if driven:
                tau_m -= k_w * w_m
            delta = th_m - th_l
            if delta > half:
                tau_t = k_bl * (delta - half) + c_bl * (w_m - w_l)
            elif delta < -half:
                tau_t = k_bl * (delta + half) + c_bl * (w_m - w_l)
            else:
                tau_t = 0.0
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

    # the transmission is dead inside the play and continuous at its edge
    check("no torque inside the backlash",
          float(transmitted(p, 0.4 * p.theta_bl, 0.0)), 0.0)
    check("continuous at the edge",
          float(transmitted(p, 0.5 * p.theta_bl + 1e-9, 0.0)), 0.0, tol=1e-5)

    # vendor consistency: the free-running speed is U / k_e, less friction
    # target far enough away that it never arrives: what is wanted is the
    # terminal speed, which is where the back-EMF cancels the applied voltage
    free = simulate(Params(J_l=1e-4, theta_bl=0.0, tau_c=0.0, b_v=0.0),
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
