"""
fit_bam.py — turn bench csv into rl/params/st3215.json.

    python bench/fit_bam.py --selftest              # no data: recover known params
    python bench/fit_bam.py --data bench/data
    python bench/fit_bam.py --data bench/data --holdout chirp --plot out.png

The model is rl/actuator.py; this file only decides which of its parameters the
data can support and finds them. It fits in three passes, because a single
blind least-squares over ten coupled parameters lands in whatever local minimum
the initial guess happens to sit next to:

  1. free swing   The oscillation period of the released arm gives the total
                  inertia directly — J = m*g*r / omega0^2 — and J_load is known
                  from the arm, so J_m falls out. This is the one place the
                  reflected rotor inertia is measured rather than inferred, and
                  rl/checks/check_model.py showed it is 73x the knee link's own
                  inertia, so it is the single most consequential number here.

  2. static holds At steady state omega = 0 and the torque balance is exact:
                  the slope of |m*g*r*sin(q)| against measured current is the
                  effective torque constant k_u*R, and the slope of the standing
                  position error against i/U_bat is R/kp. Two clean linear
                  regressions, no dynamics, no integration.

  3. everything   Those seed a bounded least_squares over all ten parameters
                  against every trajectory at once, with position and current
                  residuals weighted to comparable size.

Identifiability is checked, not assumed. Two conditions have to hold or the fit
is reported as under-determined rather than quietly returning a number:

  * more than one supply voltage. At a single voltage the electrical damping
    k_w = k_u*k_e and the viscous friction b_v enter every equation as the same
    coefficient of omega, and no amount of data separates them.
  * current actually present. Without it the same degeneracy reappears through
    a different door.

One run is held out and never enters the objective. A fit that matches the runs
it was fitted to and not the one it was not has learnt the noise.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "rl"))

import runlog                                                        # noqa: E402
import actuator as A                                                 # noqa: E402

#: The fit does NOT run in Params' own coordinates, and this is the single
#: thing that makes it converge.
#:
#: R, k_e, k_u and kp are strongly coupled: every current the bench can measure
#: depends on them only through ratios, and every torque only through products.
#: Move R and k_u together along k_u*R = const and nothing observable changes, so
#: least_squares walks along that valley, terminates on xtol, and reports the
#: initial guess back with a straight face. (Observed: R recovered to 0.1 % while
#: k_u sat on its seed 34 % away, because the seed had divided by the wrong R.)
#:
#: So the fit variables are the combinations the instruments actually see:
#:
#:   kt_eff = k_u * R    N*m/A     torque per amp — the slope of the hold data
#:   g_kp   = kp / R     A/(rad V) current per unit error per volt, unsaturated
#:   g_ke   = k_e / R    A s/rad   how fast the current falls with speed
#:   g_R    = 1 / R      A/V       stall current per volt, seen only when the
#:                                 duty saturates, which is why the big steps
#:                                 are in the trajectory set
#:
#: with R = 1/g_R, kp = g_kp*R, k_e = g_ke*R, k_u = kt_eff*g_R on the way out.
#: Everything else in Params is either a register we read (loop_hz), a numerical
#: device (v_eps, c_bl), or a modelling choice the rock test settles
#: (enc_after_backlash).
FIT = [("kt_eff", 0.05, 20.0), ("g_kp", 0.05, 500.0), ("g_ke", 0.005, 10.0),
       ("g_R", 0.005, 3.0),
       ("J_m", 1e-4, 0.2), ("tau_c", 1e-4, 1.0), ("b_v", 1e-5, 1.0),
       # Friction per N*m carried. Bounded below 0.9 because at 1.0 the friction
       # equals the torque being transmitted and the gearbox delivers nothing;
       # the optimiser will happily walk there to absorb a missing term, which
       # is precisely the failure this parameter was added to stop.
       ("mu_load", 0.0, 0.9),
       ("deadband", 1e-5, 0.05), ("punch", 0.0, 0.5), ("theta_bl", 0.0, 0.05)]

W_POS = math.radians(0.3)     # rad of position error worth one unit of residual
#: Smallest duty at which PRESENT_CURRENT / duty is worth believing.  0.02 is two
#: PRESENT_LOAD counts; the supply current there is 0.0011 A, well under the
#: register's 0.0065 A LSB.  See the note in Run.__init__.
DUTY_MIN = 0.02

#: PRESENT_CURRENT's own LSB, 6.5 mA.  Here only so the selftest can quantise its
#: synthetic current the way the hardware does; the live path reads it from
#: feetech.registers.CURRENT_LSB_A.
CURRENT_LSB = 0.0065

W_CUR = 0.03                  # A of current error worth the same
SMOOTH = 5                    # samples of moving average before comparing


def _smooth(x, n=SMOOTH):
    """Low-pass both sides before differencing them.

    Not cosmetic. A saturated step edge is an amp of current in a couple of
    milliseconds, so a parameter change that moves that edge by one sample
    produces an enormous residual with no useful direction in it — the objective
    becomes a field of spikes, the finite-difference gradient reads ~1e9, and the
    trust region collapses to steps of 1e-6 and reports convergence. (Measured,
    on synthetic data: 13 evaluations and a zero step.) Averaging over 25 ms
    keeps everything the 50 Hz policy can see and throws away the sample-level
    phase of transients, which is not what the model is being asked to match.
    """
    if n <= 1:
        return np.asarray(x)
    k = np.ones(n) / n
    return np.convolve(np.asarray(x), k, mode="same")


# ------------------------------------------------------------------- runs
class Run:
    """One trajectory, resampled onto a uniform grid the integrator can use."""

    #: Trajectories that must never be truncated. `--seconds` exists because the
    #: cost of a fit is wall-clock and most runs say what they have to say early,
    #: but a run whose measurement is a DIFFERENCE between its first half and its
    #: second is not shortened by cutting the tail, it is halved. `holdbi` walks
    #: its ladder up and then down; cut 36 s to the default 25 and the descending
    #: pass loses two thirds of its rungs, the two halves stop lining up, and
    #: seed_from_holdbi silently finds fewer than four paired angles and declines.
    #: That is exactly how it behaved when this class did not know about it —
    #: no error, no warning, mu_load left sitting on its prior.
    WHOLE_RUN = ("holdbi",)

    def __init__(self, name, meta, cols, seconds, fit_hz):
        t = np.asarray(cols["t"])
        if meta.get("trajectory", name).startswith(self.WHOLE_RUN):
            keep = np.ones(len(t), bool)
        else:
            keep = t <= (t[0] + seconds)
        t = t[keep]
        dt = 1.0 / fit_hz
        grid = np.arange(t[0], t[-1], dt)
        g = lambda c: np.interp(grid, t, np.asarray(cols[c])[keep])
        self.name, self.meta = name, meta
        self.t, self.dt = grid, dt
        self.q = g("q_rad")
        # The servo's own PRESENT_SPEED. Worth carrying rather than always
        # differentiating q: at the bench's 200 Hz one encoder count of position
        # is 0.307 rad/s, while PRESENT_SPEED quantises at 0.0767 rad/s -- four
        # times finer. Measured on an ST3215 free swing, where the reported trace
        # stepped cleanly between 1.9175 and 1.9942 rad/s while dq/dt scattered
        # over a 0.3 rad/s staircase.
        self.w_meas = g("w_rad_s") if "w_rad_s" in cols else None
        self.volt = g("volt_v")
        tgt = np.asarray(cols["target_rad"])[keep]
        self.on = ~np.isnan(tgt)
        self.target = np.interp(grid, t, np.nan_to_num(tgt))
        self.on = np.interp(grid, t, self.on.astype(float)) > 0.5
        self.trajectory = meta.get("trajectory", name)
        self.mass = float(meta.get("mass_kg", 0.0) or 0.0)
        self.radius = float(meta.get("radius_m", 0.0) or 0.0)
        self.J_load = float(meta.get("arm_inertia", 0.0) or 0.0) \
            + self.mass * self.radius ** 2
        self.u_bat = (self.volt if np.isfinite(self.volt).all() and self.volt.max() > 1
                      else np.full_like(grid, float(meta.get("psu_volts", 12.0))))
        # PRESENT_LOAD is the duty the inner loop actually applied, 0-1000,
        # sign-magnitude with bit 10 the direction. Take the magnitude BEFORE
        # resampling or the sign bit smears into the value. This is the only
        # direct evidence of saturation in the log: seed_from_saturation used to
        # infer it from position error, which on a rig where the arm slews freely
        # selects mostly unsaturated samples and drags R with them.
        lr = cols.get("load_raw")
        if lr is None or len(lr) != len(keep):
            self.duty = self.duty_signed = None
        else:
            # `load_raw` is written in TWO conventions and the column name is
            # honest about neither. Servo.decode() applies registers.SIGN_BIT, so
            # logs taken after that landed hold a SIGNED integer (-472 for a hold
            # pulling 47 % the other way); logs taken before it hold the RAW
            # register, sign-magnitude with bit 10 the direction (1152 = 0x400|128).
            # Both are in bench/data, six months apart.
            #
            # Neither a bare mask nor a bare abs() reads both. Masking a signed
            # value with 0x3FF reads two's complement as a magnitude — -472 becomes
            # 552, and -24 becomes 1000, so every lightly-loaded sample reports as
            # fully saturated. abs() on a raw negative gives 1024+mag, i.e. a duty
            # over 1. Found 2026-09-08: the mask manufactured 1385 "pinned" samples
            # across the holds, steps, reversals and triangles, every one of them
            # drawing about 0.005 A where a genuinely saturated servo pulls 2 A,
            # and seed_from_saturation regressed R = 20.2 ohm off them. The real
            # stall runs, which do saturate, say 4.0-4.1.
            #
            # The two are separable without a flag, because their ranges do not
            # overlap where it matters: a decoded value is in [-1023, 1023] and a
            # raw one in [0, 2047], so anything negative is decoded, anything at
            # or above 1024 is raw-and-negative, and [0, 1023] is the same
            # magnitude either way.
            v = np.asarray(lr, dtype=np.float64)[keep]
            mag = np.where(v < 0, -v, np.where(v >= 1024, v - 1024, v))
            self.duty = np.interp(grid, t, mag / 1000.0)
            # The same recovery keeping the DIRECTION. Every pass below that
            # divides a current wants the magnitude, but seed_from_holdbi wants
            # the sign: its whole measurement is that the duty holding a given
            # angle differs depending on which side the joint arrived from, and
            # an absolute value throws exactly that away.
            sgn = np.where(v < 0, v, np.where(v >= 1024, -(v - 1024), v))
            self.duty_signed = np.interp(grid, t, sgn / 1000.0)

        # PRESENT_CURRENT IS THE SUPPLY CURRENT, NOT THE MOTOR CURRENT, and every
        # pass below wants the motor's.  Behind a PWM bridge the motor sits at
        # I_m = d*U/R while the supply only delivers d of that, so the register
        # reads d^2*U/R and a fit that takes it raw is regressing against d^2
        # where it believes it has d.  It does not diverge, it converges on
        # nonsense: measured 2026-09-08, taking it raw returned R = 20.2 ohm
        # against a bench-measured 4.3, k_e pinned on its lower bound, and an
        # implied no-load speed of 240 rad/s against a spec 4.71.
        #
        # The correction is one division, and it is self-verifying: inverting
        # R = d^2*U/I over a hold ladder returns 4.29..4.44 ohm across a 7x range
        # of duty and two supply voltages, against the 12/2.7 = 4.44 the vendor's
        # locked-rotor spec implies - a number the register was never told.  See
        # feetech/registers.py, CURRENT_LSB_A.
        #
        # Below DUTY_MIN the division is meaningless rather than merely noisy: the
        # supply current there is under one 6.5 mA LSB, so the register reads zero
        # and the ratio is 0/0.  Clamping the divisor caps the amplification at
        # 1/DUTY_MIN and costs nothing, because the true motor current in that
        # regime is small and the residual weights it by W_CUR anyway.
        if self.duty is None:
            self.i = g("current_a")
            self.i_is_supply = True      # uncorrectable: this log predates load_raw
        else:
            self.i = g("current_a") / np.maximum(self.duty, DUTY_MIN)
            self.i_is_supply = False
        self.load = A.pendulum_load(self.mass, self.radius)

    def predict(self, p: A.Params, dt_int):
        q = A.Params(**{**p.__dict__, "J_l": max(self.J_load, 1e-6)})
        return A.simulate(q, self.target, self.dt, q0=self.q[0], u_bat=self.u_bat,
                          load_torque=self.load, torque_on=self.on, dt_int=dt_int)


def load_runs(d, seconds, fit_hz):
    runs, dry = [], []
    for name, meta, cols in runlog.load_dir(d):
        if not cols["t"]:
            continue
        # sweep.py --dry-run writes a real csv, with the real filename, into the
        # real --out directory; only the metadata says it never touched a servo.
        # Checking the rig before the bench session is the sane thing to do, so
        # these files WILL appear. Refuse the mixed corpus loudly rather than
        # fitting the ST3215 to a simulation of itself.
        if meta.get("dry_run"):
            dry.append(name)
            continue
        runs.append(Run(name, meta, cols, seconds, fit_hz))
    if dry:
        raise SystemExit(
            f"{len(dry)} dry-run file(s) in {d} — these were generated without a "
            f"servo on the bus and are not measurements:\n  "
            + "\n  ".join(sorted(dry))
            + "\nMove or delete them, then fit. (sweep.py --dry-run --out "
              "somewhere-else keeps them apart in the first place.)")
    if not runs:
        raise SystemExit(f"no csv in {d}")
    return runs


#: The trajectory names the analytic passes (seed_from_holdbi,
#: seed_from_freeswing, seed_from_holds) actually select on. Everything else in
#: the directory reaches the objective only through the --refine pass.
ANALYTIC_TRAJECTORIES = {"freeswing", "hold", "holdbi"}


def check_identifiable(runs) -> list[str]:
    problems = []
    volts = {round(float(r.meta.get("psu_volts", 0)), 1) for r in runs}
    if len(volts) < 2:
        problems.append(
            f"only one supply voltage in the data ({sorted(volts)}). The back-EMF "
            f"damping and the viscous friction are then the same column of the "
            f"regressor and the split between them is arbitrary. Re-run the holds "
            f"and the chirp at 12.6, 11.1 and 9.9 V.")
    if not any(np.nanmax(r.i) > 1e-3 for r in runs):
        problems.append(
            "no current above the noise anywhere. Present Current is what "
            "separates the electrical parameters from the mechanical; check "
            "registers.PRESENT_CURRENT is the right address for this firmware "
            "(feetech.bus --dump) before fitting.")
    if not any(r.trajectory == "freeswing" for r in runs):
        problems.append(
            "no free-swing run. J_m is then identified only through the driven "
            "trajectories, where it trades off against kp — and J_m is the number "
            "the leg's whole dynamics rests on.")
    if not any(r.trajectory == "hold" for r in runs):
        problems.append("no hold run: the torque constant loses its clean anchor.")
    # A trajectory nobody reads is worse than a missing one: it costs bench time,
    # it sits in the directory looking like evidence, and the fit comes out
    # bit-identical to the run without it. Measured 2026-09-09 — the holdbi and
    # speed ladders PLAN.md step 1 asks for were captured at three voltages, and
    # every fitted parameter matched a fit on the old data to five decimals,
    # because the analytic passes select on the two names below and the global
    # refinement only runs under --refine.
    seen = {r.trajectory for r in runs}
    unread = sorted(t for t in seen if t not in ANALYTIC_TRAJECTORIES)
    if unread:
        problems.append(
            "the analytic passes read only " + " and ".join(sorted(ANALYTIC_TRAJECTORIES))
            + f" — {', '.join(unread)} reach the fit ONLY under --refine, and are "
            f"ignored entirely without it. That is not a warning about coverage: a "
            f"run that no pass consumes changes no parameter, so do not read an "
            f"unchanged fit as confirmation that the new data agreed with the old.")
    regs = {json.dumps(r.meta.get("registers"), sort_keys=True) for r in runs}
    if len(regs) > 1:
        problems.append(
            "the runs were taken with DIFFERENT control registers. A fit is only "
            "valid for one set — these are two different machines. Re-take, or "
            "split the data by register set and fit separately.")
    return problems


# --------------------------------------------------------------- seeding
def seed_from_freeswing(runs, p: A.Params) -> A.Params:
    """Total inertia from the released arm, two ways, whichever the data supports.

    If it oscillates, the period gives it: J = m*g*r / omega0^2.

    If it does not — and with a 1:345 gearbox that is the likely case, since the
    reflected rotor inertia and the gearbox friction between them can turn the
    release into one sluggish fall — use the instant of release itself. At that
    instant omega = 0, so both friction terms vanish and the torque balance has
    exactly one unknown:

        J_total * qddot(0) = m*g*r*sin(q0)

    Fit a quadratic to the first 80 ms and read qddot off it. Friction is not
    quite zero over a finite window, so this reads the acceleration a little low
    and J_total a little high; it is a seed, and the three-parameter fit below
    refines it.

    A warning about the arm, because it decides how good this can be: J_m is
    obtained as J_total - J_arm, a difference of two similar numbers, so a heavy
    arm destroys the estimate. Do the free-swing run with the SMALLEST mass at
    the SHORTEST radius that will still overcome the Coulomb friction (m*g*r
    comfortably above tau_c), and save the heavy long arm for the holds, where a
    big torque is exactly what is wanted. One bench, two arms.
    """
    for r in runs:
        if r.trajectory != "freeswing" or r.mass <= 0 or r.radius <= 0:
            continue
        q = r.q[~r.on]
        if len(q) < 50:
            continue
        mgr = r.mass * 9.80665 * r.radius
        rest = float(np.mean(q[-len(q) // 4:]))
        sgn = np.sign(q - rest)
        cross = np.flatnonzero((sgn[:-1] < 0) & (sgn[1:] >= 0))
        if len(cross) >= 2:
            T = float(np.mean(np.diff(cross)) * r.dt)
            J_tot = mgr / (2 * math.pi / T) ** 2
            how = f"period {T*1000:.0f} ms"
        else:
            n = max(8, int(0.08 / r.dt))
            t = np.arange(n) * r.dt
            a2 = np.polyfit(t, q[:n], 2)[0]              # q ~ q0 + v0 t + a/2 t^2
            acc = abs(2 * a2)
            if acc < 1e-3:
                print("  free swing: the arm did not move on release. The gearbox "
                      "is effectively self-locking at this load; use a longer arm, "
                      "and if it still will not move J_m comes from the driven "
                      "runs alone.")
                continue
            # Friction does NOT vanish over a finite window, whatever the
            # instant of release says: the Coulomb term reaches full size within
            # a millisecond of the shaft moving, so the driving torque over the
            # 80 ms is m*g*r*sin(q0) - tau_c, not m*g*r*sin(q0). Subtracting the
            # current estimate of tau_c removes most of a bias that is otherwise
            # the ratio of the two — on a light arm that is a factor of three.
            drive = mgr * abs(math.sin(q[0])) - p.tau_c
            if drive <= 0:
                print("  free swing: m*g*r is below the friction estimate; this "
                      "arm cannot move the joint. Use a longer one.")
                continue
            J_tot = drive / acc
            how = (f"release acceleration {acc:.1f} rad/s^2, no oscillation, "
                   f"driving torque {drive:.3f} N*m after friction")
        J_m = max(1e-4, J_tot - r.J_load)
        share = r.J_load / max(J_tot, 1e-9)
        print(f"  free swing: {how} -> J_total {J_tot:.5f}, arm {r.J_load:.5f} "
              f"({share*100:.0f} % of it)  ->  J_m = {J_m:.5f} kg m^2")
        if share > 0.35:
            print("     !! the arm is more than a third of the total inertia, so "
                  "J_m is a small difference of large numbers. Re-run the free "
                  "swing with a lighter, shorter arm.")
        return A.Params(**{**p.__dict__, "J_m": J_m})
    return p


def seed_from_holdbi(runs, p: A.Params) -> A.Params:
    """tau_c and mu_load, from the hold ladder walked in BOTH directions.

    This is the only pass here that measures rather than fits, and it is the
    only one that can see load-dependent friction at all. A static hold does not
    settle at zero net torque, it settles wherever friction happens to balance
    the rest, so the duty it holds depends on which side the joint arrived from.
    Half the difference between the two approaches IS the friction at that load;
    the mean is the torque with friction removed. Every other pass in this file
    sees only one of those and has to guess how it splits.

    Anchored to GRAVITY, not to k_u, and that is the whole point of doing it this
    way. m*g*r*sin(q) is known exactly; k_u is the parameter that absorbs missing
    friction and is inflated by a factor nobody has pinned down (the fit's
    implied stall is 4.23 N*m against a spec 2.94). Seeding a friction term off
    k_u would be circular. So both friction and gravity are measured in the same
    duty*volt units within one run, and the ratio between them converts to N*m
    with no electrical parameter involved.

    Nothing here needs an integration, a voltage sweep or a prior, so it runs
    FIRST — before the free swing, which subtracts tau_c to get its driving
    torque and had been using a value roughly half this one.
    """
    per_run = []
    for r in runs:
        if r.trajectory != "holdbi" or r.mass <= 0 or r.radius <= 0:
            continue
        if r.duty is None:
            continue
        du = r.duty_signed * r.u_bat
        mgr = r.mass * 9.80665 * r.radius
        step = np.flatnonzero(np.abs(np.diff(r.target)) > 1e-6)
        edges = np.concatenate(([0], step + 1, [len(r.target)]))
        seg = []
        for a_, b_ in zip(edges[:-1], edges[1:]):
            n = b_ - a_
            if n < 20:
                continue
            sl = slice(a_ + int(0.6 * n), b_)
            seg.append((round(float(np.mean(r.target[sl])), 2),
                        float(np.mean(r.q[sl])), float(np.mean(du[sl]))))
        if len(seg) < 6:
            continue
        # The two passes are keyed by COMMANDED angle, not by index: each starts
        # one rung outside the ladder so its first measured angle is approached
        # from the right side, so the halves are offset by one and zipping them
        # pairs every rung with its neighbour.
        half = len(seg) // 2
        up = {a: (q, d) for a, q, d in seg[:half]}
        dn = {a: (q, d) for a, q, d in seg[half:]}
        ang = sorted(set(up) & set(dn))
        if len(ang) < 4:
            continue
        qm = np.array([(up[a][0] + dn[a][0]) / 2 for a in ang])
        mean = np.array([(up[a][1] + dn[a][1]) / 2 for a in ang])
        fric = np.array([abs(up[a][1] - dn[a][1]) / 2 for a in ang])
        # duty*volts per N*m, from this run's own gravity — the conversion
        slope = float(np.polyfit(np.sin(qm), -mean, 1)[0])
        if abs(slope) < 1e-9:
            continue
        k_t = mgr / slope                       # N*m per duty*volt
        tau = mgr * np.abs(np.sin(qm))
        mu, tc = np.polyfit(tau, fric * k_t, 1)
        per_run.append((float(tc), float(mu), float(k_t),
                        float(np.mean(r.u_bat)),
                        float(np.median(fric)), float(np.max(tau))))
    if not per_run:
        return p
    a = np.array(per_run)
    tc, mu = float(np.median(a[:, 0])), float(np.median(a[:, 1]))
    print(f"  hold ladder, both approaches ({len(per_run)} runs at "
          f"{', '.join(f'{v:.0f} V' for v in a[:, 3])}):")
    print(f"    tau_c {tc:.4f} N*m   mu_load {mu:.4f} N*m per N*m carried   "
          f"(spread {np.ptp(a[:, 0]):.4f} / {np.ptp(a[:, 1]):.4f})")

    # DECLINE if the hysteresis is at the noise floor, and decline LOUDLY: a
    # near-zero reading here is not "this gearbox is smooth", it is "the
    # measurement did not resolve". PRESENT_LOAD is quantised, and half the
    # difference of two quantised numbers has a floor of its own — on the light
    # arm (0.19 N*m of load against 0.19 of friction) the gravity column came
    # back non-monotonic, and a fit through it means nothing.
    #
    # Getting this wrong is worse than not running at all, because tau_c would
    # be seeded near ZERO over a perfectly reasonable prior, and the free swing
    # divides by (m*g*r*sin q0 - tau_c): measured on this file's own synthetic
    # data, which has no static friction by construction, that pushed J_m from
    # 0.017 to 0.023 and tau_c to a third of the truth. A seed that can be wrong
    # in that direction has to be able to say no.
    quantum = float(np.median(a[:, 4]))          # median friction, duty*volts
    biggest = float(np.median(a[:, 5]))          # median max load, N*m
    if tc <= 0.0 or mu <= 0.0 or tc + mu * biggest < 0.05 * biggest:
        print(f"    !! that is under 5 % of the {biggest:.2f} N*m this ladder "
              f"carries, i.e. no resolvable hysteresis. NOT seeding tau_c or "
              f"mu_load.\n"
              f"       On hardware this cannot happen — a real gearbox holding "
              f"{biggest:.2f} N*m has friction. It means either the duty column "
              f"is at its quantisation floor (a light arm), or the data came "
              f"from a model with no static friction, which actuator.py's "
              f"tanh(w/v_eps) is (see PLAN.md step 2b).")
        # mu_load goes to ZERO, not left at the Params default. This pass is its
        # only source, so declining means there is no evidence for it, and an
        # unmeasured 0.28 is not a safe fallback: the free swing is a
        # BACK-DRIVING run whose load varies through the fall, so an assumed
        # load-dependent friction lands straight on J_m. Measured on the
        # selftest, which is exactly this case: carrying the default pushed J_m
        # from 0.0047 to 0.0070 against a truth of 0.0042. Zero reproduces the
        # law as it was before this term existed, which actuator._selftest()
        # asserts explicitly.
        return A.Params(**{**p.__dict__, "mu_load": 0.0})
    if mu <= 0.0:
        print("    mu_load came out non-positive — friction is not growing with "
              "load in this data. Left at its prior rather than clamped to zero.")
        return A.Params(**{**p.__dict__, "tau_c": max(1e-4, tc)})
    # Friction that meets or exceeds the torque carried is a gearbox that cannot
    # transmit anything, and the bound in FIT says so; a value near it means the
    # ladder was measuring something other than friction.
    if mu > 0.9:
        print("    !! mu_load past 0.9: friction would exceed the load it "
              "carries. Not seeding it.")
        return A.Params(**{**p.__dict__, "tau_c": max(1e-4, tc)})
    return A.Params(**{**p.__dict__, "tau_c": max(1e-4, tc), "mu_load": mu})


def seed_from_holds(runs, p: A.Params) -> A.Params:
    """Steady-state torque balance: two linear regressions, no integration."""
    tau, cur, err, iu = [], [], [], []
    for r in runs:
        if r.trajectory != "hold":
            continue
        # the last 40 % of every dwell, where the transient is gone
        step = np.flatnonzero(np.abs(np.diff(r.target)) > 1e-6)
        edges = np.concatenate(([0], step + 1, [len(r.target)]))
        for a_, b_ in zip(edges[:-1], edges[1:]):
            n = b_ - a_
            if n < 20:
                continue
            sl = slice(a_ + int(0.6 * n), b_)
            q = float(np.mean(r.q[sl]))
            tau.append(abs(r.mass * 9.80665 * r.radius * math.sin(q)))
            cur.append(float(np.mean(np.abs(r.i[sl]))))
            err.append(abs(float(np.mean(r.target[sl] - r.q[sl]))))
            iu.append(cur[-1] / max(1e-6, float(np.mean(r.u_bat[sl]))))
    if len(tau) < 4:
        return p
    tau, cur, err, iu = map(np.asarray, (tau, cur, err, iu))
    # A hold only measures torque per amp if the MOTOR is holding the arm. Where
    # static friction is comparable to m*g*r the gearbox holds it instead: the
    # motor torque at rest is indeterminate anywhere in a band of width 2*tau_s,
    # the current falls to nothing, and the regression divides a full gravity
    # torque by ~0. This is a property of the RIG, not the servo, and no weighting
    # fixes it — the information is absent.
    #
    # Test it on the answer rather than on tau_c, which is still its prior this
    # early in the seeding: kt_eff times the stall current U/R is the implied
    # stall torque, and if that is far past the datasheet the holds are measuring
    # friction. Measured on the ST3215 bench: m*g*r = 0.380 N*m against tau_s in
    # 0.352..0.380, kt_eff = 10.3 N*m/A, implied stall 32 N*m against a spec 2.94.
    # Decline to seed and leave k_u at its prior, exactly as the reversal leaves
    # the dead zone when its threshold is never bracketed.
    good0 = np.asarray(cur) > 1e-3
    if good0.sum() >= 4:
        M0 = np.column_stack([np.asarray(cur)[good0], np.ones(int(good0.sum()))])
        kt0 = float(np.linalg.lstsq(M0, np.asarray(tau)[good0], rcond=None)[0][0])
        spec_stall = A.Params().k_u * 12.0
        implied = kt0 * 12.0 / p.R
        if implied > 3.0 * spec_stall:
            mgr_max = max((r.mass * 9.80665 * r.radius for r in runs
                           if r.trajectory == "hold"), default=0.0)
            print(f"  holds: torque per amp comes out {kt0:.2f} N*m/A, an implied "
                  f"stall of {implied:.1f} N*m against a spec {spec_stall:.2f} — "
                  f"friction, not the motor, is holding this arm\n"
                  f"         (m*g*r = {mgr_max:.3f} N*m). The torque constant is "
                  f"not identifiable from these holds; leaving it at its prior. "
                  f"Load the arm so m*g*r comfortably exceeds the joint's static "
                  f"friction and re-run.")
            return p

    d = dict(p.__dict__)
    # Both regressions carry an intercept, and leaving it out is not a detail.
    # The standing error is deadband + i*R/(kp*U): forcing the line through the
    # origin folds the dead zone into the slope and reads kp low by the ratio of
    # the two, which on this servo is a factor of two.
    good = cur > 1e-3
    if good.sum() >= 4:
        M = np.column_stack([cur[good], np.ones(good.sum())])
        kt_eff, c = np.linalg.lstsq(M, tau[good], rcond=None)[0]
        if kt_eff > 1e-3:
            d["k_u"] = float(max(0.02, min(2.0, kt_eff / p.R)))
            print(f"  holds: torque per amp = {kt_eff:.3f} N*m/A "
                  f"(intercept {c:+.3f})  ->  k_u*R = {kt_eff:.3f}")
    good = iu > 1e-6
    if good.sum() >= 4:
        M = np.column_stack([iu[good], np.ones(good.sum())])
        slope, c = np.linalg.lstsq(M, err[good], rcond=None)[0]
        if slope > 1e-9:
            d["kp"] = float(max(1.0, min(800.0, p.R / slope)))
            print(f"  holds: standing error per i/U = {slope:.4g} rad, dead zone "
                  f"intercept {math.degrees(c):+.2f} deg  ->  kp/R = {1/slope:.2f}")
        if 0 < c < 0.05:
            d["deadband"] = float(c)
    return A.Params(**d)


def seed_R_from_stationary(runs, p: A.Params) -> A.Params:
    """R from every sample where the joint is HELD STILL, at any duty.

    seed_from_saturation below needs the duty pinned at 1 so that the applied
    voltage is exactly the supply. There is a second place where the electrical
    equation collapses just as cleanly, and it is far easier to reach: standing
    still. At omega = 0 the back-EMF term vanishes on its own, so

        i = d * U / R

    with no gain, no dead zone, no friction and no speed in it. Regressing the
    measured current on d*U over the stationary samples returns 1/R directly, and
    unlike saturation it does not need the servo to be pushed to its limit — a
    hold ladder gives it at a dozen different duties.

    This exists because saturation turned out to be unreachable on a real bench.
    Measured 2026-09-08 over the full three-voltage set: max duty 0.68 at 12 V,
    0.80 at 10 V, and NINE genuinely pinned samples in the entire dataset, all in
    one 8 V run and all at one speed. With no run clearing the ten-sample bar,
    seed_from_saturation fell through to its position-error fallback — the one
    its own docstring warns "comes back an order of magnitude wrong" — and
    returned R = 20.2 ohm against a bench-measured 4.3.

    The holds have no such problem: R = d^2*U/i over the ladder lands in
    4.29..4.44 ohm across a 7x range of duty and two supply voltages.

    NOTE this gives R and not k_e: with omega pinned at zero there is nothing for
    the k_e/R column to act on. k_e still needs either genuine saturation across a
    spread of speeds, or a free-running no-load spin.
    """
    x, y = [], []
    for r in runs:
        if r.duty is None or not r.on.any():
            continue
        w = r.w_meas if r.w_meas is not None else np.gradient(r.q, r.dt)
        # Stationary, driving, and above the duty where the current LSB stops
        # meaning anything. The speed bar is one encoder count per sample at the
        # bench's 200 Hz, so "still" means still to the resolution of the log.
        still = r.on & (np.abs(w) < 0.05) & (r.duty > DUTY_MIN) & (r.i > 1e-3)
        if still.sum() < 10:
            continue
        x.append((r.duty * r.u_bat)[still]); y.append(np.abs(r.i[still]))
    if not x:
        print("  stationary: no held samples above the current LSB. R stays at "
              "its prior.")
        return p
    x, y = np.concatenate(x), np.concatenate(y)
    # Intercept carried, not forced through zero: a systematic offset in the
    # current channel would otherwise land entirely on the slope.
    M = np.column_stack([x, np.ones(len(x))])
    (g_R, c), *_ = np.linalg.lstsq(M, y, rcond=None)
    if not (1e-3 < g_R < 3.0):
        print(f"  stationary: 1/R came out at {g_R:.4g}, not physical; ignoring.")
        return p
    R = 1.0 / g_R
    print(f"  stationary: {len(x)} held samples -> 1/R = {g_R:.4f} "
          f"(R = {R:.2f} ohm, intercept {c:+.4f} A)")
    # k_u*R and kp/R are the well-measured products; hold them and let the
    # individual parameters follow the new R, exactly as saturation does.
    kt = p.k_u * p.R
    return A.Params(**{**p.__dict__, "R": R, "k_u": kt / R,
                       "kp": (p.kp / p.R) * R})


def seed_from_saturation(runs, p: A.Params) -> A.Params:
    """R and k_e from the moments the duty is pinned at 1, in one regression.

    When the commanded error is big enough that the inner loop saturates, the
    duty is exactly 1 and the applied voltage is exactly the supply. The current
    is then

        i = U_bat / R  -  (k_e / R) * omega

    with nothing else in it — no gain, no dead zone, no friction. Regressing the
    measured current on the supply voltage and the measured speed over those
    samples returns 1/R and k_e/R directly, and it is the ONLY place in the data
    where R appears on its own: everywhere else it hides inside k_u*R and kp/R,
    which is why the big steps are in the trajectory set and why the runs are
    taken at three voltages.

    This has to be a seed rather than something the global fit discovers, because
    the global fit uses a robust loss — and the saturated current spikes, being
    the largest residuals in the set, are exactly what a robust loss discounts.
    Left to itself the fit throws away the one measurement that pins R.
    """
    # If ANY run in the set actually pins the duty, trust that everywhere and let
    # the runs that never saturate contribute nothing. Falling back per-run is
    # worse than useless: the runs with no saturated samples are exactly the ones
    # whose position-error samples are unsaturated, so the fallback tops up the
    # regression with the very rows it is meant to exclude. Measured: 567 genuinely
    # pinned samples give R = 3.88 ohm, and topping them up to 928 with inferred
    # ones gives 7.92.
    have_duty = any(r.duty is not None and (r.duty > 0.99).sum() >= 10 for r in runs)
    if not have_duty and any(r.duty is not None for r in runs):
        # The position-error fallback below is only defensible when no run logged
        # a duty at all. When the duty IS logged and simply never pins, the honest
        # reading is "this bench cannot saturate this servo", and the fallback
        # actively harms: it admits fast-moving samples whose back-EMF is large,
        # which is the failure this docstring already describes. R comes from
        # seed_R_from_stationary instead; leave it alone here.
        print("  saturation: the duty is logged but never pins — this trajectory "
              "set does not saturate the servo. R is left to the stationary "
              "holds; k_e keeps its prior and needs a no-load spin.")
        return p
    i, w, u = [], [], []
    for r in runs:
        if not r.on.any():
            continue
        if have_duty and (r.duty is None or (r.duty > 0.99).sum() < 10):
            continue
        # Prefer the measured duty: "past saturation" is what this regression
        # needs, and a large position error only implies it when the joint cannot
        # slew away from the error. On a lightly loaded arm it usually can, so the
        # error test admits fast-moving low-current samples whose back-EMF term
        # is large -- and 1/R comes back an order of magnitude wrong.
        big = (r.duty > 0.99) if have_duty else (np.abs(r.target - r.q) > 0.15)
        big &= r.on & (np.abs(r.i) > 1e-3)
        if big.sum() < 10:
            continue
        # Speed enters this regression as the whole k_e/R column, and it trades
        # off directly against 1/R -- so its noise lands straight on R. Prefer the
        # servo's reported speed, which is the finer of the two at this rate (see
        # Run.w_meas); differentiate only when the log predates it.
        wq = r.w_meas if r.w_meas is not None else np.gradient(r.q, r.dt)
        i.append(np.abs(r.i[big])); w.append(np.abs(wq[big])); u.append(r.u_bat[big])
    if not i:
        print("  saturation: no samples with the duty pinned — the steps never "
              "saturated. R stays at its prior and k_u inherits that error.")
        return p
    i, w, u = np.concatenate(i), np.concatenate(w), np.concatenate(u)
    M = np.column_stack([u, -w])
    (g_R, g_ke), *_ = np.linalg.lstsq(M, i, rcond=None)
    if not (1e-3 < g_R < 3.0):
        print(f"  saturation: 1/R came out at {g_R:.4g}, which is not physical; "
              f"ignoring it.")
        return p
    R = 1.0 / g_R
    k_e = max(0.05, g_ke * R)
    print(f"  saturation: {len(i)} pinned samples -> 1/R = {g_R:.4f} (R = {R:.2f} "
          f"ohm), k_e/R = {g_ke:.4f} (k_e = {k_e:.2f} V s/rad)")
    # k_u*R is the well-measured product; hold it and let k_u follow the new R.
    kt = p.k_u * p.R
    return A.Params(**{**p.__dict__, "R": R, "k_e": k_e, "k_u": kt / R,
                       "kp": (p.kp / p.R) * R})


def seed_from_reversal(runs, p: A.Params) -> A.Params:
    """The smallest commanded change that actually moves the joint.

    Read straight off the reversal trajectory: for each commanded step, did the
    joint move by more than a couple of encoder counts? The largest command that
    produced nothing and the smallest that produced something bracket the dead
    zone plus stiction, and the midpoint is a far better estimate of `deadband`
    than an optimiser will find — its gradient there is a staircase, since the
    error it is compared against has already been quantised.

    This threshold is also the single most policy-relevant number on the servo.
    It is the resolution of the robot's actuation: a commanded correction smaller
    than this does nothing at all, so a policy that has learnt to make fine
    adjustments in simulation loses them entirely on hardware unless the same
    dead zone was there while it learnt.
    """
    moved, still = [], []
    for r in runs:
        if r.trajectory != "reversal" or not r.on.any():
            continue
        edge = np.flatnonzero(np.abs(np.diff(r.target)) > 1e-9) + 1
        for k, e in enumerate(edge):
            nxt = edge[k + 1] if k + 1 < len(edge) else len(r.target)
            if nxt - e < 20:
                continue
            d_cmd = abs(float(r.target[e] - r.target[e - 1]))
            settle = slice(e + int(0.6 * (nxt - e)), nxt)
            d_q = abs(float(np.mean(r.q[settle]) - r.q[e - 1]))
            (moved if d_q > 2 * A.ENC_STEP_RAD else still).append(d_cmd)
    if not moved or not still:
        # Only one side of the bracket: every commanded reversal moved the joint,
        # or none did. That is not a measurement of the threshold, it is a
        # statement that the trajectory did not reach it. Say so rather than
        # taking the smallest command tried as the answer — that reads the dead
        # zone as whatever the experiment happened to stop at.
        print(f"  reversal: {len(moved)} commands moved the joint, {len(still)} did "
              f"not — the threshold is not bracketed, so the dead zone is left at "
              f"its prior. Run the full reversal trajectory (it ends at 0.1 deg).")
        return p
    hi_still, lo_moved = max(still), min(moved)
    thr = 0.5 * (hi_still + lo_moved)
    print(f"  reversal: largest command that moved nothing "
          f"{math.degrees(hi_still):.2f} deg, smallest that moved "
          f"{math.degrees(lo_moved):.2f} deg  ->  dead zone ~ "
          f"{math.degrees(thr):.2f} deg ({thr/A.ENC_STEP_RAD:.1f} encoder counts)")
    return A.Params(**{**p.__dict__, "deadband": float(min(0.05, thr))})


# ------------------------------------------------------------------- fit
def to_vec(p: A.Params) -> np.ndarray:
    return np.array([p.k_u * p.R, p.kp / p.R, p.k_e / p.R, 1.0 / p.R,
                     p.J_m, p.tau_c, p.b_v, p.mu_load,
                     p.deadband, p.punch, p.theta_bl])


def to_params(x, base: A.Params) -> A.Params:
    x = [float(min(hi, max(lo, v))) for (_, lo, hi), v in zip(FIT, x)]
    kt, g_kp, g_ke, g_R = x[:4]
    R = 1.0 / g_R
    d = dict(base.__dict__)
    d.update(R=R, kp=g_kp * R, k_e=g_ke * R, k_u=kt * g_R,
             J_m=x[4], tau_c=x[5], b_v=x[6], mu_load=x[7],
             deadband=x[8], punch=x[9], theta_bl=x[10])
    return A.Params(**d)


def _oscillates(runs) -> bool:
    for r in runs:
        q = r.q[~r.on]
        if len(q) < 50:
            continue
        c = np.sign(q - float(np.mean(q[-len(q) // 4:])))
        if len(np.flatnonzero((c[:-1] < 0) & (c[1:] >= 0))) >= 2:
            return True
    return False


def residual_free(x, base, runs, dt_int):
    """The free-swing runs, over J_m, tau_c and b_v alone.

    With Torque Enable off there is no duty, no current and no back-EMF, so
    exactly three parameters enter and the other seven are invisible. Fitting
    them here rather than in the global pass is not a convenience: in the global
    pass this run's residual is the largest in the set and it is dominated by
    PHASE — a pendulum whose period is 2 % wrong accumulates a half cycle of
    error in twenty five swings, which reads as an enormous position residual and
    drags the optimiser away from parameters that were already right. Alone, over
    three parameters, it is a well-conditioned little problem.
    """
    d = dict(base.__dict__)
    d["J_m"], d["tau_c"] = float(abs(x[0])), float(abs(x[1]))
    if len(x) > 2:
        d["b_v"] = float(abs(x[2]))
    p = A.Params(**d)
    return np.concatenate([(r.predict(p, dt_int)["q"] - r.q) / W_POS for r in runs])


def residual(x, base, runs, dt_int):
    """Model against measurement, position and current, weighted to comparable size.

    Note `s["q"]`, not `s["q_meas"]`: the prediction is NOT re-quantised. The
    encoder step is measurement noise on the data, and rounding the model too
    makes the objective a staircase — a finite-difference step smaller than
    0.088 deg then moves no sample at all, the Jacobian column comes back zero,
    and least_squares reports convergence without having taken a step. That is
    not a subtle loss of accuracy; it is the difference between fitting and
    returning the initial guess.
    """
    p = to_params(x, base)
    out = []
    for r in runs:
        s = r.predict(p, dt_int)
        out.append((_smooth(s["q"]) - _smooth(r.q)) / W_POS)
        if r.on.any():
            m = r.on
            out.append((_smooth(np.abs(s["i"]))[m] - _smooth(np.abs(r.i))[m]) / W_CUR)
    return np.concatenate(out)


def rms(base, runs, dt_int):
    p = base
    rows = []
    for r in runs:
        s = r.predict(p, dt_int)
        rows.append((r.name, math.degrees(float(np.sqrt(np.mean(
            (s["q"] - r.q) ** 2)))),
            float(np.sqrt(np.mean((np.abs(s["i"]) - np.abs(r.i)) ** 2)))))
    return rows


def fit(runs, holdout=None, dt_int=1e-4, max_nfev=200, base=None,
        n_refine=5, refine_pass=False):
    base = base or A.Params()
    train = [r for r in runs if holdout is None or holdout not in r.trajectory]
    test = [r for r in runs if r not in train]
    print(f"\nseeding from {len(train)} runs "
          f"({', '.join(sorted({r.trajectory for r in train}))})")
    # First: it depends on no other parameter, and the free swing below
    # SUBTRACTS tau_c to get its driving torque — it had been using a value
    # roughly half of what the bidirectional ladder measures.
    base = seed_from_holdbi(train, base)
    base = seed_from_freeswing(train, base)
    base = seed_R_from_stationary(train, base)  # R where omega = 0, any duty
    base = seed_from_saturation(train, base)   # R first: the holds divide by it
    base = seed_from_holds(train, base)
    base = seed_from_reversal(train, base)

    free = [r for r in train if r.trajectory == "freeswing"]
    driven = [r for r in train if r.trajectory != "freeswing"]
    if free:
        if len({round(r.J_load, 6) for r in free}) < 2:
            print("  !! only one free-swing arm. J_m and tau_c both scale the "
                  "release, so one arm cannot separate them: a heavier J_m and a "
                  "smaller tau_c fit the same curve. Run the free swing with TWO "
                  "different arms — that is what makes this pass identifiable.")
        # A released arm that oscillates carries enough information for three
        # parameters. One that just falls and stops does not: an overdamped
        # first-order fall is matched about equally well by a small inertia with
        # heavy viscous friction and by a large one with light, and the optimiser
        # will happily drive J_m to the floor to get there. Observed on synthetic
        # data with a KNOWN answer: J_m came back ten times too small with b_v
        # taking up the slack. So when there is no oscillation, b_v is held out
        # of this pass — the driven runs identify it, at speeds where it matters.
        osc = _oscillates(free)
        n_free = 3 if osc else 2
        print(f"  free swing: the arm {'oscillates' if osc else 'does not oscillate'}"
              f", fitting {n_free} parameter{'s' if n_free > 1 else ''}"
              + ("" if osc else " (b_v left to the driven runs)"))
        for it in range(2):        # one refinement: the analytic J_m seed used a
            x0f = [base.J_m, base.tau_c, base.b_v][:n_free]   # prior tau_c
            sol = least_squares(
                residual_free, x0f,
                bounds=([1e-4, 1e-5, 1e-6][:n_free], [0.2, 1.0, 1.0][:n_free]),
                x_scale=[0.005, 0.05, 0.02][:n_free], diff_step=1e-2,
                loss="soft_l1", f_scale=3.0, ftol=1e-12, xtol=1e-12,
                max_nfev=150, args=(base, free, dt_int))
            d = {**base.__dict__, "J_m": float(sol.x[0]), "tau_c": float(sol.x[1])}
            if n_free > 2:
                d["b_v"] = float(sol.x[2])
            base = A.Params(**d)
            if it == 0 and not osc:
                base = seed_from_freeswing(free, base)   # redo it with a real tau_c
        print(f"  free swing fit: J_m {base.J_m:.5f}, tau_c {base.tau_c:.4f}, "
              f"b_v {base.b_v:.4f}")
    train = driven or train

    print(f"\nanalytic passes done over {sum(len(r.t) for r in train)} samples")
    if not refine_pass:
        # The analytic passes above ARE the identification: each parameter they
        # produce comes from an experiment that isolates it. The trajectory
        # refinement below is optional, and on synthetic data with a known answer
        # it did not improve on them — 200 Powell evaluations came back with a
        # cost 3.5x worse than the seeds and the guard threw the result away.
        # That is a statement about matching switching dynamics sample by sample,
        # not about the servo. Pass --refine to run it anyway; it is guarded, so
        # it can decline but not damage.
        p = A.Params(**base.__dict__)
        p.fitted = True
        p.servo_ids = tuple(sorted({int(r.meta.get("servo_id", 0)) for r in runs}))
        p.source = (f"fit_bam.py analytic passes over {len(train)} runs, "
                    f"{sorted({round(float(r.meta.get('psu_volts', 0)), 1) for r in train})} V, "
                    f"holdout={holdout or 'none'}, no trajectory refinement")
        return p, train, test

    print(f"refining {len(FIT)} parameters ...")
    lo = np.array([l for _, l, _ in FIT])
    hi = np.array([h for _, _, h in FIT])
    # Indexed BY NAME, not by position. Both of the lines below used to be
    # literal offsets into FIT, and inserting mu_load in the middle of it broke
    # them in different ways: `scale` stayed ten long against an eleven-long
    # vector and Powell died on the broadcast, while the punch nudge silently
    # started nudging the DEAD ZONE instead — the same edit, one crash and one
    # wrong answer. A list whose order is load-bearing should not be indexed by
    # counting.
    at = {name: k for k, (name, _, _) in enumerate(FIT)}
    x0 = np.clip(to_vec(base), lo, hi)
    x0[at["punch"]] = max(x0[at["punch"]], 0.01)   # off the bound, for a gradient
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)

    # Characteristic magnitude of each fit variable. Powell searches along
    # directions, so this is what stops it from taking a step in g_kp (order 10)
    # that is meaningless in J_m (order 0.005).
    SCALE = {"kt_eff": 0.5, "g_kp": 2.0, "g_ke": 0.2, "g_R": 0.05, "J_m": 0.005,
             "tau_c": 0.05, "b_v": 0.02, "mu_load": 0.05, "deadband": 0.003,
             "punch": 0.03, "theta_bl": 0.005}
    missing = [n for n, _, _ in FIT if n not in SCALE]
    if missing:
        raise SystemExit(f"no Powell step size for {missing}; add it to SCALE")
    scale = np.array([SCALE[n] for n, _, _ in FIT])

    # The refinement runs on a SUBSET: one run per (trajectory, voltage) family,
    # at most `n_refine` of them. Every evaluation integrates every run at 0.1 ms,
    # so the cost is wall-clock seconds of data times evaluations, and twelve runs
    # times four hundred evaluations is twenty minutes for a pass whose job is to
    # polish what the analytic passes already found. Diversity buys more here than
    # volume: one hold, one step and one triangle at different voltages constrain
    # more than five holds do.
    seen, refine = set(), []
    for r in sorted(train, key=lambda r: r.trajectory):
        key = (r.trajectory, round(float(r.meta.get("psu_volts", 0)), 1))
        if key in seen:
            continue
        seen.add(key)
        refine.append(r)
    refine = refine[:n_refine] or train
    print("  refining on %d of %d runs: %s" % (len(refine), len(train), ", ".join(
        "%s@%sV" % (r.trajectory, r.meta.get("psu_volts")) for r in refine)))

    # NOT least_squares, and the reason is worth recording because it looks
    # like the obvious tool. A trajectory residual over this model is not smooth
    # in the parameters: the dead zone, the punch, the backlash engagement and
    # the duty saturation are all switches, and a parameter change that moves a
    # saturated step edge by one sample changes the residual there by an amp.
    # Finite differences across that read a first-order optimality of ~1e9, the
    # trust region collapses to steps of 1e-6, and trf reports convergence after
    # thirteen evaluations with a step of exactly zero. Measured, on synthetic
    # data with a known answer, with and without smoothing.
    #
    # Powell does line searches along directions and never differentiates, so a
    # spiky objective costs it accuracy rather than the whole run. It is slower
    # per unit of progress, which is affordable here because the analytic passes
    # above have already done the identification — this pass is a refinement, and
    # it is guarded below so that it can only help.
    from scipy.optimize import minimize

    def cost(x):
        r = residual(x, base, refine, dt_int)
        return float(np.sum(f_scale ** 2 * (np.sqrt(1.0 + (r / f_scale) ** 2) - 1.0)))

    f_scale = 3.0
    c0 = cost(x0)
    sol = minimize(cost, x0, method="Powell", bounds=list(zip(lo, hi)),
                   options=dict(maxfev=max_nfev, xtol=1e-3, ftol=1e-4,
                                direc=np.diag(scale)))
    print(f"  {sol.nfev} evaluations, cost {c0:.4g} -> {sol.fun:.4g}")
    if sol.fun >= c0:
        print("  the refinement did not improve on the seeds; keeping them.")
        sol.x = x0
    p = to_params(np.asarray(sol.x), base)
    p.fitted = True
    p.servo_ids = tuple(sorted({int(r.meta.get("servo_id", 0)) for r in runs}))
    p.source = (f"fit_bam.py over {len(train)} runs, "
                f"{sorted({round(float(r.meta.get('psu_volts', 0)), 1) for r in train})} V, "
                f"holdout={holdout or 'none'}, cost={float(sol.fun):.1f}")
    return p, train, test


# -------------------------------------------------------------- self-test
def _selftest(dt_int=1e-4) -> int:
    """Generate data from a KNOWN servo, then see if the fit finds it again.

    This is the only check available before hardware exists, and it is worth
    more than it looks: it tests the trajectory set as much as the optimiser.
    If a parameter cannot be recovered from noise-free data generated by the
    very model being fitted, no amount of real data will recover it either —
    the experiment design is wrong, not the servo.
    """
    # mu_load = 0 here, and it is the one parameter this test CANNOT exercise.
    # Not an oversight and not a value chosen to be easy to recover: the term is
    # measured from the difference between a hold approached from below and one
    # approached from above, and this model has no static friction to create that
    # difference — actuator._sign() is tanh(w/v_eps), which is exactly zero at
    # rest (PLAN.md step 2b). So a non-zero mu_load here would be generated into
    # the data, be invisible to the pass that exists to measure it, and land on
    # J_m through the free swing instead: measured, J_m went 0.0042 -> 0.0070.
    # Setting it to zero keeps this test honest about what it covers. The
    # hardware number (0.286, three voltages) is in
    # ST3215_STS3215_measured_parameters.md, and bench/hysteresis.py reproduces
    # it from the raw csv without going through any of this file.
    truth = A.Params(R=6.2, k_e=2.10, k_u=0.190, J_m=0.0042, tau_c=0.075,
                     b_v=0.030, mu_load=0.0, kp=48.0,
                     deadband=3 * A.ENC_STEP_RAD,
                     punch=0.03, theta_bl=math.radians(0.42))
    rng = np.random.default_rng(0)
    tmp = os.path.join(os.environ.get("TMPDIR", "/tmp"), "bam_selftest")
    os.makedirs(tmp, exist_ok=True)
    for f in os.listdir(tmp):
        os.remove(os.path.join(tmp, f))

    hz, dt = 200.0, 1.0 / 200.0
    plans = []
    for volts, mass, radius in ((12.6, 0.25, 0.10), (11.1, 0.50, 0.15),
                                (9.9, 0.25, 0.15)):
        plans += [("hold", volts, mass, radius), ("step", volts, mass, radius),
                  ("chirp", volts, mass, radius)]
    # the free swing gets the light short arm, for the reason
    # seed_from_freeswing() explains; the holds and the triangle get the heavy
    # long one, where torque is what is being measured
    plans += [("freeswing", 12.6, 0.25, 0.06), ("freeswing", 12.6, 0.50, 0.12),
              ("triangle", 12.6, 0.50, 0.15), ("reversal", 12.6, 0.50, 0.15)]
    # The bidirectional ladder, on the heavy arm at all three voltages, because
    # it is the only run that carries mu_load. It also has to survive the 8 s
    # truncation below, which would cut the descending pass off entirely and
    # leave the new term untested by the test that exists to test it — so it
    # runs at a shorter DWELL rather than for a shorter time.
    plans += [("holdbi", v, 0.50, 0.15) for v in (12.6, 11.1, 9.9)]

    import sweep
    for traj, volts, mass, radius in plans:
        if traj == "holdbi":
            name, T, fn, torque_all = sweep.traj_holdbi(1.4, dwell=0.8)
        else:
            name, T, fn, torque_all = sweep.TRAJ[traj](1.4)
            T = min(T, 8.0)
        t = np.arange(0, T, dt)
        target = np.array([fn(float(x)) for x in t])
        on = np.ones(len(t), bool)
        if not torque_all:
            on[int(1.0 / dt):] = False
        J_load = mass * radius ** 2
        p = A.Params(**{**truth.__dict__, "J_l": J_load})
        s = A.simulate(p, target, dt, q0=float(target[0]), u_bat=volts,
                       load_torque=A.pendulum_load(mass, radius),
                       torque_on=on, dt_int=dt_int)
        duty_s = np.asarray(s["u"]) / float(volts)      # applied volts -> duty
        rows = [dict(t=float(t[k]),
                     target_rad=float(target[k]) if on[k] else float("nan"),
                     q_rad=float(s["q_meas"][k]), w_rad_s=float(s["w"][k]),
                     # emulate the REGISTER, not the motor: supply current is
                     # duty*I_motor, and load_raw is that duty in per-mille,
                     # sign-magnitude with bit 10 the direction.  Writing the
                     # motor current here instead would leave the correction in
                     # Run.__init__ untested by the one test that runs every time.
                     # ... and quantised the way the register is.  A 4 mA
                     # Gaussian on the SUPPLY side is not the same noise: it is
                     # 2x the LSB's own sigma, and Run divides by duty, so it
                     # arrives at the fit amplified by 1/d.  The real channel's
                     # error is bounded and often exactly zero.
                     current_a=float(round(abs(s["i"][k]) * abs(duty_s[k])
                                           / CURRENT_LSB) * CURRENT_LSB
                                     + rng.normal(0, 0.001)),
                     volt_v=volts, temp_c=35,
                     # SIGNED, because that is what sweep.py writes: the sign
                     # bit is applied in Servo.decode(), not left in the column.
                     load_raw=int(round(duty_s[k] * 1000)),
                     counts=round(s["q_meas"][k] / A.ENC_STEP_RAD) + 2048)
                for k in range(len(t))]
        runlog.write(os.path.join(tmp, f"{name}_v{volts}_m{mass}_r{radius}.csv"),
                     dict(servo_id=1, psu_volts=volts, mass_kg=mass,
                          radius_m=radius, arm_inertia=0.0, trajectory=name,
                          registers={"P_COEF": 32}, synthetic=True), rows)

    runs = load_runs(tmp, seconds=8.0, fit_hz=hz)
    print(f"synthetic: {len(runs)} runs, "
          f"{sorted({r.meta['psu_volts'] for r in runs})} V")
    for pr in check_identifiable(runs):
        print("  !! " + pr)
    p, train, test = fit(runs, holdout="chirp", dt_int=dt_int)

    # Two blocks, and the split is the point of the test rather than a way of
    # grading on a curve. The first block is what the analytic passes measure:
    # each of those parameters has an experiment that isolates it, and if one of
    # them comes back wrong the experiment or the code is broken. The second is
    # what THIS trajectory set does not pin down — and since the set is generated
    # here, from a model with a known answer, a parameter that cannot be
    # recovered from noise-free data will not be recovered from real data either.
    # That is a statement about the bench procedure, not about the optimiser, and
    # printing it is more useful than a tolerance loose enough to pass.
    ok = True
    identified = {"R": .25, "k_e": .30, "k_u": .20, "k_u*R": .12,
                  "kp": .35, "J_m": .35, "tau_c": .40}
    weak = {
        "mu_load": "the bidirectional ladder measures it on HARDWARE (0.25-0.30 "
                   "across three voltages, robot/bench/hysteresis.py) but it is "
                   "not recoverable from this model's own output, and that is a "
                   "statement about the MODEL rather than the experiment. "
                   "actuator._sign() is tanh(w/v_eps), which is exactly zero at "
                   "rest, so the simulated servo has no static friction: a hold "
                   "approached from below and from above settles at the same "
                   "duty (measured, half-difference 0.00 and 0.03 V against the "
                   "real servo's 0.29), and the approach-dependent difference "
                   "the pass reads is not there to be read. The term is still "
                   "live in every run where the joint MOVES, which is where it "
                   "takes the load off k_u. Fixing the rest case needs a "
                   "friction that can hold at w = 0 - see PLAN.md step 2b",
        "b_v": "the free swing does not oscillate at these friction levels, so "
               "it carries no viscous information, and the driven runs see it "
               "only through the smoothed residual",
        "deadband": "the 8 s window used here stops before the reversal "
                    "trajectory reaches its small commands, so the threshold is "
                    "never bracketed. The full run goes down to 0.1 deg",
        "punch": "visible in the current just outside the dead zone, which the "
                 "same truncation removes",
        "theta_bl": "read off the hysteresis at a loaded reversal; the truncated "
                    "triangle here has one crossing",
    }

    def get(o, n):
        return o.k_u * o.R if n == "k_u*R" else getattr(o, n)

    print(f"\nidentified — each of these has an experiment that isolates it")
    print(f"{'parameter':<12}{'truth':>10}{'fitted':>10}{'error':>9}")
    for n, t in identified.items():
        t_, f_ = get(truth, n), get(p, n)
        e = abs(f_ - t_) / max(abs(t_), 1e-9)
        good = e <= t
        ok &= good
        print(f"{n:<12}{t_:10.4f}{f_:10.4f}{e*100:8.1f}%"
              + ("" if good else f"   FAIL (>{t*100:.0f}%)"))
    print(f"\nnot identified by this trajectory set — reported, not asserted")
    for n, why in weak.items():
        t_, f_ = get(truth, n), get(p, n)
        e = abs(f_ - t_) / max(abs(t_), 1e-9)
        print(f"{n:<12}{t_:10.4f}{f_:10.4f}{e*100:8.1f}%   {why}")

    print(f"\n{'run':<34}{'pos RMS deg':>12}{'current RMS A':>15}")
    for name, rp, ri in rms(p, train, dt_int):
        print(f"  fit  {name:<28}{rp:11.3f}{ri:15.4f}")
    for name, rp, ri in rms(p, test, dt_int):
        print(f"  HELD {name:<28}{rp:11.3f}{ri:15.4f}")
    print("\n  " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


# ------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(ROOT, "rl", "params", "st3215.json"))
    ap.add_argument("--holdout", default="chirp",
                    help="trajectory name kept out of the objective")
    ap.add_argument("--seconds", type=float, default=25.0,
                    help="seconds used from each run; the cost is wall-clock, "
                         "not samples. 25 covers the reversal (21 s), which is "
                         "where the dead zone lives — see the note in fit()")
    ap.add_argument("--fit-hz", type=float, default=200.0)
    ap.add_argument("--dt-int", type=float, default=1e-4)
    ap.add_argument("--max-nfev", type=int, default=200)
    ap.add_argument("--n-refine", type=int, default=5,
                    help="runs used in the refinement pass")
    ap.add_argument("--refine", action="store_true",
                    help="run the Powell trajectory refinement after the analytic "
                         "passes (slow, guarded, and on synthetic data it declined)")
    ap.add_argument("--plot", default=None, help="write a comparison png here")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()

    if a.selftest:
        raise SystemExit(_selftest(a.dt_int))

    runs = load_runs(a.data, a.seconds, a.fit_hz)
    print(f"{len(runs)} runs from {a.data}")
    problems = check_identifiable(runs)
    for pr in problems:
        print("\n!! " + pr)
    p, train, test = fit(runs, a.holdout, a.dt_int, a.max_nfev,
                         n_refine=a.n_refine, refine_pass=a.refine)

    print(f"\n{'run':<34}{'pos RMS deg':>12}{'current RMS A':>15}")
    for name, rp, ri in rms(p, train, a.dt_int):
        print(f"  fit  {name:<28}{rp:11.3f}{ri:15.4f}")
    held = rms(p, test, a.dt_int)
    for name, rp, ri in held:
        print(f"  HELD {name:<28}{rp:11.3f}{ri:15.4f}")
    if held:
        p.rms_pos_deg = float(np.mean([r[1] for r in held]))
        p.rms_current_a = float(np.mean([r[2] for r in held]))
        fit_pos = float(np.mean([r[1] for r in rms(p, train, a.dt_int)]))
        if p.rms_pos_deg > 2.5 * max(fit_pos, 1e-6):
            print("\n!! the held-out run is much worse than the fitted ones: the fit "
                  "has absorbed noise or the trajectory set is too narrow.")

    print(f"\nfitted:")
    # FIT names the fit-space coordinates (kt_eff, g_kp, g_ke, g_R); `p` is a
    # Params, which carries the physical ones. Report both: the physical values
    # are what ships, and the combinations are what the bench actually measured,
    # so a reader can tell a well-determined ratio from a split that rests on a
    # poorly-conditioned g_R.
    for n in ("R", "k_e", "k_u", "kp", "J_m", "tau_c", "b_v", "mu_load",
              "deadband", "punch", "theta_bl"):
        print(f"  {n:<12} {getattr(p, n):.5f}")
    print(f"  {'kt_eff':<12} {p.k_u * p.R:.5f}   (= k_u*R, the hold slope)")
    print(f"  {'g_kp':<12} {p.kp / p.R:.5f}   (= kp/R)")
    print(f"  {'g_ke':<12} {p.k_e / p.R:.5f}   (= k_e/R)")
    print(f"  {'g_R':<12} {1.0 / p.R:.5f}   (= 1/R, from the saturated steps)")
    print(f"  {'k_w':<12} {p.k_w:.5f}   (derived, = k_u*k_e)")
    print(f"  stall  {p.stall_torque(12.0):.2f} N*m @ 12 V, "
          f"{p.stall_torque(9.9):.2f} @ 9.9 V   (spec 2.94 @ 12)")
    print(f"  no load {p.no_load_speed(12.0):.2f} rad/s @ 12 V   (spec 4.71)")
    print(f"  backlash {math.degrees(p.theta_bl):.2f} deg   (spec <= 0.5)")
    print(f"  J_m {p.J_m:.5f} kg m^2 — this is MuJoCo's `armature`, and it was "
          f"{A.Params().J_m:.5f} as a guess")
    if problems:
        p.source += "  [UNDER-DETERMINED: " + "; ".join(
            s.split(".")[0] for s in problems) + "]"
        print("\n!! writing the fit anyway, with the identifiability problems "
              "recorded in `source`. Read them before training on this.")
    p.to_json(a.out)
    print(f"\nwrote {os.path.relpath(a.out, ROOT)}")
    if a.plot:
        _plot(p, runs, a.dt_int, a.plot)


def _plot(p, runs, dt_int, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("--plot needs matplotlib (uv sync --extra fit)")
        return
    n = len(runs)
    fig, ax = plt.subplots(2, n, figsize=(3.2 * n, 5), squeeze=False)
    for k, r in enumerate(runs):
        s = r.predict(p, dt_int)
        ax[0][k].plot(r.t, np.degrees(r.q), lw=.8, label="measured")
        ax[0][k].plot(r.t, np.degrees(s["q_meas"]), lw=.8, label="model")
        ax[0][k].set_title(r.trajectory, fontsize=9)
        ax[1][k].plot(r.t, np.abs(r.i), lw=.8)
        ax[1][k].plot(r.t, np.abs(s["i"]), lw=.8)
        ax[1][k].set_xlabel("s")
    ax[0][0].set_ylabel("deg")
    ax[1][0].set_ylabel("A")
    ax[0][0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
