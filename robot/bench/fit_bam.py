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
       ("deadband", 1e-5, 0.05), ("punch", 0.0, 0.5), ("theta_bl", 0.0, 0.05)]

W_POS = math.radians(0.3)     # rad of position error worth one unit of residual
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

    def __init__(self, name, meta, cols, seconds, fit_hz):
        t = np.asarray(cols["t"])
        keep = t <= (t[0] + seconds)
        t = t[keep]
        dt = 1.0 / fit_hz
        grid = np.arange(t[0], t[-1], dt)
        g = lambda c: np.interp(grid, t, np.asarray(cols[c])[keep])
        self.name, self.meta = name, meta
        self.t, self.dt = grid, dt
        self.q = g("q_rad")
        self.i = g("current_a")
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
        self.load = A.pendulum_load(self.mass, self.radius)

    def predict(self, p: A.Params, dt_int):
        q = A.Params(**{**p.__dict__, "J_l": max(self.J_load, 1e-6)})
        return A.simulate(q, self.target, self.dt, q0=self.q[0], u_bat=self.u_bat,
                          load_torque=self.load, torque_on=self.on, dt_int=dt_int)


def load_runs(d, seconds, fit_hz):
    runs = []
    for name, meta, cols in runlog.load_dir(d):
        if not cols["t"]:
            continue
        runs.append(Run(name, meta, cols, seconds, fit_hz))
    if not runs:
        raise SystemExit(f"no csv in {d}")
    return runs


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
    i, w, u = [], [], []
    for r in runs:
        if not r.on.any():
            continue
        big = np.abs(r.target - r.q) > 0.15          # well past saturation
        big &= r.on & (np.abs(r.i) > 1e-3)
        if big.sum() < 10:
            continue
        # r.q is measured; differentiate it for speed rather than trusting the
        # servo's own Present Speed, which is coarse and lags.
        wq = np.gradient(r.q, r.dt)
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
                     p.J_m, p.tau_c, p.b_v, p.deadband, p.punch, p.theta_bl])


def to_params(x, base: A.Params) -> A.Params:
    x = [float(min(hi, max(lo, v))) for (_, lo, hi), v in zip(FIT, x)]
    kt, g_kp, g_ke, g_R = x[:4]
    R = 1.0 / g_R
    d = dict(base.__dict__)
    d.update(R=R, kp=g_kp * R, k_e=g_ke * R, k_u=kt * g_R,
             J_m=x[4], tau_c=x[5], b_v=x[6],
             deadband=x[7], punch=x[8], theta_bl=x[9])
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
    base = seed_from_freeswing(train, base)
    base = seed_from_holds(train, base)
    base = seed_from_saturation(train, base)
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
    x0 = np.clip(to_vec(base), lo, hi)
    x0[8] = max(x0[8], 0.01)          # punch off the bound, so it has a gradient
    x0 = np.clip(x0, lo + 1e-9, hi - 1e-9)

    # Characteristic magnitude of each fit variable. Powell searches along
    # directions, so this is what stops it from taking a step in g_kp (order 10)
    # that is meaningless in J_m (order 0.005).
    scale = np.array([0.5, 2.0, 0.2, 0.05, 0.005, 0.05, 0.02, 0.003, 0.03, 0.005])

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
    truth = A.Params(R=6.2, k_e=2.10, k_u=0.190, J_m=0.0042, tau_c=0.075,
                     b_v=0.030, kp=48.0, deadband=3 * A.ENC_STEP_RAD,
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

    import sweep
    for traj, volts, mass, radius in plans:
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
        rows = [dict(t=float(t[k]),
                     target_rad=float(target[k]) if on[k] else float("nan"),
                     q_rad=float(s["q_meas"][k]), w_rad_s=float(s["w"][k]),
                     current_a=float(abs(s["i"][k]) + rng.normal(0, 0.004)),
                     volt_v=volts, temp_c=35, load_raw=0,
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
    ap.add_argument("--seconds", type=float, default=8.0,
                    help="seconds used from each run; the cost is wall-clock, "
                         "not samples")
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
    for n, _, _ in FIT:
        print(f"  {n:<12} {getattr(p, n):.5f}")
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
