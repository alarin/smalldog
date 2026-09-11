"""
hysteresis.py — read the two things a one-directional bench run cannot see.

    python bench/hysteresis.py                  # every voltage in bench/data
    python bench/hysteresis.py --volts 12

`sweep.py`'s `holdbi` and `speed` ladders exist to measure friction directly
instead of letting a fit infer it, and both measurements are a DIFFERENCE between
two runs of the same trajectory rather than a number read off one. That is why
they get their own reader: `fit_bam.py` fits the whole model at once and its
analytic passes select on `freeswing` and `hold`, so nothing in it looks at a
pair of segments and subtracts them.

**The hold ladder, both approaches.** A static hold does not sit at zero net
torque; it sits wherever friction happens to balance the rest, so the settled
duty depends on which side the joint arrived from. Half the DIFFERENCE between
the two approaches is the friction mobilised at that load, and the MEAN is the
motor torque with friction cancelled out. Two consequences, both measured:

  * friction grows with load — about 0.28 N*m per N*m of load carried, on top of
    a 0.19 N*m floor — which is the term `rl/actuator.py` has no room for, and
    which is why its fit inflates `k_u` until the implied stall is 4.23 N*m
    against a spec 2.94;
  * the position-loop stiffness read off ONE approach is too soft, because it
    bills the friction band to elasticity. 40.9 N*m/rad at 12 V here, against the
    28.8 the one-directional ladder gave.

**The speed ladder.** The same triangle at five periods. Averaging the up-ramp
and down-ramp duties cancels gravity, so what is left against speed is the
intercept (kinetic Coulomb) and the slope. That slope is `b_v + k_u*k_e` and this
rig CANNOT split it: back-EMF and viscous friction both cost a motor voltage
proportional to omega and neither depends on supply, so the three-voltage sweep
that separates the other electrical terms does nothing here. Splitting them needs
the motor current, and PRESENT_CURRENT gives it only as d^2*U/R at a 6.5 mA LSB.
The sum is reported, and it is reported as a sum.

Everything is computed in motor-volt units (duty x supply) and then converted with
the run's own `k_t_eff`, so each voltage is an independent measurement of the same
physical quantity. Rows that disagree across voltage are the ones to distrust.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import runlog                                                        # noqa: E402
import sweep                                                         # noqa: E402

#: Standard gravity and the trajectory set both come from elsewhere in this
#: directory, so that this file and `fit_bam.py` cannot read the same csv and
#: disagree about what is in it. `G` was 9.81 here against 9.80665 there.
G = runlog.G
#: Dwell of one rung of traj_holdbi, and how much of it to discard as transient.
DWELL, SETTLE = 2.0, 1.2
#: The speed ladder's periods, taken from `sweep.py` because it is the thing that
#: WROTE them, minus the fast rung: at 1 s the joint does not track the command
#: (the loop runs out of authority long before the duty pins), so it is not a
#: steady speed and is left out of the regression. It is still in the data as the
#: velocity ceiling.
SPEED_PERIODS = tuple(T for T in sweep.SPEED_PERIODS if T > 1.0)


def duty(load_raw):
    """PRESENT_LOAD -> signed duty in [-1, 1]. See fit_bam.Run for why both a
    signed and a sign-magnitude convention are in bench/data."""
    v = np.asarray(load_raw, dtype=np.float64)
    return np.where(v < 0, v, np.where(v >= 1024, -(v - 1024), v)) / 1000.0


def _newest(pattern):
    f = sorted(glob.glob(pattern))
    return f[-1] if f else None


def hold_ladder(path):
    """-> (stiffness, k_t_eff, friction floor, friction per N*m), all at one supply."""
    meta, c = runlog.read(path)
    t = np.asarray(c["t"]); tgt = np.asarray(c["target_rad"])
    q = np.asarray(c["q_rad"]); U = np.asarray(c["volt_v"])
    du = duty(c["load_raw"]) * U
    g = meta["mass_kg"] * G * meta["radius_m"]

    seg = []
    for k in range(int(t[-1] // DWELL) + 1):
        s = (t >= k * DWELL + SETTLE) & (t < (k + 1) * DWELL)
        if s.sum() < 50:
            continue
        seg.append((round(float(tgt[s][0]), 2), float((tgt[s] - q[s]).mean()),
                    float(g * np.sin(q[s].mean())), float(du[s].mean())))
    # The two passes are keyed by COMMANDED angle, not by index: each starts one
    # rung outside the ladder so its first measured angle is approached from the
    # right side, so the two halves are offset by one and zipping them is wrong.
    half = len(seg) // 2
    up = {r[0]: r for r in seg[:half]}
    dn = {r[0]: r for r in seg[half:]}
    ang = sorted(set(up) & set(dn))
    err = np.array([(up[a][1] + dn[a][1]) / 2 for a in ang])
    tau = np.array([(up[a][2] + dn[a][2]) / 2 for a in ang])
    mean = np.array([(up[a][3] + dn[a][3]) / 2 for a in ang])
    fric = np.array([abs(up[a][3] - dn[a][3]) / 2 for a in ang])

    stiff = float(np.polyfit(err, tau, 1)[0])
    k_t = 1.0 / float(np.polyfit(tau, -mean, 1)[0])
    slope, floor = np.polyfit(np.abs(tau), fric, 1)
    return stiff, k_t, floor * k_t, slope * k_t, ang, fric * k_t, tau


def speed_ladder(pattern, k_t):
    """-> (kinetic Coulomb N*m, N*m per rad/s, mean |gravity| N*m) at one supply.

    The intercept is the Coulomb friction AT THE LADDER'S MEAN LOAD, not at zero
    load, and on this servo that distinction is worth 0.06 N*m: the triangle
    sweeps +-0.5 rad about the hanging position, so the arm carries a mean
    |m*g*r*sin(q)| of about 0.23 N*m throughout, and friction grows with load.
    The caller subtracts it using the hold ladder's own slope, which is the only
    reason these two trajectories have to be read together rather than separately.
    """
    pts = []
    for T in SPEED_PERIODS:
        p = _newest(pattern % T)
        if p is None:
            continue
        meta, c = runlog.read(p)
        t = np.asarray(c["t"]); w = np.asarray(c["w_rad_s"])
        du = duty(c["load_raw"]) * np.asarray(c["volt_v"])
        u = (t % T) / T
        # Each ramp trimmed at both ends: the reversal is a transient and the
        # gravity torque only cancels between a full up ramp and a full down one.
        a, b = (u > 0.08) & (u < 0.45), (u > 0.58) & (u < 0.95)
        pts.append(((np.abs(w[a]).mean() + np.abs(w[b]).mean()) / 2,
                    (abs(du[a].mean()) + abs(du[b].mean())) / 2))
        g_bar = meta["mass_kg"] * G * meta["radius_m"] * \
            np.abs(np.sin(np.asarray(c["q_rad"]))).mean()
    if len(pts) < 2:
        return None
    p = np.array(pts)
    slope, floor = np.polyfit(p[:, 0], p[:, 1], 1)
    return floor * k_t, slope * k_t, float(g_bar)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--mass", default="1.066")
    ap.add_argument("--radius", default="0.09")
    ap.add_argument("--volts", type=float, action="append")
    a = ap.parse_args()

    tag = f"m{a.mass}_r{a.radius}"
    volts = a.volts or sorted(
        {float(os.path.basename(f).split("_v")[1].split("_")[0])
         for f in glob.glob(os.path.join(a.data, f"holdbi_{tag}_v*.csv"))})
    if not volts:
        raise SystemExit(f"no holdbi runs for {tag} in {a.data} — "
                         f"sweep.py --traj holdbi")

    print(f"{'V':>5} {'stiffness':>10} {'k_t_eff':>8} | {'floor':>7} {'per N*m':>8} "
          f"| {'kin tau_c':>10} {'per rad/s':>10}")
    print(f"{'':>5} {'N*m/rad':>10} {'N*m/V':>8} | {'N*m':>7} {'N*m/N*m':>8} "
          f"| {'N*m':>10} {'N*m s/rad':>10}")
    rows = []
    for v in volts:
        h = _newest(os.path.join(a.data, f"holdbi_{tag}_v{v:g}_*.csv"))
        if h is None:
            continue
        stiff, k_t, floor, per_tau, *_ = hold_ladder(h)
        sp = speed_ladder(os.path.join(a.data, f"speed%ds_{tag}_v{v:g}_*.csv"), k_t)
        kc, kv, g_bar = sp if sp else (float("nan"),) * 3
        kc0 = kc - per_tau * g_bar          # extrapolated to zero load
        print(f"{v:5g} {stiff:10.1f} {k_t:8.3f} | {floor:7.3f} {per_tau:8.3f} "
              f"| {kc0:10.3f} {kv:10.3f}")
        rows.append((v, stiff, k_t, floor, per_tau, kc0, kv))

    if len(rows) < 2:
        return
    r = np.array(rows)
    print(f"\n{'mean':>5} {'':>10} " + " ".join(f"{x:8.3f}" for x in r[:, 2:].mean(0)))
    print(f"{'spread':>5} {'':>10} " +
          " ".join(f"{100 * np.ptp(x) / abs(x.mean()):7.0f}%" for x in r[:, 2:].T))
    # The SLOPE of stiffness against supply, fitted — not the ratio of the two
    # column means, which is what this printed until 2026-09-11. A ratio of means
    # is the line through the ORIGIN, and this line does not go through the origin:
    # over 8/10/12 V the fit is 3.20 N*m/rad per volt with a -0.5 intercept, while
    # the ratio read 3.44. It is a description of the servo either way, but only one
    # of them is the thing the sentence claims to be printing.
    per_volt = float(np.polyfit(r[:, 0], r[:, 1], 1)[0])
    print("\nstiffness is DELIBERATELY not averaged: it scales with supply "
          f"({per_volt:.2f} N*m/rad per volt, fitted over "
          f"{', '.join(f'{v:g}' for v in r[:, 0])} V), which is the "
          "one\nnumber here that is allowed to move with voltage. Everything right "
          "of it is a\nfriction or a torque constant and must not — a spread over "
          "~15 % there is a\nmeasurement problem, not a property of the servo.")


if __name__ == "__main__":
    main()
