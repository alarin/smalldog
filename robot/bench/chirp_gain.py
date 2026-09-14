#!/usr/bin/env python
"""
chirp_gain.py — the servo's frequency response, measured against the two models.

    python bench/chirp_gain.py                      # every bench/data/chirp_*.csv
    python bench/chirp_gain.py bench/data/chirp_m1.066_r0.09_v12_*.csv

Why this exists: the first RL policy on the real robot (2026-09-14) "danced in place".
In sim every checkpoint trots at a 0.20 s period — 5 Hz — with pitch joint speeds of
3 rad/s; the real servos delivered 0.6 rad/s and 7° of mean tracking error, and the
closed loop turned into a 0.74 s rocking. The question is therefore: at 5 Hz, how much
of a commanded amplitude does an ST3215 deliver, and how much does the model the policy
trained against deliver?

`sweep.py --traj chirp` swept 0.2 → 8 Hz over 30 s at ±0.26 rad (15°) with the bench arm
hanging, so the instantaneous frequency is known at every sample, and the gain and phase
at any frequency come from the window around it. Three columns:

  measured    q / target from the csv
  simulate()  rl/actuator.py's bench model, the one fit_bam.py fitted — 1 kHz inner loop
  train joint the TRAINING loop: profile_goal() + bus_torque() at 50 Hz, zero-order hold, on a rigid
              inertia J_m + J_load with the Coulomb floor and the pendulum's gravity —
              what rl/env/walk.py's physics scan does per joint, without MuJoCo. Once
              with the bus delay the env randomises (0..1 tick) and once without.

Gain is the RMS ratio inside a ±10 % frequency band; phase is the cross-correlation lag.
A gain the models hold at 5 Hz and the servo does not is the answer.
"""
from __future__ import annotations

import glob
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "rl"))
import actuator as A                                                 # noqa: E402
from bench import runlog                                             # noqa: E402

FREQS = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0)
F0, F1, T_CHIRP = 0.2, 8.0, 30.0          # sweep.py's traj_chirp defaults


def band_gain(t, f_inst, x, y, fc, width=0.10):
    """RMS gain and phase (deg, negative = lag) of y against x where f_inst ≈ fc."""
    m = (f_inst > fc * (1 - width)) & (f_inst < fc * (1 + width))
    if m.sum() < 8:
        return float("nan"), float("nan")
    xs, ys = x[m] - x[m].mean(), y[m] - y[m].mean()
    g = np.sqrt((ys ** 2).mean() / max((xs ** 2).mean(), 1e-12))
    # lag by cross-correlation, bounded to half a period
    dt = float(np.median(np.diff(t[m])))
    nmax = max(1, min(int(0.5 / fc / dt), len(xs) // 2))
    lags = range(-nmax, nmax + 1)
    c = [np.dot(xs[max(0, -k):len(xs) - max(0, k)], ys[max(0, k):len(ys) - max(0, -k)])
         for k in lags]
    k = lags[int(np.argmax(c))]
    return float(g), float(-360.0 * k * dt * fc)


def train_joint(p, target, dt, J_load, load, q0, u_bat, delay_ticks=0, h=1e-4):
    """One joint as the training scan sees it: rigid J_m + J_load, bus_torque held for
    a control tick, Coulomb floor as a stick (the crude Karnopp MuJoCo's frictionloss
    approximates), gravity from the arm."""
    n = len(target)
    q, w = float(q0), 0.0
    goal, goal_w = float(q0), 0.0
    J = p.J_m + J_load
    sub = int(round(dt / h))
    buf = [float(q0)] * (delay_ticks + 1)
    out = np.empty(n)
    for k in range(n):
        buf.append(float(target[k]))
        tgt = buf.pop(0)
        for _ in range(sub):
            goal, goal_w = A.profile_goal(p, goal, goal_w, tgt, h, xp=np)
            tau = float(A.bus_torque(p, goal - q, w, u_bat, 0.0, xp=np, tau_c_external=True))
            tau += load(q)
            if abs(w) < p.v_eps and abs(tau) <= p.tau_c:
                w = 0.0                                   # stuck
            else:
                tau -= p.tau_c * (np.sign(w) if w != 0 else np.sign(tau))
                w += tau / J * h
            q += w * h
        out[k] = q
    return out


def analyse(path, p):
    meta, cols = runlog.read(path)
    t = np.asarray(cols["t"]); tgt = np.asarray(cols["target_rad"]); q = np.asarray(cols["q_rad"])
    ok = ~np.isnan(tgt)
    t, tgt, q = t[ok], tgt[ok], q[ok]
    t = t - t[0]
    dt = 1.0 / 200.0
    grid = np.arange(0.0, min(t[-1], T_CHIRP), dt)
    tgt_g, q_g = np.interp(grid, t, tgt), np.interp(grid, t, q)
    volt = float(np.nanmedian(np.asarray(cols["volt_v"])))
    f_inst = F0 + (F1 - F0) / T_CHIRP * grid

    mass, radius = float(meta["mass_kg"]), float(meta["radius_m"])
    J_load = float(meta.get("arm_inertia", 0.0) or 0.0) + mass * radius ** 2
    load = A.pendulum_load(mass, radius)
    pj = A.Params(**{**p.__dict__, "J_l": max(J_load, 1e-6)})
    sim = A.simulate(pj, tgt_g, dt, q0=q_g[0], u_bat=volt, load_torque=load, dt_int=1e-4)
    q_sim = np.asarray(sim["q"] if isinstance(sim, dict) else sim[0])

    # the training loop runs at 50 Hz: resample the target with a zero-order hold
    dt50 = 0.02
    g50 = np.arange(0.0, grid[-1], dt50)
    tgt50 = np.interp(g50, grid, tgt_g)
    q_tr0 = np.interp(grid, g50, train_joint(p, tgt50, dt50, J_load, load, q_g[0], volt, 0))
    q_tr1 = np.interp(grid, g50, train_joint(p, tgt50, dt50, J_load, load, q_g[0], volt, 1))

    name = os.path.basename(path)
    print(f"\n{name}: {mass:g} kg at {radius:g} m, {volt:.1f} V, J_load {J_load:.4f}")
    print(f"  {'Hz':>4}  {'measured':>14}  {'simulate()':>14}  {'train 0 tick':>14}  {'train 1 tick':>14}")
    rows = []
    for fc in FREQS:
        gm, pm = band_gain(grid, f_inst, tgt_g, q_g, fc)
        gs, ps = band_gain(grid, f_inst, tgt_g, q_sim, fc)
        g0, p0 = band_gain(grid, f_inst, tgt_g, q_tr0, fc)
        g1, p1 = band_gain(grid, f_inst, tgt_g, q_tr1, fc)
        rows.append((fc, gm, gs, g0, g1))
        print(f"  {fc:4.1f}  {gm:5.2f} {pm:+6.0f}°   {gs:5.2f} {ps:+6.0f}°   "
              f"{g0:5.2f} {p0:+6.0f}°   {g1:5.2f} {p1:+6.0f}°")
    return rows


def main():
    files = sys.argv[1:] or sorted(glob.glob(os.path.join(HERE, "data", "chirp_*.csv")))
    p = A.load(quiet=True)
    print(f"actuator: {os.path.relpath(A.DEFAULT_PATH)}  kp {p.kp:g}  J_m {p.J_m:g}  "
          f"tau_c {p.tau_c:g}  loop {p.loop_hz:g} Hz")
    allrows = [analyse(f, p) for f in files]
    if len(allrows) > 1:
        print("\nmean over runs (gain only):")
        print(f"  {'Hz':>4}  {'measured':>8}  {'simulate()':>10}  {'train 0':>8}  {'train 1':>8}")
        for i, fc in enumerate(FREQS):
            g = np.nanmean([[r[i][j] for j in (1, 2, 3, 4)] for r in allrows], axis=0)
            print(f"  {fc:4.1f}  {g[0]:8.2f}  {g[1]:10.2f}  {g[2]:8.2f}  {g[3]:8.2f}")


if __name__ == "__main__":
    main()
