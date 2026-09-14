#!/usr/bin/env python
"""
trot_report.py — did the trot walk the way the gait thinks it did? From a `walk.py --log`.

    python runtime/walk.py --port /dev/ttyACM0 --go 20 --imu --log bench/data/go20.npz
    python bench/trot_report.py bench/data/go20.npz
    python bench/trot_report.py --selftest

`pack_sag.py` reads the same file for the electrical side. This one reads the motion:

  heading     yaw integrated from the gyro over the moving stretch, degrees and deg/m of
              commanded distance — whether it held a line. Needs `--imu`.
  attitude    body pitch and roll from the projected gravity, p50 / p95, and the gyro
              rates' std — how much the body rocks.
  lag         per joint: the delay that best aligns the servo to its goal (cross-
              correlation, ticks), and the amplitude it reaches against what was asked.
              A servo that gets there late is a different defect from one that never
              gets there, and the runtime's tracking error mixes them.
  footfalls   the vertical accelerometer's impulses binned by the fl leg's gait phase,
              against where the gait puts touchdown (phase 0.0 for fl/rr, 0.5 for fr/rl).
              The offset between the two is the same lag, seen from the body.
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from bench.pack_sag import load  # noqa: E402

NBIN = 20
MAX_LAG = 40


def _moving(r, settle_s=3.0):
    mv = np.any(np.nan_to_num(r["cmd"]) != 0.0, axis=1)
    if not mv.any():
        return mv
    first = int(np.argmax(mv))
    mv = mv.copy()
    mv[:first + int(settle_s / max(float(np.median(r["dt"])), 1e-3))] = False
    return mv


def lag_ticks(goal, q, max_lag=MAX_LAG):
    """Delay (ticks) that maximises the correlation of q(t) with goal(t - lag)."""
    g = goal - goal.mean()
    x = q - q.mean()
    if g.std() < 1e-6 or x.std() < 1e-6:
        return 0, 1.0
    best, bc = 0, -2.0
    for k in range(0, min(max_lag, len(g) // 2)):
        c = float(np.corrcoef(g[:len(g) - k] if k else g, x[k:])[0, 1])
        if c > bc:
            best, bc = k, c
    return best, bc


def analyse(r: dict) -> dict:
    t, dt, mv = r["t"], r["dt"], _moving(r)
    out = dict(ticks=len(t), moving_s=float(dt[mv].sum()), has_imu=bool(np.isfinite(r["imu"]).any()))
    speed = float(np.nanmedian(np.abs(r["cmd"][mv, 0]))) if mv.any() else 0.0
    out["distance_cmd"] = float(np.nansum(np.abs(r["cmd"][mv, 0]) * dt[mv]))
    out["speed_cmd"] = speed

    q, goal = r["q"], r["goal"]
    out["lag"] = {}
    for i, n in enumerate(r["joints"]):
        gi, qi = goal[mv, i], np.nan_to_num(q[mv, i], nan=float(np.nanmean(q[mv, i])) if mv.any() else 0.0)
        if not mv.any():
            break
        k, c = lag_ticks(gi, qi)
        span_g = float(np.ptp(gi))
        span_q = float(np.ptp(qi))
        err = np.degrees(np.abs(gi - qi))
        out["lag"][n] = dict(ticks=k, ms=1e3 * k * float(np.median(dt)), corr=c,
                             amp_goal=math.degrees(span_g), amp_q=math.degrees(span_q),
                             err_p50=float(np.percentile(err, 50)), err_p95=float(np.percentile(err, 95)))

    if out["has_imu"] and mv.any():
        imu = r["imu"]
        g, w, acc = imu[:, 0:3], imu[:, 3:6], imu[:, 6:9]
        wz = np.nan_to_num(w[:, 2])
        yaw = np.cumsum(wz * dt)
        yaw_mv = yaw[_moving(r, settle_s=0.0)]     # the whole walk, first step included
        out["yaw_deg"] = math.degrees(float(yaw_mv[-1] - yaw_mv[0]))
        out["yaw_per_m"] = out["yaw_deg"] / out["distance_cmd"] if out["distance_cmd"] > 0.05 else math.nan
        # a still stretch (cmd 0) before the walk: the residual bias after measure_bias
        still = ~np.any(np.nan_to_num(r["cmd"]) != 0.0, axis=1)
        still[int(np.argmax(~still)):] = False
        out["yaw_bias_dps"] = math.degrees(float(wz[still].mean())) if still.sum() > 10 else math.nan
        gx, gy, gz = (np.nan_to_num(g[mv, i]) for i in range(3))
        pitch = np.degrees(np.arctan2(gx, -gz))       # nose down positive when gravity leans +x
        roll = np.degrees(np.arctan2(gy, -gz))
        out["pitch_p50"], out["pitch_p95"] = float(np.median(pitch)), float(np.percentile(np.abs(pitch), 95))
        out["roll_p50"], out["roll_p95"] = float(np.median(roll)), float(np.percentile(np.abs(roll), 95))
        out["gyro_std"] = tuple(float(np.nanstd(w[mv, i])) for i in range(3))
        # footfalls: |a_z - mean| binned by fl phase
        az = np.nan_to_num(acc[mv, 2])
        az = np.abs(az - az.mean())
        ph = r["phase"][mv, r["legs"].index("fl")]
        b = np.minimum((ph * NBIN).astype(int), NBIN - 1)
        prof = np.array([az[b == i].mean() if (b == i).any() else np.nan for i in range(NBIN)])
        out["footfall_profile"] = prof
        out["footfall_peak_phase"] = float(np.nanargmax(prof) / NBIN)
        out["accel_z_std"] = float(np.nan_to_num(acc[mv, 2]).std())
    return out


def report(o: dict) -> str:
    s = [f"{o['ticks']} ticks, {o['moving_s']:.1f} s moving at {o['speed_cmd']:.2f} m/s: "
         f"{o['distance_cmd']:.2f} m commanded"]
    if o["has_imu"]:
        s.append(f"heading    {o['yaw_deg']:+.1f} deg over the walk"
                 + (f", {o['yaw_per_m']:+.1f} deg/m" if math.isfinite(o["yaw_per_m"]) else "")
                 + (f"; residual gyro bias {o['yaw_bias_dps']:+.2f} deg/s standing"
                    if math.isfinite(o["yaw_bias_dps"]) else ""))
        s.append(f"attitude   pitch {o['pitch_p50']:+.1f} deg (|p95| {o['pitch_p95']:.1f}), "
                 f"roll {o['roll_p50']:+.1f} (|p95| {o['roll_p95']:.1f}); gyro std "
                 + " / ".join(f"{v:.2f}" for v in o["gyro_std"]) + " rad/s")
        s.append(f"footfalls  a_z impulse peaks at fl phase {o['footfall_peak_phase']:.2f} "
                 f"(the gait lands fl/rr at 0.00, fr/rl at 0.50); a_z std {o['accel_z_std']:.2f} m/s^2")
        prof = o["footfall_profile"]
        s.append("           " + " ".join(f"{v:4.1f}" for v in prof))
        s.append("           phase 0" + " " * (5 * NBIN // 2 - 8) + "0.5" + " " * (5 * NBIN // 2 - 6) + "1")
    else:
        s.append("heading    no IMU in this log (walk.py --imu)")
    s.append("lag        joint       delay     corr   amp asked/got   err p50/p95 deg")
    for n, l in o["lag"].items():
        s.append(f"           {n:<9} {l['ticks']:3d} tk {l['ms']:4.0f} ms  {l['corr']:5.2f}   "
                 f"{l['amp_goal']:5.1f} / {l['amp_q']:5.1f}    {l['err_p50']:5.1f} / {l['err_p95']:5.1f}")
    return "\n".join(s)


# ------------------------------------------------------------------ self-test
def _synth(lag_ticks=14, yaw_dps=2.0, hz=50, seconds=25.0, seed=0):
    rng = np.random.default_rng(seed)
    n = int(seconds * hz)
    t = np.arange(n) / hz
    dt = np.full(n, 1.0 / hz)
    joints = [f"{l}_{j}" for l in ("fl", "fr", "rl", "rr") for j in ("roll", "pitch", "knee")]
    legs = ["fl", "fr", "rl", "rr"]
    period = 1.35
    moving = t >= 2.0
    phase = np.stack([(((t - 2.0) / period) * moving + o) % 1.0 for o in (0.0, 0.5, 0.5, 0.0)], 1)
    cmd = np.where(moving[:, None], np.array([0.11, 0, 0]), 0.0).astype(np.float32)
    goal = np.zeros((n, 12))
    for li, l in enumerate(legs):
        sw = np.clip(np.sin(2 * np.pi * (phase[:, li] - 0.5)), 0, None) * moving
        goal[:, 3 * li + 1] = -0.65 + 0.2 * sw
        goal[:, 3 * li + 2] = 1.24 + 0.5 * sw
    q = np.roll(goal, lag_ticks, axis=0) + 0.005 * rng.standard_normal((n, 12))
    q[:lag_ticks] = goal[:lag_ticks]
    imu = np.zeros((n, 9))
    imu[:, 2] = -1.0
    imu[:, 5] = math.radians(yaw_dps) * moving
    imu[:, 8] = 9.81 + 3.0 * np.clip(np.cos(2 * np.pi * (phase[:, 0] - lag_ticks / hz / period)), 0.8, None) * moving
    fb = np.zeros((n, 12, 6))
    fb[:, :, 0] = q
    return dict(t=t, dt=dt, goal=goal, phase=phase, cmd=cmd, imu=imu, joints=joints, legs=legs,
                gait={}, q=q, w=fb[:, :, 1], load=fb[:, :, 2], volt=fb[:, :, 3], temp=fb[:, :, 4],
                current=fb[:, :, 5])


def selftest() -> int:
    fails = 0

    def chk(name, cond, extra=""):
        nonlocal fails
        print(f"  {'ok ' if cond else 'FAIL'} {name} {extra}")
        fails += not cond

    r = _synth(lag_ticks=14, yaw_dps=2.0)
    o = analyse(r)
    print(report(o))
    k = o["lag"]["fl_knee"]["ticks"]
    chk("knee lag recovered", k == 14, f"{k} ticks")
    chk("roll (no motion) reports no lag", o["lag"]["fl_roll"]["ticks"] == 0)
    chk("yaw integrates to 2 deg/s * 23 s", abs(o["yaw_deg"] - 46.0) < 1.5, f"{o['yaw_deg']:.1f}")
    chk("footfall peak lands the lag after touchdown",
        abs(o["footfall_peak_phase"] - 14 / 50 / 1.35) < 0.1, f"{o['footfall_peak_phase']:.2f}")
    chk("amplitude is reached", abs(o["lag"]["fl_knee"]["amp_q"] - o["lag"]["fl_knee"]["amp_goal"]) < 2)
    print("selftest:", "ok" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.files:
        ap.error("give a recording, or --selftest")
    for p in a.files:
        r = load(p)
        print(f"== {p}")
        print(report(analyse(r)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
