#!/usr/bin/env python3
"""
incident.py — read a tick log after something went wrong.

    python bench/incident.py ~/smalldog_logs/tick_20260917_183012.npz
    python bench/incident.py FILE.npz --last 3 --joints fl_pitch,fr_pitch
    python bench/incident.py --selftest

Reads what `runtime/ticklog.py` writes — `walk.py --log`, the servo node's `log:=`
— and answers the questions an incident asks in the order they get asked:

  1. how did it end: the trip, the run's length, late ticks;
  2. which joint, when: per joint the peak duty (the torque the bracket saw), the
     peak current, the peak tracking error and the peak load, each with its time;
  3. what the body did: peak pitch and roll and when, the gyro peak;
  4. the last `--last` seconds, tick by tick, for the joints that were worst in
     them (or `--joints`), beside pitch and roll — the hop before the crack.

`bench/pack_sag.py` and `bench/trot_report.py` read the same file for the pack and
the gait; this one is for the day the front hip brackets broke (2026-09-17) and
nothing had been recorded.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def load(path) -> dict:
    d = np.load(path, allow_pickle=True)
    n = len(d["t"])
    joints = [str(j) for j in d["joints"]]
    fields = [str(f) for f in d["fields"]]
    fb = d["fb"]
    r = dict(t=d["t"], dt=d["dt"], goal=d["goal"], cmd=d["cmd"], joints=joints,
             imu=d["imu"] if "imu" in d.files else np.full((n, 9), np.nan),
             duty=d["duty"] if "duty" in d.files else np.full((n, len(joints)), np.nan),
             overruns=d["overruns"] if "overruns" in d.files else np.zeros(n, int),
             meta=json.loads(str(d["meta"])) if "meta" in d.files else
             (d["gait"].item() if d["gait"].shape == () else {}))
    for i, f in enumerate(fields):
        r[f] = fb[:, :, i]
    g = r["imu"][:, 0:3]
    with np.errstate(invalid="ignore"):
        r["pitch"] = np.degrees(np.arcsin(np.clip(g[:, 0], -1, 1)))
        r["roll"] = np.degrees(np.arctan2(-g[:, 1], -g[:, 2]))
    r["err"] = np.degrees(r["goal"] - r["q"])
    return r


def _peak(t, x):
    """(value, time) of the largest |x|, nan-safe."""
    if not np.isfinite(x).any():
        return math.nan, math.nan
    i = int(np.nanargmax(np.abs(x)))
    return float(x[i]), float(t[i])


def analyse(r: dict) -> dict:
    t, J = r["t"], r["joints"]
    o = dict(ticks=len(t), seconds=float(t[-1] - t[0] + r["dt"][0]) if len(t) else 0.0,
             dt_max=float(np.nanmax(r["dt"])) if len(t) else math.nan,
             overruns=int(r["overruns"][-1]) if len(t) else 0,
             trip=str(r["meta"].get("trip", "")), joints={})
    for j, n in enumerate(J):
        o["joints"][n] = dict(duty=_peak(t, r["duty"][:, j]), current=_peak(t, r["current"][:, j]),
                              err=_peak(t, r["err"][:, j]), load=_peak(t, r["load"][:, j]),
                              temp=float(np.nanmax(r["temp"][:, j])) if np.isfinite(r["temp"][:, j]).any() else math.nan,
                              missing=int(np.isnan(r["q"][:, j]).sum()))
    o["pitch"], o["roll"] = _peak(t, r["pitch"]), _peak(t, r["roll"])
    o["gyro"] = _peak(t, np.linalg.norm(r["imu"][:, 3:6], axis=1)) if np.isfinite(r["imu"][:, 3:6]).any() else (math.nan, math.nan)
    o["volt_min"] = float(np.nanmin(r["volt"])) if np.isfinite(r["volt"]).any() else math.nan
    return o


def worst_joints(r: dict, t0: float, k=3) -> list:
    """the k joints with the largest |duty| (or |err| without duty) after t0"""
    m = r["t"] >= t0
    score = r["duty"][m] if np.isfinite(r["duty"][m]).any() else r["err"][m]
    peak = np.nanmax(np.abs(np.nan_to_num(score)), axis=0)
    order = np.argsort(-peak)
    return [r["joints"][i] for i in order[:k]]


def report(r: dict, o: dict, last=2.0, joints=None, lines=40) -> str:
    s = [f"{o['ticks']} ticks, {o['seconds']:.1f} s, dt max {1e3 * o['dt_max']:.0f} ms, "
         f"{o['overruns']} late; pack min {o['volt_min']:.1f} V"]
    m = r["meta"]
    if m:
        keys = [k for k in ("node", "policy", "mode2", "kp", "kd", "kff", "duty_cap", "tilt_deg", "period", "speed") if k in m]
        s.append("run: " + ", ".join(f"{k} {m[k]}" for k in keys))
    s.append(f"END: {o['trip'] or 'no trip (a clean exit, or a kill)'}")
    s.append("\n  joint       duty @ s    current @ s     err deg @ s     load @ s   temp  miss")
    for n, v in o["joints"].items():
        s.append(f"  {n:9s} {v['duty'][0]:+6.0f} {v['duty'][1]:6.1f}  {v['current'][0]:5.2f} {v['current'][1]:6.1f}"
                 f"  {v['err'][0]:+7.1f} {v['err'][1]:6.1f}  {v['load'][0]:+6.0f} {v['load'][1]:6.1f}"
                 f"  {v['temp']:4.0f}  {v['missing']:4d}")
    s.append(f"\n  body: pitch peak {o['pitch'][0]:+.0f} deg @ {o['pitch'][1]:.1f} s, "
             f"roll peak {o['roll'][0]:+.0f} deg @ {o['roll'][1]:.1f} s, "
             f"gyro peak {o['gyro'][0]:.2f} rad/s @ {o['gyro'][1]:.1f} s")
    if last > 0 and len(r["t"]):
        t0 = r["t"][-1] - last
        js = joints or worst_joints(r, t0)
        idx = [r["joints"].index(j) for j in js]
        rows = np.flatnonzero(r["t"] >= t0)
        step = max(1, len(rows) // lines)
        s.append(f"\n  last {last:g} s, {', '.join(js)} (q/goal deg, err deg, duty, A) and the body:")
        s.append("      t   " + " ".join(f"{j:>26s}" for j in js) + "   pitch  roll  late")
        for i in rows[::step]:
            cells = " ".join(f"{math.degrees(r['q'][i, k]):+5.0f}/{math.degrees(r['goal'][i, k]):+5.0f} "
                             f"{r['err'][i, k]:+5.0f} {r['duty'][i, k]:+5.0f} {r['current'][i, k]:4.2f}"
                             for k in idx)
            s.append(f"  {r['t'][i]:6.2f} {cells}   {r['pitch'][i]:+5.0f} {r['roll'][i]:+5.0f}  "
                     f"{r['overruns'][i]:4d}")
    return "\n".join(s)


# ------------------------------------------------------------------ selftest
def _synth(path, hz=50, seconds=12.0, hop_at=8.0):
    """a quiet stand, then fl_pitch driven to full duty against a 30 deg error with
    the body pitching to 35 deg nose-down — the hop — and a trip 0.4 s later"""
    n = int(seconds * hz)
    t = np.arange(1, n + 1) / hz
    J = ["fl_roll", "fl_pitch", "fl_knee", "fr_roll", "fr_pitch", "fr_knee",
         "rl_roll", "rl_pitch", "rl_knee", "rr_roll", "rr_pitch", "rr_knee"]
    goal = np.tile([0.0, -0.8, 1.5] * 4, (n, 1)).astype(np.float32)
    q = goal + np.random.default_rng(0).normal(0, 0.01, goal.shape).astype(np.float32)
    duty = np.zeros_like(goal) + 150
    cur = np.full_like(goal, 0.2)
    imu = np.tile([0.0, 0.0, -1.0, 0, 0, 0, 0, 0, 9.81], (n, 1)).astype(np.float32)
    hop = (t >= hop_at) & (t < hop_at + 0.4)
    q[hop, 1] = goal[hop, 1] - math.radians(30)
    duty[hop, 1] = 1000
    cur[hop, 1] = 1.9
    imu[hop, 0] = math.sin(math.radians(35))
    imu[hop, 2] = -math.cos(math.radians(35))
    fb = np.stack([q, np.zeros_like(q), duty * 0.7, np.full_like(q, 11.8), np.full_like(q, 34.0), cur], axis=2)
    np.savez(path, t=t, dt=np.full(n, 1 / hz), fb=fb.astype(np.float32), goal=goal,
             phase=np.full((n, 4), np.nan, np.float32), cmd=np.zeros((n, 3), np.float32),
             imu=imu, duty=duty.astype(np.float32), overruns=np.cumsum(hop).astype(int),
             fields=np.array(["q", "w", "load", "volt", "temp", "current"]), joints=np.array(J),
             legs=np.array(["fl", "fr", "rl", "rr"]), gait=np.array({}),
             meta=np.array(json.dumps(dict(node="test", trip="tilted for 0.30 s — the robot is over"))))
    return t[hop][0]


def selftest() -> int:
    import tempfile
    ok = True

    def chk(name, cond, extra=""):
        nonlocal ok
        ok &= bool(cond)
        print(f"  {'ok  ' if cond else 'FAIL'} {name}{extra}")

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "hop.npz")
        t_hop = _synth(path)
        r = load(path)
        o = analyse(r)
        chk("the trip is read back", "tilted" in o["trip"])
        fp = o["joints"]["fl_pitch"]
        chk("the joint at full duty is found, at the hop", fp["duty"][0] == 1000 and abs(fp["duty"][1] - t_hop) < 0.05,
            f" (duty {fp['duty'][0]:.0f} @ {fp['duty'][1]:.2f}, hop at {t_hop:.2f})")
        chk("... its tracking error too", abs(abs(fp["err"][0]) - 30) < 1.5, f" ({fp['err'][0]:+.1f} deg)")
        chk("a quiet joint reads its quiet peak", o["joints"]["rr_knee"]["duty"][0] == 150)
        chk("the body's pitch peak is at the hop", abs(o["pitch"][0] - 35) < 0.5 and abs(o["pitch"][1] - t_hop) < 0.5,
            f" ({o['pitch'][0]:+.0f} deg @ {o['pitch'][1]:.2f})")
        chk("late ticks are counted", o["overruns"] == 20)
        chk("the worst joint in the last seconds is fl_pitch", worst_joints(r, r["t"][-1] - 5)[0] == "fl_pitch")
        txt = report(r, o, last=5.0)
        chk("the report names it in the tail", "fl_pitch" in txt.split("last 5 s")[1] and "+1000" in txt)
        chk("... and the end", "END: tilted" in txt)
        print(txt)
    print("incident:", "ok" if ok else "FAILED")
    return 0 if ok else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", nargs="?")
    ap.add_argument("--last", type=float, default=2.0, help="s of tail to print tick by tick (0: none)")
    ap.add_argument("--joints", default="", help="comma-separated; default: the worst three in the tail")
    ap.add_argument("--lines", type=int, default=40, help="at most this many tail rows")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.log:
        ap.error("a log file, or --selftest")
    r = load(a.log)
    js = [j.strip() for j in a.joints.split(",") if j.strip()] or None
    if js:
        bad = [j for j in js if j not in r["joints"]]
        if bad:
            ap.error(f"no joint {bad}; the log has {r['joints']}")
    print(report(r, analyse(r), last=a.last, joints=js, lines=a.lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
