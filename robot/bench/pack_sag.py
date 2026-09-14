#!/usr/bin/env python
"""
pack_sag.py — what the pack does under a trot, from a `walk.py --log` recording.

    python runtime/walk.py --profile --seconds 60 --log bench/data/trot_pack_*.npz
    python bench/pack_sag.py bench/data/trot_pack_*.npz
    python bench/pack_sag.py --selftest

Why this exists: every current and voltage number in `POWER.md` was measured at 1.55 kg
on a bench supply, and the file says outright that the number it most wants is the
whole-robot draw and the pack-terminal sag during a trot at the design mass. That only
exists untethered, on the pack, with room to walk. The twelve servos are the meter:
each one reports its own supply volt and current in the feedback the loop reads
anyway, so the bus minimum per tick IS the terminal voltage less the wiring drop, and
the sum per tick is the bus draw.

What comes out, and what each line settles:

  standing          V at rest and the standing draw (the profile's first 2 s are cmd 0)
  sag               V_rest - min over the trot: the transient the buck has to ride
  R_eff             least-squares slope of bus-min V against summed current, ohms —
                    pack internal resistance plus the P+ run to the furthest servo
  discharge         V/min over the trot, from the open voltage (median + R_eff * draw)
  current           per-servo peak, summed p50/p95/max: `POWER.md`'s "0.86 A peaks —
                    one servo or the sum?" is answered by printing both
  by phase          sum current and bus-min V per tenth of the fl leg's cycle: where in
                    the stride the pull is (stance below 0.5, swing above)
  trips             ticks under the Limits the runtime would hold on (volt_min 9.9/9.5)
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from runtime.safety import Limits  # noqa: E402

NBIN = 10


def load(path):
    d = np.load(path, allow_pickle=True)
    fields = list(d["fields"])
    fb = d["fb"]
    col = {f: fb[:, :, fields.index(f)] for f in fields}
    n = len(d["t"])
    return dict(t=d["t"], dt=d["dt"], goal=d["goal"], phase=d["phase"], cmd=d["cmd"],
                imu=d["imu"] if "imu" in d.files else np.full((n, 9), np.nan),
                joints=list(d["joints"]), legs=list(d["legs"]),
                gait=d["gait"].item() if d["gait"].shape == () else {}, **col)


def analyse(r: dict, limits: Limits = Limits()) -> dict:
    volt, cur, t = r["volt"], r["current"], r["t"]
    vmin = np.nanmin(volt, axis=1)
    vmed = np.nanmedian(volt, axis=1)
    isum = np.nansum(cur, axis=1)
    out = dict(ticks=len(t), seconds=float(t[-1] - t[0]) if len(t) > 1 else 0.0)

    still = np.all(np.nan_to_num(r["cmd"]) == 0.0, axis=1)
    moving = ~still
    rest = still if still.sum() >= 10 else np.ones_like(still)
    out["v_rest"] = float(np.nanmedian(vmed[rest]))
    out["i_rest"] = float(np.nanmedian(isum[rest]))
    out["moving_s"] = float(np.nansum(r["dt"][moving]))

    out["v_min"] = float(np.nanmin(vmin))
    out["v_p1"] = float(np.nanpercentile(vmin, 1))
    out["sag"] = out["v_rest"] - out["v_min"]
    out["sag_p1"] = out["v_rest"] - out["v_p1"]
    k = int(np.nanargmin(vmin))
    out["v_min_at"] = (float(t[k]), r["joints"][int(np.nanargmin(volt[k]))])

    # R_eff: bus-min V against summed A over the whole run. The intercept is the open
    # terminal voltage; a slope the fit cannot see (constant current) reads as 0 with
    # a warning rather than a number.
    ok = np.isfinite(vmin) & np.isfinite(isum)
    if ok.sum() > 10 and np.nanstd(isum[ok]) > 0.02:
        A = np.stack([isum[ok], np.ones(ok.sum())], 1)
        (slope, v0), res, *_ = np.linalg.lstsq(A, vmin[ok], rcond=None)
        out["r_eff"] = float(-slope)
        out["v_open"] = float(v0)
        pred = A @ np.array([slope, v0])
        ss = float(np.sum((vmin[ok] - vmin[ok].mean()) ** 2))
        out["r_eff_r2"] = 1.0 - float(np.sum((vmin[ok] - pred) ** 2)) / ss if ss > 0 else 0.0
    else:
        out["r_eff"] = math.nan
        out["v_open"] = out["v_rest"]
        out["r_eff_r2"] = 0.0

    # Discharge: the slope of the OPEN voltage (bus median + R_eff * draw) over the
    # moving ticks. The raw median has the stride's ripple on it and a step where the
    # trot starts, and both fit as a slope that is not the pack running down.
    if moving.sum() > 5 * 50 and (t[moving][-1] - t[moving][0]) > 5:
        tm = t[moving]
        r_fit = out["r_eff"] if math.isfinite(out["r_eff"]) else 0.0
        vopen = np.nan_to_num(vmed[moving], nan=out["v_rest"]) + r_fit * isum[moving]
        A = np.stack([tm, np.ones_like(tm)], 1)
        (dv, _), *_ = np.linalg.lstsq(A, vopen, rcond=None)
        out["discharge_v_per_min"] = float(dv * 60.0)
    else:
        out["discharge_v_per_min"] = math.nan

    out["i_servo_max"] = {n: float(np.nanmax(cur[:, i])) for i, n in enumerate(r["joints"])}
    out["i_sum_p50"] = float(np.nanpercentile(isum, 50))
    out["i_sum_p95"] = float(np.nanpercentile(isum, 95))
    out["i_sum_max"] = float(np.nanmax(isum))
    out["i_sum_max_at"] = float(t[int(np.nanargmax(isum))])
    out["i_over_limit_s"] = float(np.nansum(r["dt"][np.nanmax(cur, axis=1) >= limits.current_a]))

    # by phase of the fl leg, moving ticks only
    ph = r["phase"][:, r["legs"].index("fl")]
    b = np.minimum((ph * NBIN).astype(int), NBIN - 1)
    out["by_phase"] = []
    for i in range(NBIN):
        m = moving & (b == i)
        out["by_phase"].append((i / NBIN, int(m.sum()),
                                float(np.nanmean(isum[m])) if m.any() else math.nan,
                                float(np.nanmean(vmin[m])) if m.any() else math.nan))

    out["under_volt_min_s"] = float(np.nansum(r["dt"][vmin < limits.volt_min]))
    out["under_9p9_s"] = float(np.nansum(r["dt"][vmin < 9.9]))
    out["blind"] = int(np.isnan(volt).sum())
    return out


def report(o: dict, gait: dict | None = None) -> str:
    s = []
    if gait:
        s.append(f"trot: {gait.get('speed', float('nan')):.2f} m/s, period "
                 f"{gait.get('period', float('nan')):.2f} s, {gait.get('hz', 50):.0f} Hz")
    s.append(f"{o['ticks']} ticks, {o['seconds']:.1f} s, {o['moving_s']:.1f} s moving, "
             f"{o['blind']} blind joint reads")
    s.append(f"standing   {o['v_rest']:.2f} V, {o['i_rest']:.2f} A on the bus")
    s.append(f"sag        {o['sag']:.2f} V to the floor ({o['v_min']:.2f} V at "
             f"t={o['v_min_at'][0]:.1f} s on {o['v_min_at'][1]}); {o['sag_p1']:.2f} V at p1")
    if math.isfinite(o["r_eff"]):
        s.append(f"R_eff      {1e3 * o['r_eff']:.0f} mohm (bus-min V vs summed A, r2 "
                 f"{o['r_eff_r2']:.2f}); open {o['v_open']:.2f} V")
    else:
        s.append("R_eff      not fittable: the current never varied")
    if math.isfinite(o["discharge_v_per_min"]):
        s.append(f"discharge  {1e3 * o['discharge_v_per_min']:+.0f} mV/min over the run")
    top = sorted(o["i_servo_max"].items(), key=lambda kv: -kv[1])[:3]
    s.append(f"current    summed p50 {o['i_sum_p50']:.2f} / p95 {o['i_sum_p95']:.2f} / max "
             f"{o['i_sum_max']:.2f} A (at t={o['i_sum_max_at']:.1f} s); one-servo peaks "
             + ", ".join(f"{n} {v:.2f}" for n, v in top)
             + (f"; {o['i_over_limit_s']:.2f} s with a servo at the runtime's limit"
                if o["i_over_limit_s"] > 0 else ""))
    s.append("by phase   (fl leg; stance < 0.5 < swing)   sum A   bus-min V")
    for ph, n, i, v in o["by_phase"]:
        s.append(f"           {ph:.1f}  {n:5d} ticks   {i:6.2f}   {v:7.2f}")
    if o["under_volt_min_s"] > 0 or o["under_9p9_s"] > 0:
        s.append(f"!! {o['under_9p9_s']:.2f} s under 9.9 V, {o['under_volt_min_s']:.2f} s under "
                 f"volt_min — the runtime holds {Limits.volt_hold_s:.2f} s before it trips")
    else:
        s.append("no tick under 9.9 V")
    return "\n".join(s)


# ------------------------------------------------------------------ self-test
def _synth(r_ohm=0.12, v_open=11.8, hz=50, seconds=20.0, seed=0):
    """A pack with a known resistance under a made-up trot: 2 s standing, then a stride
    whose current peaks at the fl leg's mid-stance. The fit must find r_ohm."""
    rng = np.random.default_rng(seed)
    n = int(seconds * hz)
    t = np.arange(n) / hz
    dt = np.full(n, 1.0 / hz)
    joints = [f"{l}_{j}" for l in ("fl", "fr", "rl", "rr") for j in ("roll", "pitch", "knee")]
    legs = ["fl", "fr", "rl", "rr"]
    period = 0.5
    phase = np.stack([((t / period) + o) % 1.0 for o in (0.0, 0.5, 0.5, 0.0)], 1)
    moving = t >= 2.0
    cmd = np.where(moving[:, None], np.array([0.2, 0, 0]), 0.0).astype(np.float32)
    per = 0.05 + 0.6 * moving[:, None] * (0.5 + 0.5 * np.cos(2 * np.pi * (phase[:, [0]] - 0.25)))
    cur = np.repeat(per, 12, axis=1) * (1 + 0.1 * rng.standard_normal((n, 12)))
    isum = cur.sum(1)
    volt = v_open - r_ohm * isum[:, None] - 0.05 * rng.random((n, 12))
    fb_fields = ("q", "w", "load", "volt", "temp", "current")
    return dict(t=t, dt=dt, goal=np.zeros((n, 12)), phase=phase, cmd=cmd, joints=joints,
                legs=legs, gait=dict(period=period, speed=0.2, hz=hz), q=np.zeros((n, 12)),
                w=np.zeros((n, 12)), load=np.zeros((n, 12)), volt=volt, temp=np.full((n, 12), 30.0),
                current=cur, fields=fb_fields)


def selftest() -> int:
    fails = 0

    def chk(name, cond, extra=""):
        nonlocal fails
        print(f"  {'ok ' if cond else 'FAIL'} {name} {extra}")
        fails += not cond

    r = _synth(r_ohm=0.12)
    o = analyse(r)
    print(report(o, r["gait"]))
    chk("R_eff recovered", abs(o["r_eff"] - 0.12) < 0.01, f"{o['r_eff']:.3f}")
    chk("standing is the cmd-0 stretch", abs(o["i_rest"] - 0.6) < 0.1, f"{o['i_rest']:.2f} A")
    chk("sag is positive and R*Imax", abs(o["sag"] - 0.12 * o["i_sum_max"]) < 0.1,
        f"{o['sag']:.2f} V")
    peak = max(o["by_phase"][2:4], key=lambda x: x[2])
    chk("peak draw lands at fl mid-stance", 0.2 <= peak[0] <= 0.3, f"bin {peak[0]:.1f}")
    chk("no trip on a healthy pack", o["under_9p9_s"] == 0.0)
    chk("discharge slope is flat on a synthetic pack", abs(o["discharge_v_per_min"]) < 0.02,
        f"{1e3 * o['discharge_v_per_min']:+.0f} mV/min")
    r2 = _synth(r_ohm=0.5, v_open=11.0)
    o2 = analyse(r2)
    chk("a soft pack is reported under 9.9 V", o2["under_9p9_s"] > 0, f"{o2['under_9p9_s']:.2f} s")
    print("selftest:", "ok" if not fails else f"{fails} FAILED")
    return 1 if fails else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", help="walk.py --log recordings (.npz)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    if not a.files:
        ap.error("give a recording, or --selftest")
    for p in a.files:
        r = load(p)
        print(f"== {p}")
        print(report(analyse(r), r["gait"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
