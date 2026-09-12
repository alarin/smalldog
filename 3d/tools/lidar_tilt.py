"""LIDAR_TILT sweep: what each lean of the L2 buys, with the measured scan pattern.

Casts lidar.Scanner off `out/sim/mini_dog.xml` with the robot standing on a flat floor,
the sensor site re-posed for each tilt about the seat point mini_dog holds (the CAD is not
rebuilt - the site and the L2's visual body are the only things that move; the bracket
under them stays as exported, and it is in the sensor's own body, which the scanner
excludes anyway).  This is the sweep that put the axis on the horizon; keep it for the
next time the mount moves.  Three probes are added to the scene:

    step   50 mm tall, full width, 0.5 m ahead of the nose  - a threshold
    box    0.3 x 0.3 x 0.3 m, 1.0 m ahead                    - a chair / a wall corner
    leg    30 mm dia, 0.7 m tall, 1.5 m ahead                - a table leg

and for each tilt the per-frame (1/12 s) counts on each, on the floor strip the front
feet are about to step into (50..350 mm past the leading foot, +-100 mm), the near edge
of the floor on the centreline, the rays that return nothing (in this empty world: the
horizon and above, where a room's walls would be) and the rays that hit the robot's own
legs.  `body` is the margin of the deck lip - the top of the static robot - outside the
cone, measured like mini_dog.lidar_fov_clear from the optical centre; negative means the
lip is in the view at that tilt.

    .venv/bin/python tools/lidar_tilt.py                 # 20..60 deg by 5
    .venv/bin/python tools/lidar_tilt.py 30 45 --frames 12
"""
import argparse, math, os, re, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import mujoco
import mini_dog as md
import lidar as ld

XML = os.path.join(ROOT, "out", "sim", "mini_dog.xml")

PROBES = """
    <geom name="probe_step" type="box" pos="{step:.3f} 0 0.025" size="0.05 1.0 0.025" rgba="0.8 0.3 0.3 1"/>
    <geom name="probe_box"  type="box" pos="{box:.3f} 0 0.15" size="0.15 0.15 0.15" rgba="0.3 0.8 0.3 1"/>
    <geom name="probe_leg"  type="cylinder" pos="{leg:.3f} 0 0.35" size="0.015 0.35" rgba="0.3 0.3 0.8 1"/>
"""


def scene(tilt):
    """the exported MJCF with the sensor re-posed for `tilt` and the probes added"""
    md.LIDAR_TILT = tilt
    p, q, _ = ld.pose_m()
    c = tuple(v / 1000.0 for v in md.lidar_com())
    f = lambda v: " ".join(f"{x:.6g}" for x in v)
    x = open(XML).read()
    x, n1 = re.subn(r'(<site name="lidar" )pos="[^"]*" quat="[^"]*"',
                    rf'\1pos="{f(p)}" quat="{f(q)}"', x)
    x, n2 = re.subn(r'(<geom name="lidar_body" [^>]*?)pos="[^"]*" quat="[^"]*"',
                    rf'\1pos="{f(c)}" quat="{f(q)}"', x)
    assert n1 == 1 and n2 == 1, "no lidar site in out/sim/mini_dog.xml - run export_sim.py"
    nose = md.BODY_L / 2000.0
    x = x.replace("  </worldbody>", PROBES.format(step=nose + 0.5, box=nose + 1.0 + 0.15,
                                                   leg=nose + 1.5) + "  </worldbody>")
    return x


def lip_margin():
    """degrees the deck's stiffening lip sits outside the NEGA cone, at the current tilt"""
    (sx, _, sz), n = md.lidar_pose()
    o = (sx + n[0] * md.LIDAR_OPT, sz + n[2] * md.LIDAR_OPT)
    worst = 180.0
    for x, z in ((md.BODY_L / 2, md.BODY_Z1 + md.DECK_T + 6.0),
                 (md.OPI_X + md.OPI_HOLES[0] / 2, md.BODY_Z1 + md.DECK_T + md.OPI_STAND_H),
                 (md.GPS_X + md.GPS_PLATE[0] / 2, md.GPS_SEAT_Z)):
        d = (x - o[0], z - o[1]); L = math.hypot(*d)
        worst = min(worst, math.degrees(math.acos((d[0] * n[0] + d[1] * n[2]) / L)))
    return worst - md.LIDAR_FOV_NEGA


def run(tilt, frames):
    # from_xml_string resolves mesh files against the CWD, the export against itself
    path = os.path.join(os.path.dirname(XML), f"_tilt_{tilt:g}.xml")
    open(path, "w").write(scene(tilt))
    try:
        m = mujoco.MjModel.from_xml_path(path)
    finally:
        os.remove(path)
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = m.key_ctrl[0]
    for _ in range(1500):
        mujoco.mj_step(m, d)
    sc = ld.Scanner(m)
    if not sc.ok:
        sys.exit(sc.missing)
    gid = {n: m.geom(n).id for n in ("floor", "probe_step", "probe_box", "probe_leg")}
    feet = [m.geom(i).id for i in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i) or "").startswith("foot_")]
    foot = max(d.geom_xpos[i][0] for i in feet)
    acc = dict(step=0, box=0, leg=0, strip=0, sky=0, self=0, near=[])
    t = d.time
    for k in range(frames):
        c = sc.scan(m, d, t + k / sc.frame_hz, t + (k + 1) / sc.frame_hz)
        g, w = c["geom"], c["world"]
        for key in ("step", "box", "leg"):
            acc[key] += int((g == gid["probe_" + key]).sum())
        gnd = w[g == gid["floor"]]
        strip = gnd[(gnd[:, 0] > foot + 0.05) & (gnd[:, 0] < foot + 0.35) & (abs(gnd[:, 1]) < 0.1)]
        acc["strip"] += len(strip)
        mid = gnd[abs(gnd[:, 1]) < 0.05]
        acc["near"].append((mid[:, 0].min() - foot) if len(mid) else float("nan"))
        acc["sky"] += c["n_rays"] - len(g)
        acc["self"] += int(sum((g == i).sum() for i in range(m.ngeom)
                              if m.geom_bodyid[i] not in (0, sc.body) and i not in feet)
                          + sum((g == i).sum() for i in feet))
    n = frames
    return dict(tilt=tilt, body=lip_margin(), rays=c["n_rays"],
                step=acc["step"] / n, box=acc["box"] / n, leg=acc["leg"] / n,
                strip=acc["strip"] / n, near=float(np.nanmedian(acc["near"])) * 1000,
                sky=acc["sky"] / n / c["n_rays"], self=acc["self"] / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tilts", nargs="*", type=float, default=list(range(30, 125, 10)))
    ap.add_argument("--frames", type=int, default=6, help="frames averaged per row")
    a = ap.parse_args()
    print(f"probes: step 0.5 m, box 1.0 m, leg 1.5 m ahead of the nose; per frame"
          f" (1/{md.LIDAR_FRAME_HZ:.0f} s), {a.frames} frames averaged;"
          f" LIDAR_SEAT_Z {md.LIDAR_SEAT_Z:.0f}, tilt now {md.LIDAR_TILT:.0f}")
    print(f"{'tilt':>5} {'body':>6} | {'strip':>6} {'/cm2':>5} {'near':>6} |"
          f" {'step':>6} {'box':>6} {'leg':>5} | {'none':>5} {'legs':>5}")
    for tilt in a.tilts:
        r = run(tilt, a.frames)
        flag = "  !! deck in the view" if r["body"] < 0 else ""
        print(f"{tilt:5.0f} {r['body']:+6.1f} | {r['strip']:6.0f} {r['strip']/600:5.2f}"
              f" {r['near']:+6.0f} | {r['step']:6.0f} {r['box']:6.0f} {r['leg']:5.0f} |"
              f" {r['sky']*100:4.0f}% {r['self']:5.0f}{flag}")
    print("strip: floor returns 50..350 mm past the leading foot, +-100 mm (600 cm2);"
          " near: first centreline floor return past that foot, mm")


if __name__ == "__main__":
    main()
