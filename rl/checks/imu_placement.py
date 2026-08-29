#!/usr/bin/env python
"""
imu_placement.py — how much does it matter WHERE the BMI088 is bolted?

    python checks/imu_placement.py                      # the default candidates
    python checks/imu_placement.py --at -22,0,49        # one point, mm, base_link frame
    python checks/imu_placement.py --speed 0.35         # ... at a faster trot

An accelerometer that is not at the point the model calls `imu` does not read
what the model's accelerometer reads.  Rigidly attached at offset r from that
point it also picks up

    a_extra = omega x (omega x r) + alpha x r

which on a trotting robot is dominated by `alpha` at footfall.  This script
measures the size of that term on the gait that already exists, by adding real
accelerometer sensors at candidate mounting points through MjSpec and comparing
their output to the site the MJCF ships with.  No derivation to get wrong: the
numbers are what MuJoCo's own sensors report.

Why it exists.  The answer decides an ordering, not a tuning knob.  If the term
is small, the board goes wherever it fits and the observation can be frozen now.
If it is not, the mounting point has to be chosen in `3d/mini_dog.py` first, the
`imu` site follows it out of the generator, and only then can step 4 fix the
observation vector.  Re-run it whenever the mount or the gait moves.

It reads the trot out of ros2/smalldog_walker, the same way ros2/tools/
standalone_sim.py does — the gait has no ROS imports, and using the real one
beats exciting the robot with something invented for this measurement.
"""
import argparse, json, math, os, sys
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
DESC = os.path.join(ROOT, "ros2", "smalldog_description")
sys.path.insert(0, os.path.join(ROOT, "ros2", "smalldog_walker"))

# Candidates, in mm in the base_link frame. The origin is where the MJCF's `imu`
# site sits today; the rest are places a 20 x 15 mm board could plausibly go.
# `deck` is the obvious one — flat, accessible, next to the Orange Pi — and it is
# also the worst of these, which is the point of measuring.
DEFAULT = {
    "model imu site": (0, 0, 0),
    "25 mm forward": (25, 0, 0),
    "50 mm forward": (50, 0, 0),
    "50 mm up": (0, 0, 50),
    "deck, by the Pi": (-22, 0, 49),
    "under the deck": (0, 0, 26),
}


def build(offsets_mm, scene):
    """The generated MJCF plus one accelerometer/gyro pair per candidate.

    Done through MjSpec rather than by editing XML on purpose: `3d/CLAUDE.md`
    makes the description generated output, never hand-tuned, and this is a
    measurement tool, not a model change. It is also the same mechanism rl/model.py
    will use to add backlash and a fitted actuator for training.
    """
    spec = mujoco.MjSpec.from_file(scene)
    base = spec.body("base_link")
    for i, (name, mm) in enumerate(offsets_mm.items()):
        site = f"probe{i}"
        base.add_site(name=site, pos=[v * 1e-3 for v in mm], size=[0.003, 0, 0])
        for kind, tag in ((mujoco.mjtSensor.mjSENS_ACCELEROMETER, "acc"),
                          (mujoco.mjtSensor.mjSENS_GYRO, "gyr")):
            s = spec.add_sensor()
            s.name = f"{tag}{i}"
            s.type = kind
            s.objtype = mujoco.mjtObj.mjOBJ_SITE
            s.objname = site
    return spec.compile()


def run(m, P, speed, seconds, settle):
    from smalldog_walker.gait import TrotGait
    d = mujoco.MjData(m)
    gait = TrotGait(P)
    gait.body_height = 0.158
    act = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in P["joint_names"]]
    for n in P["joint_names"]:
        kind = n.split("_")[1]
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)
        d.qpos[m.jnt_qposadr[j]] = 0.0 if kind == "roll" else P["stance_rad"][kind]
    d.qpos[2] = P["stance_base_height_m"]

    DT = 0.01                                   # the walker's own 100 Hz tick
    sub = max(1, int(round(DT / m.opt.timestep)))
    n_probe = sum(1 for i in range(m.nsensor)
                  if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_SENSOR, i) or "").startswith("acc"))
    adr = {t: [m.sensor_adr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, f"{t}{i}")]
               for i in range(n_probe)] for t in ("acc", "gyr")}
    acc, fell = [], False
    for step in range(int(seconds / DT)):
        vx = 0.0 if step * DT < settle else speed
        for a, v in zip(act, gait.joint_targets(DT, vx, 0.0, 0.0)):
            d.ctrl[a] = v
        for _ in range(sub):
            mujoco.mj_step(m, d)
        if step * DT >= settle + 0.5:
            acc.append([d.sensordata[a:a + 3].copy() for a in adr["acc"]])
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, d.qpos[3:7])
        if R.reshape(3, 3)[2, 2] < 0.5:
            fell = True
            break
    return np.array(acc), fell, float(d.qpos[0])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--at", action="append", default=None,
                    help="candidate as x,y,z in mm, base_link frame; repeatable")
    ap.add_argument("--speed", type=float, default=0.20, help="trot speed, m/s")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--settle", type=float, default=1.0, help="stand this long first, s")
    ap.add_argument("--terrain", action="store_true")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()

    cands = dict(DEFAULT)
    if a.at:
        cands = {"model imu site": (0, 0, 0)}
        for s in a.at:
            v = tuple(float(x) for x in s.split(","))
            cands[f"({s})"] = v

    scene = os.path.join(DESC, "mujoco",
                         "scene_terrain.xml" if a.terrain else "scene.xml")
    P = json.load(open(os.path.join(DESC, "robot_params.json")))
    m = build(cands, scene)
    acc, fell, x = run(m, P, a.speed, a.seconds, a.settle)
    print(f"scene   {os.path.basename(scene)}")
    print(f"trot    {a.speed:g} m/s, {a.seconds:g} s, travelled {x*1000:.0f} mm"
          + ("   !! it fell" if fell else ""))
    print(f"samples {len(acc)} at 100 Hz\n")

    ref = acc[:, 0, :]
    print(f"{'mount point':<20}{'offset mm':<18}{'|d a| p50':>10}{'p95':>8}{'max':>8}"
          f"{'  worst apparent tilt':>22}")
    out = {}
    for i, (name, mm) in enumerate(cands.items()):
        e = np.linalg.norm(acc[:, i, :] - ref, axis=1)
        p50, p95, mx = (*np.percentile(e, [50, 95]), e.max())
        tilt = math.degrees(math.atan(mx / 9.81))
        print(f"{name:<20}{str(tuple(int(v) for v in mm)):<18}"
              f"{p50:10.3f}{p95:8.3f}{mx:8.3f}{tilt:19.1f} deg")
        out[name] = dict(offset_mm=list(mm), p50=p50, p95=p95, max=mx, tilt_deg=tilt)
    print("\nunits m/s^2 against 9.81. 'apparent tilt' is how far the gravity direction")
    print("appears to move at the peak — the quantity the policy actually observes.")
    print("This is not noise: it is a deterministic function of the robot's own motion,")
    print("so a policy trained at one mounting point and deployed at another sees a bias")
    print("correlated with its own actions, which is the hardest kind to diagnose later.")
    if a.json:
        json.dump(out, open(a.json, "w"), indent=2)
        print(f"\nwrote {a.json}")


if __name__ == "__main__":
    main()
