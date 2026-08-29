#!/usr/bin/env python
"""
standalone_sim.py — SmallDog in MuJoCo with keyboard control, no ROS 2 needed.

    python tools/standalone_sim.py              # interactive viewer
    python tools/standalone_sim.py --headless   # self-test: stand, then trot
    python tools/standalone_sim.py --terrain    # same, on the rough-ground scene
    python tools/standalone_sim.py --blind      # ... with the terrain feedback switched off
    python tools/standalone_sim.py --course     # 25 s over the ramp/wall/log course
    python tools/standalone_sim.py --lidar      # ... plus the L2: cloud in the viewer,
                                                #     what it saw in the headless runs

Keys in the viewer window (same bindings as the ROS 2 teleop node):
    W/S forward/back   A/D strafe   Q/E turn   space stop
    R/F body up/down   ,/. slower/faster   T toggle gait
"""
import os, sys, math, json, time, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
WS   = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, "smalldog_walker"))

import numpy as np
import mujoco
from smalldog_walker.gait import TrotGait

MJCF   = os.path.join(WS, "smalldog_description", "mujoco")
PARAMS = os.path.join(WS, "smalldog_description", "robot_params.json")
CAD    = os.path.abspath(os.path.join(WS, "..", "3d"))


# --------------------------------------------------------------------------------- lidar
# The scan model lives in the CAD tree, in ../3d/lidar.py, next to the geometry it is a
# sensor for - the same file both exporters use to write the site and the <custom>
# numerics into the model.  There is deliberately no copy of it here: the parameters come
# out of the compiled model, and the code that turns them into rays comes out of that one
# file.  It is the only thing in this script that needs the CAD tree beside the workspace,
# and it is only needed with --lidar.
def scanner(model):
    if not os.path.isdir(CAD):
        raise SystemExit(f"--lidar needs the CAD tree at {CAD} (that is where lidar.py is)")
    sys.path.insert(0, CAD)
    import lidar
    sc = lidar.Scanner(model)
    if not sc.ok:
        raise SystemExit(f"--lidar: {sc.missing}"
                         "  (run smalldog_description/scripts/generate_model.py)")
    return sc


def lidar_report(model, data, sc, pad="         "):
    """One frame, described: coverage, what it hit, and how far ahead it reaches.

    The centreline figure is the one worth watching - it is the near edge of the cone the
    45 deg tilt exists to buy, and it says how much ground the robot sees before it walks
    onto it.
    """
    c = sc.scan(model, data)
    floor = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    gnd = c["world"][c["geom"] == floor]
    n = len(c["range"])
    print(f"{pad}lidar {c['n_rays']} rays/frame -> {n} returns"
          f" ({100.0*n/max(1, c['n_rays']):.0f} %),"
          f" {n - len(gnd)} of them not the ground;"
          f" range {c['range'].min():.2f}..{c['range'].max():.2f} m")
    mid = gnd[abs(gnd[:, 1] - data.qpos[1]) < 0.05]
    if len(mid):
        print(f"{pad}      ground on the centreline from x {mid[:, 0].min()*1000:+.0f} mm"
              f" ({(mid[:, 0].min() - data.qpos[0])*1000:+.0f} mm ahead of the body)"
              f" out to {mid[:, 0].max()*1000:+.0f} mm")
    return c


CLOUD_MAX = 6000            # spheres in the viewer; beyond this the frame rate is the cost

def draw_cloud(scn, frames, r=0.006):
    """Put the accumulated cloud into the viewer's own scene, coloured by height.

    World coordinates, several frames deep, and subsampled to CLOUD_MAX - the viewer redraws
    every one of these at 60 Hz, and a full 10 Hz frame is 2160 points before accumulation.
    """
    pts = np.concatenate(frames) if frames else np.zeros((0, 3))
    if len(pts) > CLOUD_MAX:
        pts = pts[::int(np.ceil(len(pts) / CLOUD_MAX))]
    eye = np.eye(3).reshape(-1)
    size = np.array([r, r, r])
    z0, z1 = (pts[:, 2].min(), pts[:, 2].max()) if len(pts) else (0.0, 1.0)
    scn.ngeom = 0
    for p in pts:
        if scn.ngeom >= scn.maxgeom:
            break
        u = (p[2] - z0) / max(1e-6, z1 - z0)
        mujoco.mjv_initGeom(scn.geoms[scn.ngeom], mujoco.mjtGeom.mjGEOM_SPHERE,
                            size, p, eye,
                            np.array([0.25 + 0.75*u, 0.85 - 0.5*u, 1.0 - 0.7*u, 1.0],
                                     dtype=np.float32))
        scn.ngeom += 1

def scene(terrain):
    return os.path.join(MJCF, "scene_terrain.xml" if terrain else "scene.xml")

def load(terrain=False):
    params = json.load(open(PARAMS))
    model = mujoco.MjModel.from_xml_path(scene(terrain))
    data = mujoco.MjData(model)
    gait = TrotGait(params)
    act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
           for n in gait.joint_names]
    if -1 in act:
        raise RuntimeError("actuator/joint name mismatch between MJCF and robot_params.json")
    return model, data, gait, act, params


def sensors(model):
    """address of each sensor the gait feeds on, by name. -1 if the model lacks it."""
    def adr(name):
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return model.sensor_adr[i] if i >= 0 else -1
    return dict(quat=adr("imu_quat"), gyro=adr("imu_gyro"),
                contact={l: adr(f"{l}_contact") for l in ("fl", "fr", "rl", "rr")})


def feed(gait, data, sens, blind=False):
    """hand the gait this step's IMU and foot contacts (`blind` = don't, stay open loop)"""
    if blind or sens["quat"] < 0:
        return
    q, g = sens["quat"], sens["gyro"]
    gait.feedback(quat=tuple(data.sensordata[q:q + 4]),
                  gyro=tuple(data.sensordata[g:g + 3]),
                  contact={l: data.sensordata[a] > 1e-6
                           for l, a in sens["contact"].items() if a >= 0})

def settle(model, data, gait, act, seconds=1.0, sens=None, blind=False):
    q = gait.joint_targets(0.0, 0.0, 0.0, 0.0)
    for i, a in enumerate(act):
        data.ctrl[a] = q[i]
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        if sens:
            feed(gait, data, sens, blind)

def run(model, data, gait, act, cmd, seconds, sens=None, blind=False):
    n = int(seconds / model.opt.timestep)
    for _ in range(n):
        q = gait.joint_targets(model.opt.timestep, *cmd)
        for i, a in enumerate(act):
            data.ctrl[a] = q[i]
        mujoco.mj_step(model, data)
        if sens:
            feed(gait, data, sens, blind)

def state(data):
    x, y, z = data.qpos[0:3]
    w, qx, qy, qz = data.qpos[3:7]
    roll = math.atan2(2*(w*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    pitch = math.asin(max(-1, min(1, 2*(w*qy - qz*qx))))
    return dict(x=x, y=y, z=z, roll=math.degrees(roll), pitch=math.degrees(pitch))

def headless(terrain=False, blind=False, lidar=False):
    model, data, gait, act, params = load(terrain)
    sens = sensors(model)
    sc = scanner(model) if lidar else None
    print(f"scene: {os.path.basename(scene(terrain))}"
          f"   gait: {'open loop (blind)' if blind else 'IMU + foot contact'}")
    print(f"model ok: {model.nq} dof, {model.nu} actuators, "
          f"mass {sum(model.body_mass):.3f} kg")
    mujoco.mj_resetData(model, data)
    settle(model, data, gait, act, 1.5, sens, blind)
    s0 = state(data)
    print(f"  stand    z={s0['z']*1000:6.1f} mm  roll={s0['roll']:+5.1f} pitch={s0['pitch']:+5.1f}")
    if sc:
        lidar_report(model, data, sc)
    run(model, data, gait, act, (0.0, 0.0, 0.0), 1.0, sens, blind)
    s1 = state(data)
    print(f"  hold 1s  z={s1['z']*1000:6.1f} mm  roll={s1['roll']:+5.1f} pitch={s1['pitch']:+5.1f}"
          f"  drift={math.hypot(s1['x'],s1['y'])*1000:5.1f} mm")
    run(model, data, gait, act, (0.20, 0.0, 0.0), 5.0, sens, blind)
    s2 = state(data)
    print(f"  trot 5s  z={s2['z']*1000:6.1f} mm  roll={s2['roll']:+5.1f} pitch={s2['pitch']:+5.1f}"
          f"  travelled x={(s2['x']-s1['x'])*1000:6.1f} mm  y={(s2['y']-s1['y'])*1000:6.1f} mm")
    if sc:
        lidar_report(model, data, sc)
    run(model, data, gait, act, (0.0, 0.0, 1.2), 4.0, sens, blind)
    s3 = state(data)
    print(f"  turn 4s  z={s3['z']*1000:6.1f} mm  roll={s3['roll']:+5.1f} pitch={s3['pitch']:+5.1f}")
    ok = (s2["z"] > 0.10 and abs(s2["roll"]) < 25 and abs(s2["pitch"]) < 25
          and (s2["x"] - s1["x"]) > 0.20)
    print("RESULT:", "OK — stands and trots forward" if ok else "FAIL")
    return 0 if ok else 1

KEYMAP = {
    ord('W'): ("vx", +1), ord('S'): ("vx", -1),
    ord('A'): ("vy", +1), ord('D'): ("vy", -1),
    ord('Q'): ("wz", +1), ord('E'): ("wz", -1),
    ord(' '): ("stop", 0),
    ord('R'): ("h", +1), ord('F'): ("h", -1),
    ord(','): ("scale", -1), ord('.'): ("scale", +1),
    ord('T'): ("toggle", 0),
}

def interactive(terrain=False, blind=False, lidar=False):
    import mujoco.viewer
    model, data, gait, act, params = load(terrain)
    sens = sensors(model)
    sc = scanner(model) if lidar else None
    cmd = {"vx": 0.0, "vy": 0.0, "wz": 0.0}
    st = {"scale": 0.20, "wscale": 1.2, "enabled": True}

    def on_key(keycode):
        k = KEYMAP.get(keycode if keycode < 128 else keycode)
        if k is None:
            return
        what, sign = k
        if what == "stop":
            cmd.update(vx=0.0, vy=0.0, wz=0.0)
        elif what == "vx":  cmd["vx"] = sign * st["scale"]
        elif what == "vy":  cmd["vy"] = sign * st["scale"]
        elif what == "wz":  cmd["wz"] = sign * st["wscale"]
        elif what == "h":   gait.body_height = min(0.20, max(0.09, gait.body_height + sign*0.005))
        elif what == "scale":
            st["scale"] = min(0.45, max(0.05, st["scale"] + sign * 0.05))
            print(f"speed {st['scale']:.2f} m/s")
        elif what == "toggle":
            st["enabled"] = not st["enabled"]
            print("gait", "on" if st["enabled"] else "off")

    print(__doc__)
    # the cloud is drawn from the last CLOUD_FRAMES frames, in world coordinates, which is
    # what a non-repetitive scan is for: stand still and the picture keeps filling in.
    CLOUD_FRAMES, cloud = 6, []
    # Pace the loop against the wall clock, and sync the viewer at VIEW_HZ rather than once
    # per step.  Neither is cosmetic: the physics runs 26-37x real time here (measured,
    # 1 ms steps, flat and heightfield), so an unpaced loop does not run "fast" - it runs
    # as fast as `sync()` allows, which is a scene copy under a lock and an order of
    # magnitude dearer than the step it was following.  The result was a robot in slow
    # motion for no reason at all.  `catchup` caps how much sim time one wall-clock frame
    # may make up, so that a window that was dragged, or a scene heavy enough to fall
    # behind, degrades to slow motion instead of sprinting to catch up.
    VIEW_HZ, CATCHUP = 60.0, 0.10
    dt = model.opt.timestep
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as v:
        wall0, sim0 = time.perf_counter(), data.time
        next_sync = data.time
        while v.is_running():
            elapsed = time.perf_counter() - wall0
            behind = (sim0 + elapsed) - data.time
            if behind <= 0:                     # ahead of the clock: hand the time back
                time.sleep(min(0.005, -behind))
                continue
            if behind > CATCHUP:                # too far behind to ever catch up: give up
                wall0, sim0 = time.perf_counter(), data.time
                behind = dt
            for _ in range(max(1, int(behind / dt))):
                c = (cmd["vx"], cmd["vy"], cmd["wz"]) if st["enabled"] else (0.0, 0.0, 0.0)
                q = gait.joint_targets(dt, *c)
                for i, a in enumerate(act):
                    data.ctrl[a] = q[i]
                mujoco.mj_step(model, data)
                feed(gait, data, sens, blind)
                if sc:
                    f = sc.frame(model, data)
                    if f is not None:
                        cloud = (cloud + [f["world"]])[-CLOUD_FRAMES:]
            if data.time >= next_sync:
                if sc and cloud:
                    draw_cloud(v.user_scn, cloud)
                v.sync()
                next_sync = data.time + 1.0 / VIEW_HZ

def course(blind=False, lidar=False):
    """Walk the obstacle course in scene_terrain.xml and report how far it got.

    Deterministic, unlike a seed sweep: the scene is always written at terrain.SEED, so
    this is one repeatable number rather than a sample.  It is a *report*, not the gait
    regression — that is `--headless --terrain`, whose 5 s trot deliberately stops short
    of the first obstacle.  Judge a gait change on both, and remember that across terrain
    seeds the course spreads +-570 mm and puts the robot down on 2 seeds in 6, so a single
    run moving by a few hundred mm has not told you anything.

    An obstacle counts as cleared only if the robot got past it *while still inside its
    width*.  It has no heading control and wanders on rough ground — on the default seed
    it ends 1.65 m off the centreline — so scoring on x alone credits it for obstacles it
    walked around.
    """
    model, data, gait, act, params = load(True)
    sens = sensors(model)
    sc = scanner(model) if lidar else None
    nm = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
    obs = [(nm(g), float(model.geom_pos[g][0]), float(model.geom_size[g][1]))
           for g in range(model.ngeom) if nm(g).startswith("obs")]
    label = lambda n: n.split("_", 1)[1]
    print(f"scene: {os.path.basename(scene(True))}"
          f"   gait: {'open loop (blind)' if blind else 'IMU + foot contact'}")
    print("course: " + ", ".join(f"{label(n)} at {x*1000:.0f}" for n, x, _ in obs) + " mm")
    mujoco.mj_resetData(model, data)
    settle(model, data, gait, act, 1.5, sens, blind)
    run(model, data, gait, act, (0.0, 0.0, 0.0), 1.0, sens, blind)
    s0 = state(data)
    cleared, reach = set(), 0.0
    # with the lidar on, note how far ahead each obstacle was FIRST seen.  The gait cannot
    # use it - nothing is wired to it - so this is a measurement of the sensor, not of the
    # walker: how much warning a perception layer would have had.
    seen = {}
    for i in range(250):
        run(model, data, gait, act, (0.20, 0.0, 0.0), 0.1, sens, blind)
        s = state(data)
        if sc:
            f = sc.scan(model, data)
            for g in set(int(v) for v in f["geom"]):
                if nm(g).startswith("obs") and nm(g) not in seen:
                    d = f["world"][f["geom"] == g]
                    seen[nm(g)] = float(min(d[:, 0]) - s["x"])
        for n, x, half in obs:
            if s["x"] > x + 0.09 and abs(s["y"]) < half:
                cleared.add(n)
        if abs(s["y"]) < 0.35:
            reach = max(reach, s["x"])
        if (i + 1) % 50 == 0:
            print(f"  {(i+1)//10:3d}s  x={s['x']*1000:7.1f} mm  y={s['y']*1000:+7.1f}"
                  f"  z={s['z']*1000:5.1f}  roll={s['roll']:+5.1f} pitch={s['pitch']:+5.1f}")
    s1 = state(data)
    got = [label(n) for n, _, _ in obs if n in cleared]
    print(f"  cleared {len(got)}/{len(obs)}: {' '.join(got) or '(none)'}")
    if sc:
        print("  lidar first saw: " + (", ".join(
            f"{label(n)} {seen[n]*1000:.0f} mm ahead" for n, _, _ in obs if n in seen)
            or "(nothing — it never got near one)"))
    print(f"  furthest x while inside the course corridor: {reach*1000:.0f} mm")
    up = s1["z"] > 0.10 and abs(s1["roll"]) < 25 and abs(s1["pitch"]) < 25
    print(f"RESULT: {'upright' if up else 'DOWN'} at x={s1['x']*1000:.0f} mm,"
          f" y={s1['y']*1000:+.0f} mm after 25 s"
          f"  (travelled {math.hypot(s1['x']-s0['x'], s1['y']-s0['y'])*1000:.0f} mm)")
    return 0 if up else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--terrain", action="store_true",
                    help="run on mujoco/scene_terrain.xml (heightfield) instead of the plane")
    ap.add_argument("--blind", action="store_true",
                    help="do not feed the gait the IMU or the foot contacts — the "
                         "open-loop baseline, for before/after on rough ground")
    ap.add_argument("--course", action="store_true",
                    help="25 s trot over the ramp/wall/log course in scene_terrain.xml,"
                         " reported obstacle by obstacle (implies --terrain)")
    ap.add_argument("--lidar", action="store_true",
                    help="scan the Unitree L2 out of the model (../3d/lidar.py): the cloud"
                         " is drawn in the viewer, and the headless runs report what it"
                         " saw.  Nothing is wired to the gait — it still walks blind")
    a = ap.parse_args()
    sys.exit(course(a.blind, a.lidar) if a.course
             else (headless(a.terrain, a.blind, a.lidar) if a.headless
                   else (interactive(a.terrain, a.blind, a.lidar) or 0)))
