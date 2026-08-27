#!/usr/bin/env python
"""
standalone_sim.py — SmallDog in MuJoCo with keyboard control, no ROS 2 needed.

    python tools/standalone_sim.py              # interactive viewer
    python tools/standalone_sim.py --headless   # self-test: stand, then trot

Keys in the viewer window (same bindings as the ROS 2 teleop node):
    W/S forward/back   A/D strafe   Q/E turn   space stop
    R/F body up/down   ,/. slower/faster   T toggle gait
"""
import os, sys, math, json, argparse

HERE = os.path.dirname(os.path.abspath(__file__))
WS   = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, "smalldog_walker"))

import numpy as np
import mujoco
from smalldog_walker.gait import TrotGait

SCENE  = os.path.join(WS, "smalldog_description", "mujoco", "scene.xml")
PARAMS = os.path.join(WS, "smalldog_description", "robot_params.json")

def load():
    params = json.load(open(PARAMS))
    model = mujoco.MjModel.from_xml_path(SCENE)
    data = mujoco.MjData(model)
    gait = TrotGait(params)
    act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
           for n in gait.joint_names]
    if -1 in act:
        raise RuntimeError("actuator/joint name mismatch between MJCF and robot_params.json")
    return model, data, gait, act, params

def settle(model, data, gait, act, seconds=1.0):
    q = gait.joint_targets(0.0, 0.0, 0.0, 0.0)
    for i, a in enumerate(act):
        data.ctrl[a] = q[i]
    for _ in range(int(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)

def run(model, data, gait, act, cmd, seconds):
    n = int(seconds / model.opt.timestep)
    for _ in range(n):
        q = gait.joint_targets(model.opt.timestep, *cmd)
        for i, a in enumerate(act):
            data.ctrl[a] = q[i]
        mujoco.mj_step(model, data)

def state(data):
    x, y, z = data.qpos[0:3]
    w, qx, qy, qz = data.qpos[3:7]
    roll = math.atan2(2*(w*qx + qy*qz), 1 - 2*(qx*qx + qy*qy))
    pitch = math.asin(max(-1, min(1, 2*(w*qy - qz*qx))))
    return dict(x=x, y=y, z=z, roll=math.degrees(roll), pitch=math.degrees(pitch))

def headless():
    model, data, gait, act, params = load()
    print(f"model ok: {model.nq} dof, {model.nu} actuators, "
          f"mass {sum(model.body_mass):.3f} kg")
    mujoco.mj_resetData(model, data)
    settle(model, data, gait, act, 1.5)
    s0 = state(data)
    print(f"  stand    z={s0['z']*1000:6.1f} mm  roll={s0['roll']:+5.1f} pitch={s0['pitch']:+5.1f}")
    run(model, data, gait, act, (0.0, 0.0, 0.0), 1.0)
    s1 = state(data)
    print(f"  hold 1s  z={s1['z']*1000:6.1f} mm  roll={s1['roll']:+5.1f} pitch={s1['pitch']:+5.1f}"
          f"  drift={math.hypot(s1['x'],s1['y'])*1000:5.1f} mm")
    run(model, data, gait, act, (0.20, 0.0, 0.0), 5.0)
    s2 = state(data)
    print(f"  trot 5s  z={s2['z']*1000:6.1f} mm  roll={s2['roll']:+5.1f} pitch={s2['pitch']:+5.1f}"
          f"  travelled x={(s2['x']-s1['x'])*1000:6.1f} mm  y={(s2['y']-s1['y'])*1000:6.1f} mm")
    run(model, data, gait, act, (0.0, 0.0, 1.2), 4.0)
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

def interactive():
    import mujoco.viewer
    model, data, gait, act, params = load()
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
    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as v:
        while v.is_running():
            c = (cmd["vx"], cmd["vy"], cmd["wz"]) if st["enabled"] else (0.0, 0.0, 0.0)
            q = gait.joint_targets(model.opt.timestep, *c)
            for i, a in enumerate(act):
                data.ctrl[a] = q[i]
            mujoco.mj_step(model, data)
            v.sync()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true")
    a = ap.parse_args()
    sys.exit(headless() if a.headless else (interactive() or 0))
