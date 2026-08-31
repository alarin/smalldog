#!/usr/bin/env python
"""
replay.py — a hundred robots on one screen, replayed kinematically.

    python replay.py runs/A --robots 100                  # one policy, a herd
    python replay.py runs/A runs/B runs/C --robots 96     # a ROW PER CHECKPOINT
    python replay.py runs/A --robots 25 --seconds 6 --out runs/A/herd.mp4

The second form is the one worth having. Front row epoch 0 face-planting, back
row the latest, everything in between, one frame: that reads as progress rather
than as a hundred identical failures, and a regression between checkpoints is
visible instantly instead of being a wobble in a reward curve. A single-policy
herd is the degenerate case of it with one checkpoint repeated.

Why this is a debugger and not a demo
-------------------------------------
A FAILING policy produces trajectories too, and they are more informative than a
working one's. Watching a hundred copies belly-flop separates dragging from one
leg twitching from diving on spawn, long before the reward curve separates those
cases — and the reward curve may never separate them at all, because all three
score about the same.

Why it is a separate offline pass
---------------------------------
MJX has no renderer, and `num_envs` is an array axis, not a hundred robots
standing in a world. So this cannot be a training byproduct — which is the good
outcome here: rendering competes for nothing while PPO holds the card.

The load-bearing trick: KINEMATIC replay
----------------------------------------
The rollout records qpos and nothing else — megabytes, and the physics is already
done. The render scene then writes those qpos values in and calls `mj_forward`,
never `mj_step`. Two consequences, both of which are the whole point:

  * No physics in the render scene means no contacts BETWEEN the copies. They can
    be packed as tightly as the grid spacing allows, and 100 x ~40 geoms costs
    the renderer and nothing else. Stepping them would have a hundred robots
    colliding with each other and with a hundred floors.
  * The video shows exactly the trajectory that was recorded. A re-simulation
    would drift — different contact ordering, and MuJoCo is only deterministic
    for a given model, seed and version — and then the video would be of a run
    that never happened.

The scene is assembled through MjSpec, like everything else in this tree: nothing
is written into ros2/smalldog_description, and the grid is parameterised by N
rather than being a hand-written XML with a hundred copy-pasted bodies.

Output goes to runs/, which is gitignored. If a clip is worth keeping it is worth
handing to a person, not committing.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def parse():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+",
                    help="one or more runs/<name> directories. More than one "
                         "gives a row per checkpoint, oldest at the front.")
    ap.add_argument("--robots", type=int, default=100)
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--command", default="0.4,0,0", help="vx,vy,yaw")
    ap.add_argument("--spacing", type=float, default=0.55,
                    help="metres between copies. They cannot collide, so this is "
                         "a framing choice and not a physical one.")
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=960)
    ap.add_argument("--out", default=None)
    ap.add_argument("--mem-fraction", type=float, default=0.60)
    ap.add_argument("--elevation", type=float, default=-18.0)
    ap.add_argument("--azimuth", type=float, default=135.0)
    return ap.parse_args()


# ================================================================= recording
def record(run_dirs, n_each, seconds, command, mem_fraction):
    """Roll every checkpoint out in MJX and keep only qpos.

    Returns (qpos, labels): qpos is (T, n_robots, nq) and labels names which
    checkpoint each robot came from.
    """
    import jax
    import jax.numpy as jnp
    from brax.io import model as brax_io_model
    from brax.training.acme import running_statistics
    from brax.training.agents.ppo import networks as ppo_networks

    from env import Walk

    env = None
    tracks, labels = [], []
    n_steps = int(seconds * 50)

    for run_dir in run_dirs:
        with open(os.path.join(run_dir, "run.json")) as f:
            targs = json.load(f)["args"]
        if env is None:
            env = Walk(terrain=targs["terrain"], n_boxes=targs["boxes"])
            reset = jax.jit(jax.vmap(env.reset))
            step = jax.jit(jax.vmap(env.step))

        networks = ppo_networks.make_ppo_networks(
            observation_size=env.observation_size, action_size=env.action_size,
            preprocess_observations_fn=running_statistics.normalize,
            policy_hidden_layer_sizes=(128, 128, 128),
            value_hidden_layer_sizes=(256, 256, 256))
        params = brax_io_model.load_params(os.path.join(run_dir, "params"))
        if len(params) > 2:
            params = params[:2]
        policy = jax.jit(ppo_networks.make_inference_fn(networks)(
            params, deterministic=True))

        keys = jax.random.split(jax.random.PRNGKey(0), n_each)
        st = reset(keys)
        cmd = jnp.tile(jnp.asarray(command), (n_each, 1))
        st = st.replace(info={**st.info, "command": cmd})

        qs = []
        for _ in range(n_steps):
            act, _ = policy(st.obs, jax.random.PRNGKey(0))
            st = step(st, act)
            # The command is held: an episode that ends is auto-reset by nothing
            # here, so a fallen robot stays fallen and that is what we want to see.
            st = st.replace(info={**st.info, "command": cmd})
            qs.append(np.asarray(st.pipeline_state.qpos))
        tracks.append(np.stack(qs))                      # (T, n_each, nq)
        labels += [os.path.basename(run_dir.rstrip("/"))] * n_each
        print(f"  recorded {n_each:3d} x {seconds:g} s from "
              f"{os.path.basename(run_dir.rstrip('/'))}")

    return np.concatenate(tracks, axis=1), labels, env


# ============================================================== the render scene
def build_grid(n, spacing):
    """One ground plane and n copies of the robot, assembled through MjSpec.

    Copies are attached at the origin and placed by their own free joint at
    replay time, so the grid offset is arithmetic on qpos rather than a hundred
    frames in the model. They share one set of meshes rather than carrying a
    prefixed copy each, which is what makes n=100 fit in memory at all -- see
    the comment on the deduplication below. Returns (model, nq_per_robot,
    offsets).
    """
    import mujoco
    import model as model_mod

    scene = os.path.join(model_mod.MJCF, "scene.xml")
    base = mujoco.MjSpec.from_file(scene)

    # Everything after the first copy is the same spec with its world furniture
    # removed — one floor, one light, one skybox for the whole herd.
    def stripped():
        s = mujoco.MjSpec.from_file(scene)
        for g in list(s.worldbody.geoms):
            if g.name == "floor":
                s.delete(g)
        for lt in list(s.worldbody.lights):
            s.delete(lt)
        return s

    child = stripped()
    prefixes = []
    for i in range(1, n):
        # attach() needs a frame to hang the child on. It sits at the origin and
        # stays there: every copy's root is a freejoint, so its pose comes
        # entirely from qpos and the frame's position would be ignored anyway.
        # The grid is arithmetic on qpos at replay time instead.
        # child.copy() and not child: attach() RENAMES the child it is given, so
        # reusing one spec produces c2_c1_fl_roll on the second pass and then a
        # compile error about incompatible actuator ids. Copy per attachment.
        prefixes.append(f"c{i}_")
        base.attach(child.copy(), prefix=prefixes[-1], frame=base.worldbody.add_frame())

    # The copies share one set of meshes, and that is the difference between a
    # 1.3 GB render scene and a 9 GB one that takes the machine down with it.
    # attach() prefixes ASSET names along with everything else, so a hundred
    # copies arrive as a hundred sets of the robot's 25 meshes -- 2500 copies of
    # the same 62k vertices, which MuJoCo has no way to know are identical.
    # Measured on this box: n=16 costs 1545 MB as attached and 475 MB
    # deduplicated, and n=100 never finishes -- it reached 9.4 GB of virtual
    # memory in 7.7 GB of RAM and the OOM killer took WSL down with it.
    # Deduplicating renders a pixel-for-pixel identical frame, verified at n=4.
    # It is safe because the mesh geoms here are visual only (group 2,
    # contype 0) -- collision lives on the primitives, and this scene never
    # collides anything anyway.
    if prefixes:
        pset = tuple(prefixes)
        for b in base.bodies:
            for g in b.geoms:
                for attr in ("meshname", "material"):
                    v = getattr(g, attr, "") or ""
                    if v.startswith(pset):
                        setattr(g, attr, v.split("_", 1)[1])
        for assets in (base.meshes, base.materials):
            for a in list(assets):
                if (a.name or "").startswith(pset):
                    base.delete(a)

    m = base.compile()
    nq = m.nq // n
    assert nq * n == m.nq, f"{m.nq} qpos does not divide by {n} copies"

    side = int(np.ceil(np.sqrt(n)))
    offsets = np.zeros((n, 2))
    for i in range(n):
        r, c = divmod(i, side)
        offsets[i] = [(c - (side - 1) / 2) * spacing, -r * spacing]
    return m, nq, offsets


def render(m, nq, offsets, qpos, out, fps, w, h, elevation, azimuth, spacing):
    """Write qpos in, mj_forward, one frame. Never mj_step."""
    import mujoco
    import imageio

    os.environ.setdefault("MUJOCO_GL", "egl")
    n = offsets.shape[0]
    T = qpos.shape[0]

    d = mujoco.MjData(m)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    side = int(np.ceil(np.sqrt(n)))
    cam.lookat = [0.0, -side * spacing * 0.35, 0.15]
    cam.distance = max(2.0, side * spacing * 1.9)
    cam.elevation = elevation
    cam.azimuth = azimuth

    renderer = mujoco.Renderer(m, h, w)
    t0 = time.time()
    with imageio.get_writer(out, fps=fps, macro_block_size=1) as vid:
        for t in range(T):
            q = qpos[t].copy()                            # (n, nq_robot)
            q[:, 0] += offsets[:, 0]
            q[:, 1] += offsets[:, 1]
            d.qpos[:] = q.reshape(-1)
            mujoco.mj_forward(m, d)
            renderer.update_scene(d, cam)
            vid.append_data(renderer.render())
    el = time.time() - t0
    print(f"  rendered {T} frames of {n} robots ({m.ngeom} geoms) in "
          f"{el:.1f} s — {T/el:.1f} fps, {el/T*1000:.0f} ms/frame")
    return el


def main():
    a = parse()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import jaxenv
    jaxenv.configure(a.mem_fraction)

    cmd = [float(x) for x in a.command.split(",")]
    n_ckpt = len(a.runs)
    n_each = max(1, a.robots // n_ckpt)
    n = n_each * n_ckpt
    out = a.out or os.path.join(a.runs[0].rstrip("/"), "herd.mp4")

    print(f"\n{n} robots = {n_ckpt} checkpoint(s) x {n_each}, "
          f"command {cmd}, {a.seconds:g} s")
    qpos, labels, env = record(a.runs, n_each, a.seconds, cmd, a.mem_fraction)
    print(f"  qpos {qpos.shape} = {qpos.nbytes/1e6:.1f} MB — this is the whole "
          f"recording; the physics is done and never runs again")

    m, nq, offsets = build_grid(n, a.spacing)
    if nq != qpos.shape[2]:
        print(f"!! the render scene has {nq} qpos per robot and the rollout "
              f"recorded {qpos.shape[2]}; they must be the same model")
        return 1

    render(m, nq, offsets, qpos, out, a.fps, a.width, a.height,
           a.elevation, a.azimuth, a.spacing)
    print(f"\nwrote {out}")
    print("runs/ is gitignored — hand the file to a person, do not commit it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
