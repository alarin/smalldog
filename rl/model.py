#!/usr/bin/env python
"""
model.py — the model we train against: the generated MJCF plus the edits that
training, and only training, needs.

    python model.py                 # report every edit, and whether MJX takes it
    python model.py --terrain       # ... on the heightfield scene
    python model.py --boxes 24      # ... with procedural boxes under the robot

`ros2/smalldog_description` is output of `3d/mini_dog.py`. Nothing here writes to
it. Every change below is made through `mujoco.MjSpec` at load time, which is the
mechanism `checks/imu_placement.py` demonstrates and `rl/CLAUDE.md` requires:
nothing hand-tuned may land in a committed XML, because the committed XMLs are
not ours.

What gets changed, and why each one is a training concern rather than a model fix
--------------------------------------------------------------------------------

1. POSITION ACTUATORS BECOME TORQUE ACTUATORS.
   The committed model drives each joint with MuJoCo's `position` actuator at
   kp=25, dampratio=1. That is a controller we do not have. The robot has an
   ST3215: a voltage-driven brushed motor behind its own register-configured
   inner loop, whose torque ceiling and speed limit both fall as the pack drains.
   `actuator.py` is that law. So the actuators here are plain motors and the
   torque they receive is computed by `actuator.py` at every control step.
   The kp=25 in the MJCF is not wrong — it is `3d/mini_dog.py`'s stand-in for a
   servo, good enough for the analytic trot in `ros2/`, and check_model.py lists
   it under "GUESSED — and not a servo model".

2. ARMATURE BECOMES THE REFLECTED ROTOR INERTIA, DAMPING GOES TO ZERO, AND
   FRICTIONLOSS BECOMES THE MEASURED COULOMB FLOOR.
   Not a tightening — a move. `armature`, `damping` and `frictionloss` in the
   MJCF are three of the four guesses check_model.py flags. `actuator.py`
   supplies viscous friction (b_v) and back-EMF damping (k_w = k_u*k_e), so
   leaving MuJoCo's own damping in place counts the same physics twice. The
   reflected inertia J_m has to stay in MuJoCo — it is inertia, it belongs in the
   mass matrix, and at 1:345 it is ~151x the knee link's own, which check_model.py
   measures and calls the dominant term.
   It is also where J_m is RANDOMISED, for the same reason: `env/randomize.py`
   scales `dof_armature` per environment, because that field is what the physics
   reads. A per-episode draw on `actuator.Params.J_m` moved nothing at all — no
   function on the training path reads it — and it shipped that way until
   check_model.py's probe was taught to ask whether a drawn field is consumed.

   `frictionloss` USED to go to zero on the same argument, and that was the one
   place the argument failed (PLAN.md 2b). `actuator.py`'s Coulomb term is
   `(tau_c + mu_load*|tau_t|) * tanh(w/v_eps)`, which is exactly zero at rest, so
   a standing or stancing robot here felt NO joint friction at all while the real
   servo breaks away at 0.18-0.35 N*m and the ROS 2 model carries 0.184. Karnopp
   fixes this inside `actuator.simulate()`, which owns every torque as state; it
   cannot be done on this path, where MuJoCo owns the load and the net torque
   does not exist when the law is called. So the floor is MuJoCo's own
   `frictionloss` — a real stick-slip constraint, solved with everything else —
   set to the fitted `tau_c`, and every MuJoCo caller of `actuator.motor_torque`
   passes `tau_c_external=True` so it is applied once. What MuJoCo cannot carry
   is the load-dependent half: `frictionloss` is a constant per joint, so
   `mu_load*|tau_t|` (~0.11 N*m of the ~0.30 at stance) keeps the smooth law and
   keeps lacking stick.
   `tau_c` is therefore randomised exactly where J_m is — `dof_frictionloss` in
   `env/randomize.py`, per environment — and for exactly the same reason: the
   field the physics reads is the field the draw has to touch. Two constants left
   the episode draw by two different routes and arrived at the same place.

3. THE FEET KEEP priority=1.
   The generated model already ships `priority="1"` on the foot geoms: the CAD's
   MJ_FOOT_PRIORITY went in on 2026-09-08 (3d/CLAUDE.md, "The foot's contact
   lives there too"), and check_model.py now reports the applied solref as the
   foot's own 0.008 s rather than the 0.014 s average it used to warn about. So
   this edit is a BELT, not a fix — it re-asserts the priority so a training
   model built from a scene that lost it still gets the foot's own touchdown
   stiffness instead of the solmix mean of two unrelated choices, and
   `foot_priority=False` is there for anyone who wants to measure the difference.

4. impratio GOES TO 10.
   check_model.py's second warning: at impratio=1 friction is no stiffer than the
   normal direction, so feet slip under load more than mu=1.2 suggests, and a
   policy learns to exploit a slip the robot will not have.

5. BACKLASH IS NOT INSTALLED. This is a deliberate omission, not an oversight.
   `actuator.transmitted()` models the gearbox play as a dead-zone spring between
   two inertias, and `rl/CLAUDE.md` lists backlash among the things a training
   model may add. It is left out here because both of its parameters are guesses
   and one of them is stiff: theta_bl = 0.5 deg is vendor-shaped, k_bl = 3000
   N*m/rad is marked "stiff, not identified", and 3000 N*m/rad against J_m =
   0.008 kg*m^2 is a 612 rad/s mode — ten integration steps per period at the
   scene's 1 ms timestep. Training a policy against a stiff spring whose
   stiffness is invented buys nothing and can destabilise the sim. It goes in as
   a randomisation axis after robot/bench/fit_bam.py has measured it; the code
   for it already exists and is already backend-agnostic.

Procedural terrain
------------------
`rl/README.md` prescribes flat ground plus procedural boxes for training, on the
assumption that MJX cannot take the heightfield. Measured on this lock file
(mujoco 3.12.0), that assumption is wrong in a useful direction: the heightfield
IS supported. What MJX refuses is the obstacle course's two `type="cylinder"`
logs against the robot's boxes — `(mjGEOM_CYLINDER, mjGEOM_BOX) collisions not
implemented`. So `--terrain` works here, with the two logs' collisions disabled
through MjSpec, and the logs stay for the vanilla-MuJoCo sim-to-sim pass in
`eval.py`, which is where they were always going to matter.

Boxes are still the default rough ground for training, for a reason the
heightfield cannot meet: the hfield is one shared 8 x 8 m surface with 446k data
points, and randomising it per environment would cost 3.6 GB of VRAM at 2048
environments on a card that has 8. Boxes randomise per environment for the price
of their positions. They are off by default because each one multiplies the
collision pairs MJX allocates for, and VRAM is this box's binding constraint.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import mujoco

import actuator

HERE = os.path.dirname(os.path.abspath(__file__))
DESC = os.path.abspath(os.path.join(HERE, "..", "ros2", "smalldog_description"))
MJCF = os.path.join(DESC, "mujoco")
PARAMS = os.path.join(DESC, "robot_params.json")
DOMAIN_RAND = os.path.join(HERE, "params", "domain_rand.json")
BUS_TIMING = os.path.join(HERE, "params", "bus_timing.json")

# Margin on the derived torque ceiling below. 1.25 rather than a round number
# because it has one job: leave the law's own arithmetic untouched at the top of
# every randomisation range, and still be a wall a NaN cannot walk through.
TORQUE_CEILING_MARGIN = 1.25

BOX_HALF = (0.13, 0.13, 0.06)     # a procedural terrain box, half-extents, m
BOX_PATCH_M = 4.0                 # boxes are scattered in +-this square


def robot_params() -> dict:
    with open(PARAMS) as f:
        return json.load(f)


def bus_timing(path: str = BUS_TIMING) -> dict:
    """What the bus actually costs, as robot/bench/bus_probe.py measured it."""
    with open(path) as f:
        return json.load(f)


def domain_ranges() -> dict:
    """The randomisation ranges, with the bus delay filled in from the bench.

    Every range in params/domain_rand.json is a literal EXCEPT the bus delay,
    which is read out of params/bus_timing.json here so that the measurement is
    the only copy of it. Until 2026-09-11 the JSON carried a guessed 0..20 ms and
    said the timing file "does not exist"; it has existed since 2026-09-07.

    The composite is `sync_read` + `sync_write`: the feedback the policy computes
    a target from is already one SyncRead old when it arrives, and the target is
    one SyncWrite away from the servo acting on it. p50 and p95 of each, so the
    band is [3.25, 8.90] ms of a 20 ms tick rather than the whole tick.
    """
    with open(DOMAIN_RAND) as f:
        r = {k: v for k, v in json.load(f).items() if not k.startswith("_")}
    t = bus_timing()
    d = r["bus"]["delay_s_abs"]
    d["range"] = [(t["sync_read"][q] + t["sync_write"][q]) * 1e-3
                  for q in ("p50_ms", "p95_ms")]
    d["evidence"] = "measured"
    return r


def torque_ceiling(base: actuator.Params | None = None,
                   ranges: dict | None = None,
                   margin: float = TORQUE_CEILING_MARGIN) -> float:
    """The torque ceiling handed to MuJoCo, in N*m.

    Not a servo limit — the servo's own limit emerges from the law (k_u*U falling
    with the pack, back-EMF capping the speed). This is only a guard against a
    NaN driving the solver, and it has to sit above the highest torque the law
    can produce or it stops being a guard and becomes a second, invisible servo
    model.

    Which is what it was. A hardcoded 5.0 was written as "k_u * 12.6 V with a
    margin for the k_u randomisation range" and the margin was never done:
    k_u * 1.35 * 12.6 V = 5.99 N*m, so 13.2 % of the (k_u, u_bat) draws asked for
    more torque than `forcerange` would pass, and the environments that drew a
    strong servo on a fresh pack were quietly given a weaker one. Derived here
    from the same two ranges the draw uses, so it cannot fall behind them again.
    """
    base = base or actuator.load(quiet=True)
    ranges = ranges or domain_ranges()
    peak = (base.k_u * ranges["actuator"]["k_u"]["range"][1]
            * ranges["supply"]["u_bat_abs"]["range"][1])
    return float(np.ceil(peak * margin * 10.0) / 10.0)


# ===================================================================== build
def build_spec(terrain: bool = False, n_boxes: int = 0, p: actuator.Params | None = None,
               impratio: float = 10.0, foot_priority: bool = True,
               mjx_safe: bool = True, frictionloss: bool = True):
    """The training spec. Returns (spec, notes) where notes lists every edit."""
    p = p or actuator.load(quiet=True)
    scene = os.path.join(MJCF, "scene_terrain.xml" if terrain else "scene.xml")
    spec = mujoco.MjSpec.from_file(scene)
    notes = [f"scene {os.path.basename(scene)}"]

    # 1. position -> motor. The law computes the torque; MuJoCo just applies it.
    ceiling = torque_ceiling(p)
    for a in spec.actuators:
        a.set_to_motor()
        a.gear = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.ctrlrange = [-ceiling, ceiling]
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-ceiling, ceiling]
    notes.append(f"{len(spec.actuators)} position actuators -> torque motors, "
                 f"+-{ceiling:g} N*m ceiling (derived: k_u x the top of its own "
                 f"range x a full pack, x{TORQUE_CEILING_MARGIN:g})")

    # 2. armature <- J_m; damping goes to the law; frictionloss is the floor
    #    the law cannot supply at rest (docstring 2, PLAN.md 2b).
    n = 0
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        j.armature = float(p.J_m)
        j.damping = [0.0, 0.0, 0.0]     # MjsJoint.damping is a 3-vector, not a scalar
        # frictionloss=False is a DIAGNOSTIC (train_ppo --no-frictionloss): the
        # Coulomb floor off entirely, on the MuJoCo path and the law's alike,
        # to ask whether the floor is what stops a policy finding a gait.
        # Not a robot that exists; eval.py builds the floor back in.
        j.frictionloss = float(p.tau_c) if frictionloss else 0.0
        n += 1
    notes.append(f"{n} joints: armature <- J_m = {p.J_m:g} kg*m^2, "
                 f"damping -> 0 (actuator.py supplies b_v and k_w), "
                 + (f"frictionloss <- tau_c = {p.tau_c:g} N*m (MuJoCo sticks; the "
                    f"law drops its own tau_c on this path)" if frictionloss else
                    "frictionloss -> 0 (DIAGNOSTIC: no Coulomb floor anywhere)"))

    # 3. the feet win their own contact parameters.
    if foot_priority:
        feet = [g for b in spec.bodies for g in b.geoms if g.name.endswith("_foot")]
        for g in feet:
            g.priority = 1
        notes.append(f"{len(feet)} foot geoms: priority=1, so solref "
                     f"{list(np.round(feet[0].solref, 4)) if feet else '?'} is what runs")

    # 4. friction as stiff as the normal direction.
    spec.option.impratio = impratio
    notes.append(f"impratio {impratio:g}")

    # 5. the two course logs are the only thing MJX cannot collide.
    if terrain and mjx_safe:
        k = 0
        for b in spec.bodies:
            for g in b.geoms:
                if g.type == mujoco.mjtGeom.mjGEOM_CYLINDER and (g.contype or g.conaffinity):
                    g.contype = 0
                    g.conaffinity = 0
                    k += 1
        notes.append(f"{k} course logs: collisions off — (CYLINDER, BOX) is "
                     f"unimplemented in MJX; eval.py meets them in vanilla MuJoCo")
    elif terrain:
        notes.append("course logs left colliding — vanilla MuJoCo only, this "
                     "model will NOT go through mjx.put_model")

    # 6. procedural boxes, parked below the floor until randomisation lifts them.
    if n_boxes:
        w = spec.worldbody
        side = int(np.ceil(np.sqrt(n_boxes)))
        for i in range(n_boxes):
            gx, gy = divmod(i, side)
            g = w.add_geom()
            g.name = f"tbox{i}"
            g.type = mujoco.mjtGeom.mjGEOM_BOX
            g.size = list(BOX_HALF)
            g.pos = [(gx / max(side - 1, 1) - 0.5) * 2 * BOX_PATCH_M,
                     (gy / max(side - 1, 1) - 0.5) * 2 * BOX_PATCH_M,
                     -BOX_HALF[2] - 1.0]          # buried; z is the randomised axis
            g.rgba = [0.45, 0.42, 0.38, 1.0]
            g.condim = 3
            g.contype = 1
            g.conaffinity = 6                    # feet (2) and body collision (4)
            g.friction = [1.0, 0.005, 0.0001]
            g.group = 3
        notes.append(f"{n_boxes} procedural boxes, {2*BOX_HALF[0]*1000:.0f} mm square, "
                     f"buried at z={-BOX_HALF[2]-1.0:.2f} m until randomisation raises them")

    # Not an edit — a reading, and it is in the notes because the observation
    # hangs off it. The `imu` site is generated out of 3d/mini_dog.py's IMU_*
    # block and it MOVES: it was at the base_link origin, then 23.4 mm, and since
    # 2026-09-09 it is on top of the deck. A policy trained at one height saw a
    # different accelerometer signal from one trained at another (rl/CLAUDE.md's
    # re-baseline rule), so the number the run was trained against belongs in the
    # run's own log rather than in a docstring that goes stale in silence.
    imu = spec.site("imu")
    notes.append(f"imu site at {tuple(round(float(v), 4) for v in imu.pos)} m of "
                 f"base_link — read off the model, not assumed; the observation's "
                 f"accelerometer channel is measured HERE")

    return spec, notes


def build(terrain: bool = False, n_boxes: int = 0, **kw):
    """The compiled training model. `mjx_safe=False` keeps the obstacle-course
    logs colliding, which vanilla MuJoCo handles and MJX does not."""
    spec, notes = build_spec(terrain=terrain, n_boxes=n_boxes, **kw)
    return spec.compile(), notes


# =============================================================== the stance
def stance_qpos(m: mujoco.MjModel, P: dict) -> np.ndarray:
    """The CAD stance, as a full qpos. The same pose check_model.py stands in."""
    q = np.zeros(m.nq)
    q[:3] = [0.0, 0.0, P["stance_base_height_m"]]
    q[3] = 1.0
    for name in P["joint_names"]:
        kind = name.split("_")[1]
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        q[m.jnt_qposadr[j]] = 0.0 if kind == "roll" else P["stance_rad"][kind]
    return q


def box_geom_ids(m: mujoco.MjModel, n_boxes: int) -> list[int]:
    """The procedural terrain boxes, found by name.

    NOT `ngeom - n_boxes`. The boxes go on the worldbody and worldbody geoms
    compile FIRST -- geom 0 is the floor, 1..n are the boxes, and the CAD's
    geoms follow. A tail slice lands on the rear-right leg, so randomising it
    teleports the robot's own collision shapes once per episode. Measured on a
    6-box model: ngeom 53, the last six geoms are rr_hip/rr_thigh/rr_shin and
    rr_foot, and a policy asked to walk in that scene falls on every seed.

    env/randomize.py carried the tail slice, with a comment asserting the
    opposite, from the day it was written. It never fired because every run so
    far passed --boxes 0; rl/README.md prescribes boxes, so it was waiting.
    """
    ids = []
    for i in range(n_boxes):
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"tbox{i}")
        if g < 0:
            raise ValueError(f"tbox{i} is not in this model — was it built with "
                             f"n_boxes={n_boxes}?")
        ids.append(int(g))
    return ids


def compile_with_boxes(tops, xy, terrain: bool = False, **kw):
    """A model whose procedural boxes are ALREADY where you want them.

    Use this instead of writing `m.geom_pos[box_ids] = ...` on a compiled model.
    Worldbody geoms are static, and MuJoCo bakes their bounding volumes at
    compile time: move one afterwards and `geom_xpos` follows, so the box draws
    and reads back in the new place, while broadphase keeps testing the old one.
    Nothing errors. The box is simply not there.

    MJX does NOT share the fault -- it rebuilds candidate pairs from the live
    positions -- which is the dangerous part. Measured on this model, a 260 mm
    slab with its top at 60 mm under a dropped robot:

        MJX,  box moved after put_model    base settles 106.2 mm   collides
        CPU,  box moved after compile      base settles  73.5 mm   falls through
        CPU,  box placed before compile    base settles 106.4 mm   collides

    So env/randomize.py's per-environment boxes are real during training and
    would be invisible in eval.py's sim-to-sim pass, which is the one place the
    project treats as the honest engine. A terrain policy would be scored on
    flat ground and the report would say rough.

    `tops` is each box's top height in metres (negative buries it), `xy` an
    (n, 2) array of centres.
    """
    import numpy as _np
    tops = _np.asarray(tops, dtype=float)
    xy = _np.asarray(xy, dtype=float).reshape(-1, 2)
    n = len(tops)
    if len(xy) != n:
        raise ValueError(f"{n} tops but {len(xy)} centres")
    spec, notes = build_spec(terrain=terrain, n_boxes=n, **kw)
    by_name = {g.name: g for g in spec.worldbody.geoms}
    for i in range(n):
        g = by_name[f"tbox{i}"]
        g.pos = [float(xy[i, 0]), float(xy[i, 1]),
                 float(tops[i]) - BOX_HALF[2] if tops[i] >= 0.0
                 else -BOX_HALF[2] - 1.0]
    return spec.compile(), notes


def joint_order(m: mujoco.MjModel, P: dict):
    """qpos/qvel/actuator indices for the twelve joints, in robot_params order.

    Order matters and is not cosmetic: it is the order the observation, the
    action, params/st3215.json's per-servo entries and the robot's bus IDs all
    have to agree on, and robot_params.json is where that order is decided.
    """
    qadr, vadr, act = [], [], []
    for name in P["joint_names"]:
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert j >= 0 and a >= 0, f"{name} missing from the compiled model"
        qadr.append(m.jnt_qposadr[j])
        vadr.append(m.jnt_dofadr[j])
        act.append(a)
    return np.array(qadr), np.array(vadr), np.array(act)


def limits(P: dict, soft: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Per-joint limits in robot_params order.

    `soft=True` is `joint_soft_limits_rad`, and the default is soft on purpose.
    rl/CLAUDE.md: the three ladders mean three different things, and an action
    space clipped to the hard ROM limits is not the same policy as one clipped to
    the soft ones. The hard ones are where the swept-boolean scan says the parts
    collide; the soft ones are where we are willing to live.
    """
    key = "joint_soft_limits_rad" if soft else "joint_limits_rad"
    lim = np.array([P[key][n.split("_")[1]] for n in P["joint_names"]])
    return -lim, lim


# ========================================================== randomisation
N_JOINTS = 12

#: The per-episode draw, declared ONCE: (field, block, range key, per joint).
#:
#: Two RNGs walk this table — `sample_actuator_params` below on numpy, for
#: checks/check_model.py's probe, and `Walk._sample_episode` on jax keys, inside
#: the training step. They used to be two hand-written dicts and they diverged
#: exactly the way two hand-written dicts do: the check probed a draw the
#: training environment never called, so it reported green while the bus delay
#: was dead (every draw truncated to 0 ticks) and `J_m` was being drawn into a
#: field no function on the training path reads.
#:
#: `J_m` and `tau_c` are deliberately NOT here, and for the same reason. J_m is
#: inertia and the physics reads it as `dof_armature`; tau_c is the Coulomb floor
#: and the physics reads it as `dof_frictionloss` — the only thing in this tree
#: that sticks at rest (PLAN.md 2b). Both are MuJoCo model fields, so both are
#: randomised per environment in env/randomize.py, where the physics can see
#: them. See the module docstring, item 2.
#:
#: Removing one of these does NOT leave a hole in the key numbering. This table
#: is walked by POSITION IN ITSELF — `sample_episode` hands `uniform` the index
#: of the field it is drawing — so a field that leaves simply stops being drawn
#: and the ones after it close up. (b7ff02d kept `keys[4]` spare against the day
#: `tau_c` left a hand-numbered `jax.random.split(rng, 13)`; that numbering is
#: gone, so there is no spare key to preserve. Seeds are not comparable across
#: this table changing either way, which the 2026-09-11 re-baseline already says.)
#:
#: A key ending in `_abs` means the range is the value itself in SI; otherwise it
#: multiplies the nominal in params/st3215.json.
EPISODE_DRAW = (
    # field      block       range key       per joint
    ("k_u",      "actuator", "k_u",          True),
    ("k_e",      "actuator", "k_e",          True),
    ("R",        "actuator", "R",            True),
    ("b_v",      "actuator", "b_v",          True),
    ("mu_load",  "actuator", "mu_load",      True),
    ("kp",       "actuator", "kp",           True),
    ("deadband", "actuator", "deadband_abs", True),
    ("punch",    "actuator", "punch_abs",    True),
    ("u_bat",    "supply",   "u_bat_abs",    False),
    ("sag",      "supply",   "sag_ohm_abs",  False),
    ("delay_s",  "bus",      "delay_s_abs",  False),
)

#: What `dof_armature` is scaled by, per environment. The same range the old
#: per-episode `J_m` draw read, applied where the physics can see it.
ARMATURE_RANGE = ("actuator", "J_m")

#: What `dof_frictionloss` is scaled by, per environment — the same arrangement
#: one field over. `tau_c` left this table on 2026-09-11 for the same reason
#: `J_m` did: the physics reads a MuJoCo field, not a Params attribute, and a
#: draw that does not touch that field is a dead axis. See the module docstring,
#: item 2, and env/randomize.py.
FRICTIONLOSS_RANGE = ("actuator", "tau_c")


def sample_episode(uniform, ranges: dict | None = None,
                   base: actuator.Params | None = None) -> dict:
    """The per-episode servo, pack and bus draw, from whatever RNG is handed in.

    `uniform(i, lo, hi, shape)` draws uniforms on [lo, hi) of that shape; `i` is
    the field's index in EPISODE_DRAW, so a keyed RNG can split on it and an
    unkeyed one can ignore it. Everything else — which fields exist, which range
    each reads, which are per joint — is EPISODE_DRAW, which is what keeps the
    numpy caller and the jax caller describing the same draw.

    Per joint where the spread is per-servo (twelve different motors out of one
    bag), per robot where it is not (one pack, one bus).

    Neither `J_m` nor `tau_c` is here, and both absences say the same thing: the
    physics reads `dof_armature` and `dof_frictionloss`, which are MuJoCo model
    fields, so only brax's randomization_fn can move them and their draws live in
    env/randomize.py — same ranges, same evidence, per environment rather than
    per episode.
    """
    ranges = ranges or domain_ranges()
    base = base or actuator.load(quiet=True)
    out = {}
    for i, (field, block, key, per_joint) in enumerate(EPISODE_DRAW):
        lo, hi = ranges[block][key]["range"]
        shape = (N_JOINTS,) if per_joint else ()
        u = uniform(i, lo, hi, shape)
        out[field] = u if key.endswith("_abs") else u * getattr(base, field)
    return out


def sample_actuator_params(rng, n: int, ranges: dict | None = None,
                           base: actuator.Params | None = None) -> dict:
    """`n` per-episode draws on a numpy Generator, as plain (n,) / (n, 12) arrays.

    The numpy half of `sample_episode` above, and the only caller is
    checks/check_model.py's probe — which is the point of it existing: that check
    runs with mujoco and numpy alone, on the robot and on the mac, and must not
    drag jax in to ask whether the training draw is sane.
    """
    def uniform(_i, lo, hi, shape):
        return np.asarray(rng.uniform(lo, hi, (n,) + shape))

    return sample_episode(uniform, ranges, base)


def delay_ticks(latency_s, frac_u, ctrl_hz: float, xp=np):
    """A whole-tick command delay from a latency in seconds. Stochastic rounding.

    The action buffer's quantum is one control tick: the bus delivers this tick's
    target or the last one. The measured latency is a FRACTION of a tick — 0.16
    to 0.44 of 20 ms, params/bus_timing.json — and neither of the two obvious
    roundings survives contact with that. Truncating deletes the axis outright,
    which is what shipped: `uniform(0, 0.020) * 50` is [0, 1.0) and `.astype(int)`
    made every one of 2000 draws a zero. Rounding to nearest deletes it just as
    thoroughly in the other direction, and would put the whole band at 0 anyway.

    So the fractional tick becomes the PROBABILITY of the extra tick: an episode
    whose latency is 0.30 ticks has a 30 % chance of running one tick late for
    all of it. Unbiased in the mean, and it leaves the policy meeting both cases
    often enough to be robust to either. `frac_u` is one uniform on [0, 1).
    """
    t = latency_s * ctrl_hz
    whole = xp.floor(t)
    return whole + (frac_u < (t - whole))


# ======================================================================= cli
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--terrain", action="store_true")
    ap.add_argument("--boxes", type=int, default=0)
    ap.add_argument("--no-mjx", action="store_true", help="skip the MJX check")
    a = ap.parse_args()

    p = actuator.load()                      # the un-quiet load: it warns, loudly
    m, notes = build(terrain=a.terrain, n_boxes=a.boxes, p=p)
    P = robot_params()

    print("\n== edits, all through MjSpec — nothing on disk is touched ==========")
    for n in notes:
        print(f"  {n}")

    print("\n== the compiled training model ====================================")
    print(f"  {m.nq} qpos, {m.nv} dof, {m.nu} actuators, "
          f"{m.ngeom} geoms, mass {m.body_mass.sum():.4f} kg")
    qadr, vadr, act = joint_order(m, P)
    lo, hi = limits(P, soft=True)
    print(f"  joints in robot_params order, action clipped to the SOFT limits: "
          f"roll +-{hi[0]:.2f}, pitch +-{hi[1]:.2f}, knee +-{hi[2]:.2f} rad")
    d = mujoco.MjData(m)
    d.qpos[:] = stance_qpos(m, P)
    mujoco.mj_forward(m, d)
    print(f"  stance holds at z = {d.qpos[2]*1000:.1f} mm with zero torque applied")

    print("\n== what the law will produce, at the ends of the supply range ======")
    # The nominal servo is NOT the case the ceiling has to clear. Every episode
    # draws k_u out of its own range, so the torque the law can ask for is the
    # nominal times the top of that range — which is why this row is here and why
    # the ceiling is derived rather than typed. Checking the nominal alone is how
    # a 5.0 N*m ceiling sat under a 5.99 N*m draw for the whole of step 4.
    ranges = domain_ranges()
    ceiling = torque_ceiling(p, ranges)
    k_u_hi = ranges["actuator"]["k_u"]["range"][1]
    for u in (12.6, 12.0, 9.9):
        tau = p.stall_torque(u)
        w = p.no_load_speed(u)
        worst = tau * k_u_hi
        print(f"  {u:5.1f} V   stall {tau:5.2f} N*m   no-load {w:5.2f} rad/s"
              f"   randomised peak {worst:5.2f} N*m"
              f"   ({'ceiling ok' if worst < ceiling else 'CEILING TOO LOW'})")
    print(f"  the ceiling handed to MuJoCo is {ceiling:g} N*m: k_u x{k_u_hi:g} at "
          f"{ranges['supply']['u_bat_abs']['range'][1]:g} V, x{TORQUE_CEILING_MARGIN:g}")
    print(f"  the joint velocity limit the CAD reports is "
          f"{P['joint_velocity_limit']:.2f} rad/s, MEASURED on the free hub; the "
          f"law's free speed above OVERSHOOTS it, because the real servo stops "
          f"at a firmware plateau the law does not carry (PLAN.md 3c)")

    if not a.no_mjx:
        print("\n== MJX ===========================================================")
        try:
            import mujoco.mjx as mjx
            mjx.put_model(m)
            print("  put_model ok — this model trains in MJX")
        except Exception as e:
            print(f"  put_model REFUSED: {type(e).__name__}: {e}")
            print("  train on flat plus procedural boxes and keep the refused "
                  "scene for eval.py's vanilla-MuJoCo pass")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
