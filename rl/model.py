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

2. ARMATURE BECOMES THE REFLECTED ROTOR INERTIA, DAMPING AND FRICTIONLOSS GO TO ZERO.
   Not a tightening — a move. `armature`, `damping` and `frictionloss` in the
   MJCF are three of the four guesses check_model.py flags. Once `actuator.py`
   supplies Coulomb friction (tau_c), viscous friction (b_v) and back-EMF damping
   (k_w = k_u*k_e), leaving MuJoCo's own damping and frictionloss in place counts
   the same physics twice. The reflected inertia J_m has to stay in MuJoCo — it
   is inertia, it belongs in the mass matrix, and at 1:345 it is ~73x the knee
   link's own, which check_model.py measures and calls the dominant term.

3. THE FEET GET priority=1.
   check_model.py's first warning: the foot declares solref 0.008 s, the floor
   declares 0.02 s, geom priority is equal, so MuJoCo averages them and the
   touchdown stiffness that actually runs is 0.014 s — the mean of two unrelated
   choices, chosen by neither. Priority makes the foot's own number win. The
   proper fix is a deliberate contact pair in the CAD's exporter; until it is
   there, training against an accident is worse than training against a choice.

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

# The torque ceiling handed to MuJoCo. Not a servo limit — the servo's own limit
# emerges from the law (k_u*U falling with the pack, back-EMF capping the speed).
# This is only a guard against a NaN driving the solver, set above the highest
# torque the law can produce: k_u * 12.6 V at the top of a fresh 3S pack, with a
# margin for the k_u randomisation range.
TORQUE_CEILING_NM = 5.0

BOX_HALF = (0.13, 0.13, 0.06)     # a procedural terrain box, half-extents, m
BOX_PATCH_M = 4.0                 # boxes are scattered in +-this square


def robot_params() -> dict:
    with open(PARAMS) as f:
        return json.load(f)


def domain_ranges() -> dict:
    with open(DOMAIN_RAND) as f:
        return {k: v for k, v in json.load(f).items() if not k.startswith("_")}


# ===================================================================== build
def build_spec(terrain: bool = False, n_boxes: int = 0, p: actuator.Params | None = None,
               impratio: float = 10.0, foot_priority: bool = True,
               mjx_safe: bool = True):
    """The training spec. Returns (spec, notes) where notes lists every edit."""
    p = p or actuator.load(quiet=True)
    scene = os.path.join(MJCF, "scene_terrain.xml" if terrain else "scene.xml")
    spec = mujoco.MjSpec.from_file(scene)
    notes = [f"scene {os.path.basename(scene)}"]

    # 1. position -> motor. The law computes the torque; MuJoCo just applies it.
    for a in spec.actuators:
        a.set_to_motor()
        a.gear = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.ctrlrange = [-TORQUE_CEILING_NM, TORQUE_CEILING_NM]
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-TORQUE_CEILING_NM, TORQUE_CEILING_NM]
    notes.append(f"{len(spec.actuators)} position actuators -> torque motors, "
                 f"+-{TORQUE_CEILING_NM:g} N*m ceiling")

    # 2. armature <- J_m; damping and frictionloss go to the law.
    n = 0
    for j in spec.joints:
        if j.type == mujoco.mjtJoint.mjJNT_FREE:
            continue
        j.armature = float(p.J_m)
        j.damping = [0.0, 0.0, 0.0]     # MjsJoint.damping is a 3-vector, not a scalar
        j.frictionloss = 0.0
        n += 1
    notes.append(f"{n} joints: armature <- J_m = {p.J_m:g} kg*m^2, "
                 f"damping and frictionloss -> 0 (actuator.py supplies b_v, tau_c, k_w)")

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
def sample_actuator_params(rng, n: int, ranges: dict | None = None,
                           base: actuator.Params | None = None) -> dict:
    """Per-environment servo, supply and bus draws, as plain arrays.

    Returns a dict of (n,) or (n, 12) arrays rather than an actuator.Params,
    because these are sampled per EPISODE and live in the env state, while the
    things brax's randomization_fn can touch live in the model and are fixed for
    the whole run. Both feed the same equations in actuator.py — there is still
    exactly one copy of the law.

    Per-joint where the spread is per-servo (twelve different motors out of one
    bag), per-environment where it is not (one pack, one bus).
    """
    ranges = ranges or domain_ranges()
    base = base or actuator.load(quiet=True)
    A, S, B = ranges["actuator"], ranges["supply"], ranges["bus"]

    def mul(key, shape):
        lo, hi = A[key]["range"]
        return np.asarray(rng.uniform(lo, hi, shape)) * getattr(base, key)

    def absolute(d, key, shape):
        lo, hi = d[key]["range"]
        return np.asarray(rng.uniform(lo, hi, shape))

    return dict(
        # per servo
        k_u=mul("k_u", (n, 12)), k_e=mul("k_e", (n, 12)), R=mul("R", (n, 12)),
        J_m=mul("J_m", (n, 12)), tau_c=mul("tau_c", (n, 12)), b_v=mul("b_v", (n, 12)),
        kp=mul("kp", (n, 12)),
        deadband=absolute(A, "deadband_abs", (n, 12)),
        punch=absolute(A, "punch_abs", (n, 12)),
        # per robot
        u_bat=absolute(S, "u_bat_abs", (n,)),
        sag=absolute(S, "sag_ohm_abs", (n,)),
        delay_s=absolute(B, "delay_s_abs", (n,)),
    )


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
    for u in (12.6, 12.0, 9.9):
        tau = p.stall_torque(u)
        w = p.no_load_speed(u)
        print(f"  {u:5.1f} V   stall {tau:5.2f} N*m   no-load {w:5.2f} rad/s"
              f"   ({'ceiling ok' if tau < TORQUE_CEILING_NM else 'CEILING TOO LOW'})")
    print(f"  the joint velocity limit the CAD reports is "
          f"{P['joint_velocity_limit']:.2f} rad/s — the law reaches it only "
          f"unloaded and on a full pack, which is the point of modelling it")

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
