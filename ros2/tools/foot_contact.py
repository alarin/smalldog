#!/usr/bin/env python
"""foot_contact.py — measure the foot's contact with the ground, and try to make it bigger.

    python tools/foot_contact.py                      # every arm, flat, 5 s trot
    python tools/foot_contact.py --rough --seeds 1 2 3 4 5 6
    python tools/foot_contact.py --push               # stand, lean on it, twist it
    python tools/foot_contact.py --variant pad        # one arm only

The printed foot is a @26 TPU dome and the model says so: one `sphere` geom per leg.  A
sphere on a plane is ONE contact point at any attitude, and the foot class ships
`condim="3"` **until 2026-09-08**, which threw away the torsion and roll columns of its own
`friction` - so a planted foot resisted no twist and no roll whatever, and the numbers that
would have resisted them sat in the file unused.  That part is fixed (mini_dog.py's
MJ_FOOT_CONDIM); the one contact point is not, and cannot be while the foot is a sphere.
Measured: 1.00 contact point per loaded foot, 0.0 mm of patch, and the shin 34.6 deg off
vertical while it is carrying load (46.7 deg worst), so the dome touches a third of the
way round its side.

The arms:
  base    as shipped.  The control - it runs beside the others, never from memory.
  site    base with nothing changed but the touch site's radius.  A second control, and
          not a pedantic one: the site is how the walker feels the ground, and every arm
          that moves the contact outward has to move it too (see SITE_R).
  condim3 the fix undone: MuJoCo's default condim 3, which reads only the first friction
          column, and the torsion number the ROS 2 generator used to ship.  `base` carries
          the fix now, so this is the arm that says what it bought.
  pad     truncate the dome to a flat face, raked to the attitude the shin actually holds
          under load.  Geometry, no mechanism.
  tripod  three lobes on a ring inside the same @26 envelope: a support triangle per foot.
  ankle   a flat pad on a passive sprung rocker at the dome's centre, so the face stays
          flat over the whole 25 deg the shin sweeps.  A mechanism, and the price of one.

Every arm keeps the sole in the SAME place (lowest point 13 mm under the ankle, along the
loaded shin's own vertical) and moves no mass - `ankle` takes its rocker's 8 g out of the
shin - so the flat trot's mass cliff, 3d/CLAUDE.md step 6, cannot be what moves a distance.

Read the caveat before reading the numbers: MuJoCo contact is rigid, so a real contact
PATCH - the thing a TPU dome makes by squashing - is not in this model at any foot shape.
What is in it is contact COUNT, torsion, roll and skid.  This test can say whether more
contact points help the walker.  It cannot say what the pressure on the TPU is.

This is a diagnostic: it mutates the *compiled* model through MjSpec and writes nothing
except its own scratch scenes.  Whichever arm wins has to go back into 3d/mini_dog.py and
both exporters to be real.
"""
import os, sys, math, json, argparse, itertools

HERE = os.path.dirname(os.path.abspath(__file__))
WS   = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(WS, "smalldog_walker"))
sys.path.insert(0, HERE)

import numpy as np
import mujoco
from smalldog_walker.gait import TrotGait
import standalone_sim as ss

MJCF   = os.path.join(WS, "smalldog_description", "mujoco")
PARAMS = os.path.join(WS, "smalldog_description", "robot_params.json")
LEGS   = ("fl", "fr", "rl", "rr")

ROUGH = [False]
VARIANTS = ("base", "site", "condim3", "pad", "tripod", "ankle")


# ------------------------------------------------------------------- the loaded attitude
def stance_rake(scene_path):
    """Where 'world up' points, in each shin's own frame, averaged over LOADED stance.

    Not the stand pose: measured 0.4 deg off the shin axis there, and that number is a
    trap.  During the trot the shin is 35 deg off vertical while it is carrying load and
    sweeps ~25 deg across the stance, so a pad raked to the standing pose would stand on
    its edge for the whole stride.  This runs the real gait and averages the shin's
    attitude weighted by the normal force it is actually carrying, which is the only
    direction a flat face can sensibly point.
    """
    m = mujoco.MjModel.from_xml_path(scene_path)
    met, _, _ = trot(m, 5.0)
    return {leg: met.z[leg]["up"] / max(1e-9, np.linalg.norm(met.z[leg]["up"]))
            for leg in LEGS}


def quat_z_to(u):
    """quaternion rotating +z onto the unit vector u"""
    z = np.array([0.0, 0.0, 1.0])
    v, c = np.cross(z, u), float(np.dot(z, u))
    s = np.linalg.norm(v)
    if s < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0]) if c > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    a = math.atan2(s, c)
    v = v / s
    return np.array([math.cos(a/2), *(math.sin(a/2) * v)])


def basis(u):
    """two unit vectors spanning the plane normal to u"""
    t = np.array([1.0, 0.0, 0.0]) if abs(u[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(u, t); e1 /= np.linalg.norm(e1)
    return e1, np.cross(u, e1)


# ------------------------------------------------------------------------ the four arms
PAD_TRUNC = 3.5e-3          # how deep the pad plane cuts into the @26 dome -> @17.6 face
LOBE_R    = 5.0e-3          # tripod lobe radius
LOBE_RING = 8.0e-3          # lobe centres this far off the ankle axis: 2*(8+5) = the @26
ANK_K, ANK_B, ANK_RANGE = 0.08, 0.005, 50.0     # Nm/rad, Nms/rad, deg
ANK_M = 0.008               # kg moved from the shin into the rocker, so the total is equal
# The touch site the gait's contact feedback reads is a SPHERE round the ankle, and it
# counts any contact whose position falls inside it.  A foot that is wider than the dome
# puts its contacts further from the ankle than the shipped 14 mm reaches - and a foot the
# walker cannot feel is not a better foot, it is a blind one.  Getting this wrong is not a
# subtle bias: with the site left at 14 mm the `pad` arm was on its back 0.4 s into the
# terrain trot, and walked 554 mm with the feedback switched off.  So every arm that moves
# the contact out gets the same enlarged site, and `site` is the arm that isolates it: the
# shipped dome, nothing but the bigger site.
SITE_R = 0.017
# Kept for the record; mini_dog.MJ_FOOT_FRICTION is where this number now lives.
# The torsion column is not a free parameter: MuJoCo's torsional friction has units of
# LENGTH, and for a circular patch of radius a it caps the twist torque at ~(2/3)*a*mu*fn.
# The TPU dome carries 25 N mean / 57 N peak per foot, and 95A TPU works at ~4 MPa, so the
# patch is r = 1.4..2.1 mm and the honest coefficient is 0.667*2.0e-3*1.2 = 1.6e-3 m.  The
# 0.02 the model ships is an order of magnitude generous and 0.05 in export_sim.py is 25x -
# neither was ever chosen, they were just never used.
GRIP_FRICTION = (1.2, 0.002, 0.001)

def apply(spec, kind, rake):
    """mutate the compiled-model spec in place.  Returns a one-line description."""
    if kind == "base":
        return "@26 sphere, condim 3 (as shipped)"

    if kind == "site":
        for leg in LEGS:
            spec.site(f"{leg}_foot_site").size = [SITE_R, 0, 0]
        return f"@26 sphere, condim 3, touch site grown to r={SITE_R*1000:.0f} mm (control)"

    if kind == "condim3":
        # The `grip` arm, run backwards.  It went in on 2026-09-08 (mini_dog.py's
        # MJ_FOOT_CONDIM / MJ_FOOT_FRICTION), so `base` above IS that fix now and the arm
        # that has to be measured beside it is the one that undoes it: MuJoCo's default
        # condim 3, which reads only the first friction column, and the torsion number the
        # ROS 2 generator used to ship.  Keeping it is not nostalgia - it is the only way
        # the before/after is measured on ONE tree instead of against a remembered number.
        for leg in LEGS:
            g = spec.geom(f"{leg}_foot")
            g.condim = 3
            g.friction = [1.2, 0.02, 0.001]
        return "@26 sphere, condim 3 + the old torsion (the fix, undone)"

    if kind == "ankle":
        return _ankle(spec)

    for leg in LEGS:
        g = spec.geom(f"{leg}_foot")
        r = float(g.size[0])
        c = np.array(g.pos)
        u = rake[leg]                       # world-up, in shin coordinates
        site = spec.site(f"{leg}_foot_site")

        if kind == "pad":
            # The dome truncated by a plane raked to the stance shin: a real flat face,
            # @17.6, whose lowest point is exactly where the sphere's was.  MuJoCo gives a
            # cylinder on a plane up to four contacts, so the patch is a patch.
            rp = math.sqrt(r*r - (r - PAD_TRUNC)**2)
            h  = 5.0e-3                     # half-height; the top is buried in the ankle
            g.type = mujoco.mjtGeom.mjGEOM_CYLINDER
            g.size = [rp, h, 0.0]
            g.pos  = c - (r - h) * u
            g.quat = quat_z_to(u)
            g.condim = 3
            site.size = [SITE_R, 0, 0]      # site.pos stays at the ankle
        elif kind == "tripod":
            # Three lobes on the same raked plane, on a @16 ring inside the same @26
            # envelope.  Three contact points per foot = a support triangle per foot, and
            # a twist about the ankle is resisted by geometry, not by a friction column.
            e1, e2 = basis(u)
            g.type = mujoco.mjtGeom.mjGEOM_SPHERE
            g.size = [LOBE_R, 0.0, 0.0]
            base = c - (r - LOBE_R) * u
            g.pos = base + LOBE_RING * e1
            for k in (1, 2):
                a = 2.0 * math.pi * k / 3.0
                nb = spec.body(f"{leg}_shin").add_geom()
                nb.name = f"{leg}_foot_{k}"
                nb.classname = g.classname
                nb.type = mujoco.mjtGeom.mjGEOM_SPHERE
                nb.size = [LOBE_R, 0.0, 0.0]
                nb.pos = base + LOBE_RING * (math.cos(a) * e1 + math.sin(a) * e2)
                nb.condim = 3
            g.condim = 3
            site.size = [SITE_R, 0, 0]      # site.pos stays at the ankle
        else:
            raise SystemExit(f"unknown variant {kind}")

    return {"pad":    f"dome truncated {PAD_TRUNC*1000:.1f} mm to a raked @"
                      f"{2000*math.sqrt(13e-3**2 - (13e-3-PAD_TRUNC)**2):.1f} flat pad",
            "tripod": f"three @{2000*LOBE_R:.0f} lobes on a @{2000*LOBE_RING:.0f} ring,"
                      f" raked; same @26 envelope"}[kind]


def _ankle(spec):
    """A passive sprung rocker inside the foot, carrying the flat pad.

    The `pad` arm above has to pick ONE rake and live with it, because the shin sweeps
    ~25 deg across a stance.  This one does not pick: the pad hangs on a hinge at the
    dome's own centre, 13 mm above the sole, with a weak spring to neutral, so it lies
    flat whatever the shin is doing.  It costs a moving part and a DOF that nothing
    controls - the honest price of a flat foot on a rigid two-link leg.
    """
    for leg in LEGS:
        g = spec.geom(f"{leg}_foot")
        r = float(g.size[0])
        c = np.array(g.pos)
        rp = math.sqrt(r*r - (r - PAD_TRUNC)**2)
        h  = 4.0e-3
        shin = spec.body(f"{leg}_shin")
        shin.mass = shin.mass - ANK_M            # the total mass must not move: the flat
        g.contype = g.conaffinity = 0            # trot sits on a cliff in it
        g.group = 4

        b = shin.add_body()
        b.name = f"{leg}_ankle"
        b.pos = c
        b.explicitinertial = True
        b.mass = ANK_M
        b.ipos = [0.0, 0.0, -(r - h)]
        b.inertia = [2e-7, 2e-7, 3e-7]
        j = b.add_joint()
        j.name = f"{leg}_ankle_pitch"
        j.type = mujoco.mjtJoint.mjJNT_HINGE
        j.axis = [0.0, 1.0, 0.0]
        j.range = [-math.radians(ANK_RANGE), math.radians(ANK_RANGE)]
        j.limited = 1
        # stiffness/damping/armature/frictionloss are set on the COMPILED model in
        # build() - the joint default class here is the servo's (armature 0.008,
        # frictionloss 0.02), which on a foot rocker is a welded ankle, not a spring.
        pad = b.add_geom()
        pad.name = f"{leg}_foot_pad"
        pad.classname = g.classname
        pad.type = mujoco.mjtGeom.mjGEOM_CYLINDER
        pad.size = [rp, h, 0.0]
        pad.pos = [0.0, 0.0, -(r - h)]
        pad.mass = 0.0
        pad.condim = 3
        spec.site(f"{leg}_foot_site").size = [SITE_R, 0, 0]
    return (f"@{2*rp*1000:.1f} flat pad on a passive rocker at the dome centre,"
            f" k={ANK_K} Nm/rad, +-{ANK_RANGE:.0f} deg")


def build(scene_path, kind, rake):
    spec = mujoco.MjSpec.from_file(scene_path)
    if os.path.basename(scene_path).startswith("_footcontact_"):
        # the scratch scene has to be written next to scene.xml (its <include>s and
        # robot.xml's meshdir are both relative to the model directory), so it goes as
        # soon as it has been read.  Nothing of this tool's is left in the package.
        os.remove(scene_path)
    note = apply(spec, kind, rake)
    model = spec.compile()
    if kind == "ankle":
        for leg in LEGS:
            j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_ankle_pitch")
            dof = model.jnt_dofadr[j]
            model.jnt_stiffness[j] = ANK_K
            model.dof_damping[dof] = ANK_B
            model.dof_armature[dof] = 1e-6
            model.dof_frictionloss[dof] = 0.0
    return model, note


# ---------------------------------------------------------------------------- the meter
class Meter:
    """Per-step contact bookkeeping for the four feet.

    Everything here is measured on the foot that is CARRYING LOAD - a contact under 0.5 N
    is the swing foot brushing the ground and tells you nothing about the patch.
    """
    LOAD = 0.5          # N

    def __init__(self, model):
        self.m = model
        nm = lambda g: mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        self.foot = {g: nm(g)[:2] for g in range(model.ngeom)
                     if nm(g)[:2] in LEGS and "foot" in nm(g)}
        self.z = {k: dict(steps=0, pts=0.0, spread=0.0, slip=0.0, spin=0.0,
                          sliding=0, fn=0.0, fnmax=0.0, tilt=0.0, tiltmax=0.0,
                          up=np.zeros(3))
                  for k in LEGS}
        self.n = 0

    def step(self, data):
        self.n += 1
        m, d = self.m, data
        per = {k: [] for k in LEGS}
        f6 = np.zeros(6)
        for i in range(d.ncon):
            c = d.contact[i]
            leg = self.foot.get(c.geom1) or self.foot.get(c.geom2)
            if leg is None:
                continue
            gid = c.geom1 if c.geom1 in self.foot else c.geom2
            mujoco.mj_contactForce(m, d, i, f6)
            if f6[0] < self.LOAD:
                continue
            mu = c.friction[0]
            per[leg].append((np.array(c.pos), float(f6[0]),
                             math.hypot(f6[1], f6[2]) > 0.98 * mu * f6[0], gid,
                             np.array(c.frame).reshape(3, 3)))
        dt = m.opt.timestep
        v6 = np.zeros(6)
        for leg, cs in per.items():
            if not cs:
                continue
            z = self.z[leg]
            z["steps"] += 1
            z["pts"] += len(cs)
            fn = sum(c[1] for c in cs)
            z["fn"] += fn
            z["fnmax"] = max(z["fnmax"], fn)
            z["sliding"] += 1 if any(c[2] for c in cs) else 0
            if len(cs) > 1:
                z["spread"] += max(np.linalg.norm(a[0] - b[0])
                                   for a, b in itertools.combinations(cs, 2))
            # skid: the load-weighted tangential speed of the foot material at the
            # contact, integrated.  The ground is static, so this is the whole slip.
            skid, w = 0.0, 0.0
            for p, f, _, gid, frame in cs:
                mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, gid, v6, 0)
                vel = v6[3:6] + np.cross(v6[0:3], p - d.geom_xpos[gid])
                n = frame[0]
                skid += f * np.linalg.norm(vel - np.dot(vel, n) * n)
                w += f
            z["slip"] += skid / w * dt
            gid = cs[0][3]
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_GEOM, gid, v6, 0)
            z["spin"] += abs(v6[2]) * dt
            # how far off vertical the shin is WHILE LOADED.  A flat pad is only flat if
            # this stays small; it is what decides whether the pad idea can work at all.
            R = d.body(f"{leg}_shin").xmat.reshape(3, 3)
            t = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
            z["tilt"] += t
            z["tiltmax"] = max(z["tiltmax"], t)
            z["up"] += fn * (R.T @ np.array([0.0, 0.0, 1.0]))

    def report(self, pad="    "):
        tot = {k: sum(self.z[l][k] for l in LEGS) for k in
               ("steps", "pts", "spread", "slip", "spin", "sliding", "tilt")}
        st = max(1, tot["steps"])
        out = dict(
            stance=100.0 * tot["steps"] / max(1, 4 * self.n),
            points=tot["pts"] / st,
            spread=1000.0 * tot["spread"] / st,
            slip=1000.0 * tot["slip"] / 4.0,
            spin=math.degrees(tot["spin"]) / 4.0,
            sliding=100.0 * tot["sliding"] / st,
            fnmax=max(self.z[l]["fnmax"] for l in LEGS),
            tilt=tot["tilt"] / st, tiltmax=max(self.z[l]["tiltmax"] for l in LEGS))
        print(f"{pad}contact  {out['points']:.2f} points/loaded foot,"
              f"  patch spread {out['spread']:5.1f} mm,"
              f"  peak normal {out['fnmax']:5.1f} N")
        print(f"{pad}         skid {out['slip']:6.1f} mm/foot,"
              f"  twist {out['spin']:6.1f} deg/foot,"
              f"  at the friction limit {out['sliding']:4.1f} % of stance,"
              f"  duty {out['stance']:.0f} %")
        print(f"{pad}         shin off vertical while loaded:"
              f" {out['tilt']:.1f} deg mean, {out['tiltmax']:.1f} deg worst")
        return out


# ------------------------------------------------------------------------------ the run
def trot(model, seconds=5.0, blind=False):
    d = mujoco.MjData(model)
    gait = TrotGait(json.load(open(PARAMS)))
    act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in gait.joint_names]
    sens = ss.sensors(model)
    mujoco.mj_resetData(model, d)
    ss.settle(model, d, gait, act, 1.5, sens, blind)
    ss.run(model, d, gait, act, (0.0, 0.0, 0.0), 1.0, sens, blind)
    s0 = ss.state(d)
    met = Meter(model)
    n = int(seconds / model.opt.timestep)
    for _ in range(n):
        q = gait.joint_targets(model.opt.timestep, 0.20, 0.0, 0.0)
        for i, a in enumerate(act):
            d.ctrl[a] = q[i]
        mujoco.mj_step(model, d)
        ss.feed(gait, d, sens, blind)
        met.step(d)
    s1 = ss.state(d)
    return met, s0, s1


# ---------------------------------------------------------------- ground you can trust
# The heightfield is not usable for this question, and that is worth stating precisely
# rather than working around quietly.  terrain.py's field is CELL_MM = 12 mm and MuJoCo
# collides a heightfield as prisms at that pitch.  A @26 sphere spans two cells and
# behaves.  A @17.7 flat face or a @10 lobe is SMALLER than a cell, so what it collides
# with is the prisms' side walls: the `pad` arm was on its back before the trot started
# and the `tripod` arm stood still with 4 kN summed through its touch sensors - forty
# times the robot's weight, at a stand.  Re-generating the same noise at 4 mm cells makes
# it worse, not better (46 contacts per foot, 6.7 kN, and the CONTROL arm down from 657 to
# 363 mm), because the cost is the prism count, not the cell size.  So: on a heightfield
# this model can compare gaits and masses, but it cannot compare feet.
#
# --rough is the rough ground that can.  A plane, which every collider handles exactly,
# strewn with seeded tilted slabs - convex boxes, no prisms - across the metre the 5 s
# trot actually covers.  A foot meets edges, corners and slopes, and every contact it
# makes is one the collider is good at.
ROUGH_N, ROUGH_T, ROUGH_TILT = 90, 0.006, 9.0     # slabs, thickness m, max tilt deg

def rough_scene(seed):
    rng = np.random.default_rng(1234 if seed is None else seed)
    g = []
    for i in range(ROUGH_N):
        x = float(rng.uniform(0.10, 1.30))
        y = float(rng.uniform(-0.32, 0.32))
        sx, sy = rng.uniform(0.020, 0.045, 2)
        yaw = float(rng.uniform(0, math.pi))
        ax = rng.normal(size=2)
        ang = math.radians(float(rng.uniform(0.0, ROUGH_TILT)))
        # yaw about z, then a small tilt about a random horizontal axis
        qz = np.array([math.cos(yaw/2), 0, 0, math.sin(yaw/2)])
        u = np.array([ax[0], ax[1], 0.0]); u /= np.linalg.norm(u)
        qt = np.array([math.cos(ang/2), *(math.sin(ang/2)*u)])
        q = np.zeros(4); mujoco.mju_mulQuat(q, qt, qz)
        g.append(f'    <geom name="rough{i}" type="box" pos="{x:.4f} {y:.4f} 0.0"'
                 f' size="{sx:.4f} {sy:.4f} {ROUGH_T:.4f}"'
                 f' quat="{q[0]:.6f} {q[1]:.6f} {q[2]:.6f} {q[3]:.6f}"'
                 f' rgba="0.45 0.43 0.40 1" condim="3" contype="1" conaffinity="15"'
                 f' friction="1.0 0.005 0.0001"/>')
    tmp = os.path.join(MJCF, f"_footcontact_rough{seed or 0}.xml")
    x = open(os.path.join(MJCF, "scene.xml")).read()
    x = x.replace("</worldbody>", "\n".join(g) + "\n  </worldbody>")
    open(tmp, "w").write(x)
    return tmp


# ------------------------------------------------------------------------ the push test
# The trot is the wrong place to look for a contact patch and the numbers below say so:
# a walking foot is loaded for half a cycle and then picked up, so what it is standing on
# barely gets to matter.  A STANDING robot is where the patch earns its keep - it is the
# only thing resisting a twist about the ankle, and with a sphere at condim 3 there is
# nothing there at all.  So: stand still, lean on it, and see how far it goes.
PUSH_F, PUSH_TQ, PUSH_T = 6.0, 0.6, 1.0        # N sideways, Nm about z, seconds

def push(model, yaw=False):
    """Stand, then push (or twist) the body for PUSH_T s.  Returns how far it gave."""
    d = mujoco.MjData(model)
    gait = TrotGait(json.load(open(PARAMS)))
    act = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in gait.joint_names]
    sens = ss.sensors(model)
    mujoco.mj_resetData(model, d)
    ss.settle(model, d, gait, act, 2.0, sens)
    b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    s0 = ss.state(d)
    h0 = math.atan2(2*(d.qpos[3]*d.qpos[6] + d.qpos[4]*d.qpos[5]),
                    1 - 2*(d.qpos[5]**2 + d.qpos[6]**2))
    q = gait.joint_targets(0.0, 0.0, 0.0, 0.0)
    for i, a in enumerate(act):
        d.ctrl[a] = q[i]
    for _ in range(int(PUSH_T / model.opt.timestep)):
        d.xfrc_applied[b] = ([0, 0, 0, 0, 0, PUSH_TQ] if yaw else [0, PUSH_F, 0, 0, 0, 0])
        mujoco.mj_step(model, d)
        ss.feed(gait, d, sens)
    d.xfrc_applied[b] = np.zeros(6)
    s1 = ss.state(d)
    h1 = math.atan2(2*(d.qpos[3]*d.qpos[6] + d.qpos[4]*d.qpos[5]),
                    1 - 2*(d.qpos[5]**2 + d.qpos[6]**2))
    return dict(dy=(s1["y"] - s0["y"]) * 1000.0,
                dyaw=math.degrees(h1 - h0),
                roll=s1["roll"] - s0["roll"])


def scene_for(terrain, seed):
    if ROUGH[0]:
        return rough_scene(seed)
    if not terrain:
        return os.path.join(MJCF, "scene.xml")
    if seed is None:
        return os.path.join(MJCF, "scene_terrain.xml")
    p = os.path.join(MJCF, f"sweep_terrain_s{seed}.xml")
    if not os.path.exists(p):
        raise SystemExit(f"no {p} — write the seed sweep scenes first")
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", action="append", choices=VARIANTS,
                    help="default: all four, base first (it is the control)")
    ap.add_argument("--terrain", action="store_true")
    ap.add_argument("--seeds", type=int, nargs="*", default=[None])
    ap.add_argument("--blind", action="store_true")
    ap.add_argument("--ankle-k", type=float, help="override the rocker spring, Nm/rad")
    ap.add_argument("--push", action="store_true",
                    help=f"stand still, then lean {PUSH_F:.0f} N sideways and twist"
                         f" {PUSH_TQ} Nm about z for {PUSH_T:.0f} s, and report how far"
                         " the body gave")
    ap.add_argument("--rough", action="store_true",
                    help="a plane strewn with seeded tilted slabs — the rough ground that"
                         " can actually compare feet (see rough_scene)")
    a = ap.parse_args()
    global ANK_K
    ROUGH[0] = a.rough
    if a.ankle_k is not None:
        ANK_K = a.ankle_k
    arms = a.variant or list(VARIANTS)
    seeds = a.seeds if (a.terrain or a.rough) else [None]
    rake = stance_rake(os.path.join(MJCF, "scene.xml"))
    r0 = rake["fl"]
    print(f"stance rake: world-up sits {math.degrees(math.acos(min(1.0, r0[2]))):.1f} deg"
          f" off the shin's own axis — that is where the dome touches")
    if a.push:
        print(f"\npush: {PUSH_F:.0f} N sideways / twist: {PUSH_TQ} Nm about z,"
              f" {PUSH_T:.0f} s, from a stand")
        print(f"{'arm':8} {'side dy mm':>11} {'roll deg':>9} {'yaw deg':>9}")
        for kind in arms:
            model, note = build(scene_for(False, None), kind, rake)
            a1, a2 = push(model), push(model, yaw=True)
            print(f"{kind:8} {a1['dy']:11.1f} {a1['roll']:9.2f} {a2['dyaw']:9.2f}")
        return
    summary = {}
    for kind in arms:
        dists = []
        print(f"\n== {kind}")
        for seed in seeds:
            model, note = build(scene_for(a.terrain, seed), kind, rake)
            if seed == seeds[0]:
                print(f"   {note};  mass {sum(model.body_mass):.4f} kg")
            met, s0, s1 = trot(model, blind=a.blind)
            dx = (s1["x"] - s0["x"]) * 1000.0
            dists.append(dx)
            tag = f"seed {seed}" if seed is not None else "flat "
            print(f"   {tag}  travelled {dx:7.1f} mm  y={(s1['y']-s0['y'])*1000:+7.1f}"
                  f"  z={s1['z']*1000:5.1f}  roll={s1['roll']:+5.1f} pitch={s1['pitch']:+5.1f}")
            rep = met.report()
        summary[kind] = (float(np.mean(dists)), float(np.std(dists)), rep)
    print("\n" + "-" * 78)
    print(f"{'arm':8} {'travel mm':>16} {'pts/foot':>9} {'spread':>8} {'skid':>8}"
          f" {'twist':>8} {'@limit':>7}")
    for k, (mu, sd, r) in summary.items():
        t = f"{mu:7.1f}" + (f" +-{sd:.0f}" if len(seeds) > 1 else "       ")
        print(f"{k:8} {t:>16} {r['points']:9.2f} {r['spread']:7.1f} {r['slip']:7.1f}"
              f" {r['spin']:7.1f} {r['sliding']:6.1f}%")


if __name__ == "__main__":
    main()
