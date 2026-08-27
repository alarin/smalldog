#!/usr/bin/env python
"""
export_sim.py - ROS 2 (URDF) and MuJoCo (MJCF) export of the mini_dog model.

Nothing here is a second model: every length, mass and limit is read out of
mini_dog.py (and its ROM scan), the same way fea.py does it.  Re-run this after any
change to mini_dog.py so the simulation model never drifts from the printed one.

This is NOT the only exporter of this CAD.  ../ros2/smalldog_description/scripts/
generate_model.py builds the ROS 2 package's own description from the same mini_dog.py,
with a different link decomposition (foot merged into the shin) and its own collision
primitives.  A model change is not finished until BOTH have been re-run.  Neither may
keep its own copy of a mass or a density - they all come from mini_dog.py section 4.

    .venv/bin/python export_sim.py            # -> out/sim/{mini_dog.urdf,mini_dog.xml,meshes/}
    .venv/bin/python export_sim.py --rom-step 2   # re-scan the joint limits finely
    .venv/bin/python export_sim.py --mesh-uri relative   # non-ROS mesh paths

Conventions of the exported model
  * SI: metres, kilograms, radians.  The STL meshes stay in mm and are scaled by 0.001.
  * Frames match mini_dog's robot frame: +X forward, +Y left, +Z up, origin at the
    chassis centre in the hip-roll plane.  Zero pose = all legs straight down.
  * Joint axes are the PHYSICAL ones, identical for every leg: hip_roll about +X,
    hip_pitch / knee about +Y.  The mirrored legs therefore carry mirrored limits
    rather than mirrored axes - a positive hip_pitch swings all four legs the same
    way in world terms.
  * Visual = the real printed solids.  Collision = primitives (box / capsule /
    sphere), because a mesh collider of a lightened bracket is both slow and, in
    MuJoCo, silently replaced by its convex hull.
"""
import os, math, json, argparse

import cadquery as cq
from OCP.BRepGProp import BRepGProp
from OCP.GProp import GProp_GProps

import mini_dog as md

OUT    = os.path.join(md.OUT, "sim")
MESHES = os.path.join(OUT, "meshes")

RHO_PRINT = md.PRINT_RHO * 1e-6        # g/cm3 -> kg/mm3
TPU_RHO   = md.TPU_RHO   * 1e-6        # ditto; both live in mini_dog's mass block

# collision primitives - deliberately a little proud of the real part
R_THIGH, R_SHIN = 0.014, 0.011         # capsule radii, m
IMU_XYZ = (0.0, 0.0, md.BODY_Z1 / 1000.0)

# MuJoCo joint feel.  Not measured - the ST3215 gearbox is a black box; these are
# plausible values that keep the model stable at 2 ms.  Tune against hardware.
MJ_DAMPING, MJ_ARMATURE, MJ_FRICTIONLOSS, MJ_KP = 0.5, 0.01, 0.05, 20.0


# =====================================================================================
# mass properties
# =====================================================================================
class MP:
    """mass [kg], centre of mass [mm], inertia about the com [kg*mm^2]"""
    def __init__(self, m=0.0, c=(0.0, 0.0, 0.0), I=((0,0,0),(0,0,0),(0,0,0))):
        self.m, self.c = m, tuple(c)
        self.I = tuple(tuple(float(v) for v in row) for row in I)

    @staticmethod
    def of(shape, rho):
        """rho in kg/mm3; `shape` is a Workplane or a Shape in robot coordinates"""
        s = shape.val() if isinstance(shape, cq.Workplane) else shape
        p = GProp_GProps()
        BRepGProp.VolumeProperties_s(s.wrapped, p)
        com = p.CentreOfMass()
        m = p.MatrixOfInertia()                       # about the com, in mm^5
        # a mirrored solid can come back with reversed orientation, i.e. a negative
        # volume.  It is the same body either way - take the magnitude, don't debug it.
        sg = -1.0 if p.Mass() < 0.0 else 1.0
        return MP(p.Mass() * rho * sg, (com.X(), com.Y(), com.Z()),
                  [[m.Value(i, j) * rho * sg for j in (1, 2, 3)] for i in (1, 2, 3)])

    def __add__(self, o):
        if o.m == 0.0: return self
        if self.m == 0.0: return o
        m = self.m + o.m
        c = tuple((self.m * a + o.m * b) / m for a, b in zip(self.c, o.c))
        I = [[0.0] * 3 for _ in range(3)]
        for src in (self, o):                          # parallel axis onto the new com
            d = [a - b for a, b in zip(src.c, c)]
            d2 = sum(v * v for v in d)
            for i in range(3):
                for j in range(3):
                    I[i][j] += src.I[i][j] + src.m * (d2 * (i == j) - d[i] * d[j])
        return MP(m, c, I)

    def moved_to(self, origin):
        """same body, expressed in a frame whose origin is at `origin` (mm, no rotation)"""
        return MP(self.m, tuple(a - b for a, b in zip(self.c, origin)), self.I)

    @property
    def com_m(self):  return tuple(v / 1000.0 for v in self.c)

    @property
    def inertia(self):
        """ixx iyy izz ixy ixz iyz in kg*m^2"""
        I = self.I
        return tuple(v * 1e-6 for v in (I[0][0], I[1][1], I[2][2], I[0][1], I[0][2], I[1][2]))


def box_mp(mass, size, centre):
    sx, sy, sz = size
    return MP(mass, centre, [[mass * (sy*sy + sz*sz) / 12.0, 0, 0],
                             [0, mass * (sx*sx + sz*sz) / 12.0, 0],
                             [0, 0, mass * (sx*sx + sy*sy) / 12.0]])


def servo_mp(L, loc):
    """the real ST3215 solid at joint frame `loc`, mirrored onto leg L, at catalogue mass"""
    s = L["xf"](md.mv(md.servo_dummy(), loc))
    return MP.of(s, md.SERVO_KG / s.val().Volume())


# =====================================================================================
# leg layout
# =====================================================================================
# mesh set A is the as-modelled front-left leg; B is its mirror across XZ.  The rear
# legs are the front ones turned 180 deg about Z (mirX*mirY), so they reuse the meshes
# with rz = pi.  Axis signs: a mirror reverses rotation about every axis except its own
# normal; the 180 deg turn reverses both X and Y.
LEGS = {
    "FL": dict(sx=+1, sy=+1, mesh="A", rz=0.0,      xf=lambda w: w,
               roll=+1, pitch=+1),
    "FR": dict(sx=+1, sy=-1, mesh="B", rz=0.0,      xf=md.mirY,
               roll=-1, pitch=+1),
    "RL": dict(sx=-1, sy=+1, mesh="B", rz=math.pi,  xf=md.mirX,
               roll=+1, pitch=-1),
    "RR": dict(sx=-1, sy=-1, mesh="A", rz=math.pi,  xf=lambda w: md.mirX(md.mirY(w)),
               roll=-1, pitch=-1),
}


def leg_frames(L):
    sx, sy = L["sx"], L["sy"]
    return dict(roll=(sx*md.ROLL_X,  sy*md.ROLL_Y, md.ROLL_Z),
                pitch=(sx*md.PITCH_X, sy*md.LEG_Y,  md.PITCH_Z),
                knee=(sx*md.PITCH_X, sy*md.LEG_Y,  md.KNEE_Z),
                foot=(sx*md.PITCH_X, sy*md.LEG_Y,  md.FOOT_Z))


def limits(rom_deg, sign):
    lo, hi = rom_deg
    a, b = sorted((sign * math.radians(lo), sign * math.radians(hi)))
    return a, b


# =====================================================================================
# ROM
# =====================================================================================
# the window mini_dog.main() sweeps for each joint.  A free range that reaches its window
# is not a mechanical limit, it is the end of the scan - say so rather than exporting it
# as if the geometry had stopped the joint.
SCAN_WINDOW = {"hip_roll": 90, "hip_pitch": 150, "knee": 150}


def joint_rom(step):
    """{joint: (lo_deg, hi_deg)} in mini_dog's own (front-left) convention"""
    if step is None:
        bom = os.path.join(md.OUT, "bom.json")
        if os.path.exists(bom):
            with open(bom) as f:
                rom = json.load(f).get("rom_deg")
            if rom:
                print(f"  joint limits from {bom} (coarse 10 deg scan)")
                return {k: tuple(v) for k, v in rom.items()}
        step = 10
    print(f"  scanning joint limits at {step} deg - this is the slow part")
    return {
        "hip_roll":  md.rom_scan(md.hip_bracket(),
                                 md.chassis_bottom().union(md.mv(md.servo_dummy(), md.ROLL_LOC)),
                                 (md.ROLL_X, md.ROLL_Y, md.ROLL_Z), axis=(1, 0, 0),
                                 lo=-SCAN_WINDOW["hip_roll"], hi=SCAN_WINDOW["hip_roll"],
                                 step=step),
        "hip_pitch": md.rom_scan(md.thigh(),
                                 md.hip_bracket().union(md.mv(md.servo_dummy(), md.PITCH_LOC)),
                                 (md.PITCH_X, md.LEG_Y, md.PITCH_Z), step=step,
                                 lo=-SCAN_WINDOW["hip_pitch"], hi=SCAN_WINDOW["hip_pitch"]),
        "knee":      md.rom_scan(md.shin(),
                                 md.thigh().union(md.mv(md.servo_dummy(), md.KNEE_LOC)),
                                 (md.PITCH_X, md.LEG_Y, md.KNEE_Z), step=step,
                           lo=-SCAN_WINDOW["knee"], hi=SCAN_WINDOW["knee"]),
    }


# =====================================================================================
# meshes
# =====================================================================================
def write_meshes(solids):
    """link-local STLs, still in mm.  Returns {mesh_name: file}."""
    os.makedirs(MESHES, exist_ok=True)
    files = {}
    for name, (wp, origin) in solids.items():
        w = wp.translate(tuple(-v for v in origin)) if any(origin) else wp
        f = f"{name}.stl"
        cq.exporters.export(w, os.path.join(MESHES, f), tolerance=0.03, angularTolerance=0.2)
        files[name] = f
    return files


def sim_solids():
    """{mesh_name: (solid in its own link frame's translation, link origin)} + per-leg mass props"""
    hb, th, sh, ft = md.hip_bracket(), md.thigh(), md.shin(), md.foot()
    fl = leg_frames(LEGS["FL"])
    fr = leg_frames(LEGS["FR"])
    solids = {
        "chassis_bottom": (md.chassis_bottom(), (0, 0, 0)),
        "chassis_top":    (md.chassis_top(),    (0, 0, 0)),
        "lidar_mount":    (md.lidar_mount(),    (0, 0, 0)),
        "hip_bracket_A":  (hb,            fl["roll"]),
        "thigh_A":        (th,            fl["pitch"]),
        "shin_A":         (sh.union(ft),  fl["knee"]),
        "hip_bracket_B":  (md.mirY(hb),   fr["roll"]),
        "thigh_B":        (md.mirY(th),   fr["pitch"]),
        "shin_B":         (md.mirY(sh).union(md.mirY(ft)), fr["knee"]),
    }
    return solids, (hb, th, sh, ft)


def link_masses(parts):
    """{leg: {link: MP in that link's frame}} plus the base MP, all in link-local mm"""
    hb, th, sh, ft = parts
    legs = {}
    for tag, L in LEGS.items():
        f, xf = leg_frames(L), L["xf"]
        hip = MP.of(xf(hb), RHO_PRINT) + servo_mp(L, md.PITCH_LOC)
        thi = MP.of(xf(th), RHO_PRINT) + servo_mp(L, md.KNEE_LOC)
        shn = MP.of(xf(sh), RHO_PRINT) + MP.of(xf(ft), TPU_RHO)
        legs[tag] = dict(hip=hip.moved_to(f["roll"]),
                         thigh=thi.moved_to(f["pitch"]),
                         shin=shn.moved_to(f["knee"]))
    base = (MP.of(md.chassis_bottom(), RHO_PRINT)
            + MP.of(md.chassis_top(), RHO_PRINT)
            + MP.of(md.lidar_mount(), RHO_PRINT)
            + box_mp(md.BATTERY_KG, (md.BATT_L, md.BATT_W, md.BATT_H),
                     (0.0, 0.0, md.BODY_Z0 + 3.0 + md.BATT_H / 2.0))
            + box_mp(md.ELECTRONICS_KG, (92.0, 62.0, 20.0),
                     (-22.0, 0.0, md.BODY_Z1 + md.DECK_T + 10.0))
            + box_mp(md.LIDAR_KG, (70.0, 70.0, 60.0),     # the L2 itself, on its pedestal
                     (42.0, 0.0, md.BODY_Z1 + md.DECK_T + md.LIDAR_H + 30.0)))
    for L in LEGS.values():                          # the four hip-roll servos ride the chassis
        base = base + servo_mp(L, md.ROLL_LOC)
    return base, legs


# =====================================================================================
# URDF
# =====================================================================================
def fmt(v):  return " ".join(f"{x:.6g}" for x in v)


def urdf_inertial(mp):
    ixx, iyy, izz, ixy, ixz, iyz = mp.inertia
    return (f'      <origin xyz="{fmt(mp.com_m)}" rpy="0 0 0"/>\n'
            f'      <mass value="{mp.m:.6g}"/>\n'
            f'      <inertia ixx="{ixx:.6g}" iyy="{iyy:.6g}" izz="{izz:.6g}"'
            f' ixy="{ixy:.6g}" ixz="{ixz:.6g}" iyz="{iyz:.6g}"/>\n')


def urdf(base_mp, legmp, rom, meshes, mesh_uri):
    def mesh(name, rz=0.0, material="printed"):
        return (f'    <visual>\n'
                f'      <origin xyz="0 0 0" rpy="0 0 {rz:.6g}"/>\n'
                f'      <geometry><mesh filename="{mesh_uri(meshes[name])}"'
                f' scale="0.001 0.001 0.001"/></geometry>\n'
                f'      <material name="{material}"/>\n'
                f'    </visual>\n')

    x = ['<?xml version="1.0"?>',
         '<!-- generated by export_sim.py from mini_dog.py - do not edit -->',
         '<robot name="mini_dog">',
         '  <material name="printed"><color rgba="0.78 0.79 0.82 1"/></material>',
         '  <material name="hip"><color rgba="0.85 0.55 0.15 1"/></material>',
         '  <material name="rubber"><color rgba="0.12 0.12 0.14 1"/></material>',
         '  <link name="base_link">',
         '    <inertial>', urdf_inertial(base_mp).rstrip('\n'), '    </inertial>']
    for m in ("chassis_bottom", "chassis_top", "lidar_mount"):
        x.append(mesh(m).rstrip('\n'))
    x += [f'    <collision>\n'
          f'      <origin xyz="0 0 {(md.BODY_Z0 + md.BODY_Z1) / 2000.0:.6g}"/>\n'
          f'      <geometry><box size="{md.BODY_L/1000.0:.6g} {md.BODY_W/1000.0:.6g}'
          f' {(md.BODY_Z1 - md.BODY_Z0)/1000.0:.6g}"/></geometry>\n'
          f'    </collision>',
          '  </link>',
          '  <link name="imu_link"/>',
          f'  <joint name="imu_joint" type="fixed">\n'
          f'    <parent link="base_link"/><child link="imu_link"/>\n'
          f'    <origin xyz="{fmt(IMU_XYZ)}" rpy="0 0 0"/>\n'
          f'  </joint>']

    for tag, L in LEGS.items():
        f, rz = leg_frames(L), L["rz"]
        mp = legmp[tag]
        chain = [
            ("hip",   "base_link",        f["roll"],  (0, 0, 0),  (1, 0, 0),
             limits(rom["hip_roll"],  L["roll"]),  f"hip_bracket_{L['mesh']}", "hip"),
            ("thigh", f"hip_{tag}",       f["pitch"], f["roll"],  (0, 1, 0),
             limits(rom["hip_pitch"], L["pitch"]), f"thigh_{L['mesh']}",       "printed"),
            ("shin",  f"thigh_{tag}",     f["knee"],  f["pitch"], (0, 1, 0),
             limits(rom["knee"],      L["pitch"]), f"shin_{L['mesh']}",        "printed"),
        ]
        for link, parent, origin, poff, axis, (lo, hi), meshname, mat in chain:
            jname = {"hip": "hip_roll", "thigh": "hip_pitch", "shin": "knee"}[link]
            x += [f'  <joint name="{jname}_{tag}" type="revolute">',
                  f'    <parent link="{parent}"/><child link="{link}_{tag}"/>',
                  f'    <origin xyz="{fmt([(a - b) / 1000.0 for a, b in zip(origin, poff)])}"'
                  f' rpy="0 0 0"/>',
                  f'    <axis xyz="{fmt(axis)}"/>',
                  f'    <limit lower="{lo:.6g}" upper="{hi:.6g}"'
                  f' effort="{md.SERVO_STALL_NM:.6g}" velocity="{md.SERVO_NOLOAD_RADS:.6g}"/>',
                  f'    <dynamics damping="{MJ_DAMPING}" friction="{MJ_FRICTIONLOSS}"/>',
                  '  </joint>',
                  f'  <link name="{link}_{tag}">',
                  '    <inertial>', urdf_inertial(mp[link]).rstrip('\n'), '    </inertial>',
                  mesh(meshname, rz, mat).rstrip('\n')]
            x.append(_urdf_collision(link, f))
            x.append('  </link>')
    x.append('</robot>')
    return "\n".join(x) + "\n"


def _urdf_collision(link, f):
    if link == "hip":
        c = ((f["pitch"][0] - f["roll"][0]) / 2000.0,
             (f["pitch"][1] - f["roll"][1]) / 2000.0,
             (f["pitch"][2] - f["roll"][2]) / 2000.0)
        return (f'    <collision><origin xyz="{fmt(c)}"/>'
                f'<geometry><box size="0.036 {abs(f["pitch"][1] - f["roll"][1])/1000.0:.6g}'
                f' 0.05"/></geometry></collision>')
    if link == "thigh":
        h = (md.PITCH_Z - md.KNEE_Z) / 1000.0
        return (f'    <collision><origin xyz="0 0 {-h/2:.6g}"/>'
                f'<geometry><cylinder radius="{R_THIGH}" length="{h:.6g}"/></geometry></collision>')
    h = (md.KNEE_Z - md.FOOT_Z) / 1000.0
    return (f'    <collision><origin xyz="0 0 {-h/2:.6g}"/>'
            f'<geometry><cylinder radius="{R_SHIN}" length="{h:.6g}"/></geometry></collision>\n'
            f'    <collision><origin xyz="0 0 {-h:.6g}"/>'
            f'<geometry><sphere radius="{md.FOOT_D/2000.0:.6g}"/></geometry></collision>')


# =====================================================================================
# MJCF
# =====================================================================================
def stand_height():
    """base height that puts the four feet on z = 0 in the stand pose, m"""
    p, k = math.radians(md.STAND_PITCH), math.radians(md.STAND_PITCH + md.STAND_KNEE)
    z = md.PITCH_Z - md.L_THIGH * math.cos(p) - md.L_SHIN * math.cos(k) - md.FOOT_D / 2.0
    return -z / 1000.0


def mjcf(base_mp, legmp, rom, meshes):
    def inertial(mp):
        ixx, iyy, izz, ixy, ixz, iyz = mp.inertia
        return (f'<inertial pos="{fmt(mp.com_m)}" mass="{mp.m:.6g}"'
                f' fullinertia="{ixx:.6g} {iyy:.6g} {izz:.6g}'
                f' {ixy:.6g} {ixz:.6g} {iyz:.6g}"/>')

    def vis(name, rz):
        q = ' quat="0 0 0 1"' if abs(rz) > 1e-9 else ''
        return f'<geom class="viz" mesh="{name}"{q}/>'

    x = ['<!-- generated by export_sim.py from mini_dog.py - do not edit -->',
         '<mujoco model="mini_dog">',
         '  <compiler angle="radian" meshdir="meshes" autolimits="true"/>',
         '  <option timestep="0.002" integrator="implicitfast" cone="elliptic"/>',
         '  <visual>',
         '    <global offwidth="1920" offheight="1080"/>',
         '    <headlight ambient="0.35 0.35 0.35" diffuse="0.6 0.6 0.6"/>',
         '    <quality shadowsize="4096"/>',
         '  </visual>',
         '  <default>',
         '    <geom solref="0.005 1" friction="0.9 0.02 0.001"/>',
         f'    <joint damping="{MJ_DAMPING}" armature="{MJ_ARMATURE}"'
         f' frictionloss="{MJ_FRICTIONLOSS}"/>',
         f'    <position kp="{MJ_KP}" forcerange="{-md.SERVO_STALL_NM:.6g}'
         f' {md.SERVO_STALL_NM:.6g}"/>',
         '    <default class="viz">',
         '      <geom type="mesh" contype="0" conaffinity="0" group="2"'
         ' rgba="0.78 0.79 0.82 1"/>',
         '    </default>',
         '    <default class="col">',
         '      <geom group="3" rgba="0.9 0.3 0.2 0.35"/>',
         '    </default>',
         '  </default>',
         '  <asset>',
         '    <texture name="grid" type="2d" builtin="checker" rgb1="0.2 0.22 0.25"'
         ' rgb2="0.28 0.3 0.33" width="512" height="512"/>',
         '    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.1"/>']
    for name, f in meshes.items():
        x.append(f'    <mesh name="{name}" file="{f}" scale="0.001 0.001 0.001"/>')
    x += ['  </asset>',
          '  <worldbody>',
          '    <light pos="0 0 2.5" dir="0 0 -1" directional="true"/>',
          '    <geom name="floor" type="plane" size="5 5 0.05" material="grid"'
          ' friction="1.0 0.02 0.001"/>',
          f'    <body name="base_link" pos="0 0 {stand_height():.6g}">',
          '      <freejoint name="root"/>',
          f'      {inertial(base_mp)}',
          f'      <site name="imu" pos="{fmt(IMU_XYZ)}" size="0.005"/>']
    for m in ("chassis_bottom", "chassis_top", "lidar_mount"):
        x.append(f'      {vis(m, 0.0)}')
    x.append(f'      <geom class="col" type="box" pos="0 0'
             f' {(md.BODY_Z0 + md.BODY_Z1) / 2000.0:.6g}"'
             f' size="{md.BODY_L/2000.0:.6g} {md.BODY_W/2000.0:.6g}'
             f' {(md.BODY_Z1 - md.BODY_Z0)/2000.0:.6g}"/>')

    acts = []
    for tag, L in LEGS.items():
        f, rz, mp = leg_frames(L), L["rz"], legmp[tag]
        rl, rh = limits(rom["hip_roll"],  L["roll"])
        pl, ph = limits(rom["hip_pitch"], L["pitch"])
        kl, kh = limits(rom["knee"],      L["pitch"])
        d = lambda a, b: fmt([(p - q) / 1000.0 for p, q in zip(a, b)])
        th_h, sh_h = (md.PITCH_Z - md.KNEE_Z) / 1000.0, (md.KNEE_Z - md.FOOT_Z) / 1000.0
        x += [f'      <body name="hip_{tag}" pos="{d(f["roll"], (0,0,0))}">',
              f'        <joint name="hip_roll_{tag}" axis="1 0 0"'
              f' range="{rl:.6g} {rh:.6g}"/>',
              f'        {inertial(mp["hip"])}',
              f'        {vis("hip_bracket_" + L["mesh"], rz)}',
              f'        <geom class="col" type="capsule" size="0.018"'
              f' fromto="0 0 0 {d(f["pitch"], f["roll"])}"/>',
              f'        <body name="thigh_{tag}" pos="{d(f["pitch"], f["roll"])}">',
              f'          <joint name="hip_pitch_{tag}" axis="0 1 0"'
              f' range="{pl:.6g} {ph:.6g}"/>',
              f'          {inertial(mp["thigh"])}',
              f'          {vis("thigh_" + L["mesh"], rz)}',
              f'          <geom class="col" type="capsule" size="{R_THIGH}"'
              f' fromto="0 0 0 0 0 {-th_h:.6g}"/>',
              f'          <body name="shin_{tag}" pos="0 0 {-th_h:.6g}">',
              f'            <joint name="knee_{tag}" axis="0 1 0"'
              f' range="{kl:.6g} {kh:.6g}"/>',
              f'            {inertial(mp["shin"])}',
              f'            {vis("shin_" + L["mesh"], rz)}',
              f'            <geom class="col" type="capsule" size="{R_SHIN}"'
              f' fromto="0 0 0 0 0 {-sh_h:.6g}"/>',
              f'            <geom name="foot_{tag}" class="col" type="sphere"'
              f' size="{md.FOOT_D/2000.0:.6g}" pos="0 0 {-sh_h:.6g}"'
              f' friction="1.2 0.05 0.001" rgba="0.12 0.12 0.14 1" group="0"/>',
              f'            <site name="foot_{tag}" pos="0 0 {-sh_h:.6g}" size="0.004"/>',
              '          </body>',
              '        </body>',
              '      </body>']
        for j in (f"hip_roll_{tag}", f"hip_pitch_{tag}", f"knee_{tag}"):
            acts.append(f'    <position name="{j}" joint="{j}"/>')
    x += ['    </body>', '  </worldbody>', '  <actuator>'] + acts + ['  </actuator>']

    # stand pose keyframe: mini_dog's own STAND_PITCH / STAND_KNEE, per-leg sign applied
    ctrl = [v for L in LEGS.values()
            for v in (0.0, L["pitch"] * math.radians(md.STAND_PITCH),
                      L["pitch"] * math.radians(md.STAND_KNEE))]
    q = [0.0, 0.0, stand_height(), 1.0, 0.0, 0.0, 0.0] + ctrl
    x += ['  <keyframe>',
          f'    <key name="stand" qpos="{fmt(q)}" ctrl="{fmt(ctrl)}"/>',
          '  </keyframe>',
          '  <sensor>',
          '    <framequat name="imu_quat" objtype="site" objname="imu"/>',
          '    <gyro name="imu_gyro" site="imu"/>',
          '    <accelerometer name="imu_acc" site="imu"/>',
          '  </sensor>',
          '</mujoco>']
    return "\n".join(x) + "\n"


# =====================================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rom-step", type=int, default=None,
                    help="re-run the ROM scan at this step (deg) instead of reading out/bom.json")
    ap.add_argument("--mesh-uri", choices=("package", "relative"), default="package")
    ap.add_argument("--package", default="mini_dog_description")
    ap.add_argument("--check", action="store_true",
                    help="load both files in MuJoCo and drop the robot on the floor")
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print("\n  building solids ...")
    solids, parts = sim_solids()
    meshes = write_meshes(solids)
    base_mp, legmp = link_masses(parts)
    rom = joint_rom(a.rom_step)

    uri = ((lambda f: f"package://{a.package}/meshes/{f}") if a.mesh_uri == "package"
           else (lambda f: f"meshes/{f}"))
    with open(os.path.join(OUT, "mini_dog.urdf"), "w") as f:
        f.write(urdf(base_mp, legmp, rom, meshes, uri))
    with open(os.path.join(OUT, "mini_dog.xml"), "w") as f:
        f.write(mjcf(base_mp, legmp, rom, meshes))

    total = base_mp.m + sum(mp.m for l in legmp.values() for mp in l.values())
    print(f"\n  link masses (kg)   base {base_mp.m:.3f}"
          f"   hip {legmp['FL']['hip'].m:.3f}"
          f"   thigh {legmp['FL']['thigh'].m:.3f}"
          f"   shin+foot {legmp['FL']['shin'].m:.3f}")
    print(f"  total {total:.3f} kg   stand height {stand_height()*1000:.0f} mm")
    print("  joint range (deg, physical axes, front-left):")
    for j, sign in (("hip_roll", LEGS["FL"]["roll"]), ("hip_pitch", LEGS["FL"]["pitch"]),
                    ("knee", LEGS["FL"]["pitch"])):
        lo, hi = limits(rom[j], sign)
        w = SCAN_WINDOW[j]
        sat = "   (the scan window, not an interference limit)" \
            if min(abs(v) for v in rom[j]) >= w else ""
        print(f"    {j:10s} {math.degrees(lo):+7.1f} .. {math.degrees(hi):+7.1f}{sat}")
    print(f"\n  -> {os.path.relpath(OUT)}/mini_dog.urdf, mini_dog.xml, meshes/ "
          f"({len(meshes)} STL)")

    try:
        import fea
        ref = fea.robot_mass()
        # fea's estimate prints everything at PETG density and counts the servo_gauge
        # test print, so it sits a few grams high; a big gap means a real disagreement.
        flag = "" if abs(ref - total) < 0.05 else "   !! disagrees with fea.robot_mass()"
        print(f"  cross-check: fea.robot_mass() = {ref:.3f} kg (PETG everywhere,"
              f" incl. servo_gauge){flag}")
    except Exception as e:
        print(f"  cross-check against fea.robot_mass() skipped ({e})")

    if a.check:
        check(uri)


def check(uri):
    """load what we just wrote and actually stand on it"""
    try:
        import mujoco
    except ImportError:
        print("\n  --check needs mujoco in the venv")
        return
    print("\n  MuJoCo check")
    m = mujoco.MjModel.from_xml_path(os.path.join(OUT, "mini_dog.xml"))
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[:] = m.key_ctrl[0]
    for _ in range(1500):
        mujoco.mj_step(m, d)
    feet = sum(1 for c in d.contact[:d.ncon]
               if "foot" in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or "")
               + (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""))
    up = 1.0 - 2.0 * (d.qpos[4] ** 2 + d.qpos[5] ** 2)          # world z of the body z axis
    ok = feet == 4 and up > 0.9 and d.qpos[2] > 0.05
    print(f"    mjcf   nq {m.nq} nu {m.nu}, mass {m.body_subtreemass[1]:.3f} kg;"
          f" after 3 s in the stand pose: base z {d.qpos[2]*1000:.0f} mm,"
          f" {feet} feet down, upright {up:+.2f}"
          f"{'' if ok else '   !! it fell over'}")
    # the URDF has to survive a parser too; MuJoCo reads URDF, given relative mesh paths
    import re
    tmp = os.path.join(OUT, "_check.urdf")
    txt = re.sub(r'filename="[^"]*/meshes/', 'filename="meshes/',
                 open(os.path.join(OUT, "mini_dog.urdf")).read())
    with open(tmp, "w") as f:
        f.write(txt)
    try:
        u = mujoco.MjModel.from_xml_path(tmp)
        # a URDF root has no free joint, so MuJoCo welds base_link to the world and
        # its mass lands in body 0 - sum the bodies instead of taking a subtree.
        um = float(sum(u.body_mass))
        moving = m.body_subtreemass[1] - m.body_mass[1]      # the mjcf's 12 leg links
        dm = abs(um - moving)
        print(f"    urdf   parsed: {u.njnt} revolute joints, {u.nbody - 1} moving links,"
              f" leg mass {um:.3f} kg vs mjcf {moving:.3f} kg"
              f"{'' if dm < 1e-3 else '   !! the two files disagree'}")
    except Exception as e:
        print(f"    urdf   !! failed to parse: {e}")
    finally:
        os.remove(tmp)


if __name__ == "__main__":
    main()
