#!/usr/bin/env python
"""
generate_model.py — build the SmallDog robot description straight from the CAD.

Reads ../../../3d/mini_dog.py (the single source of truth for the mechanics),
exports one STL per link in that link's own frame, computes real mass properties
from the tessellated solids, and writes:

    meshes/*.stl        link meshes, millimetres
    urdf/smalldog.urdf  URDF for robot_state_publisher / ros2_control
    mujoco/robot.xml    MJCF body tree + position actuators
    mujoco/scene.xml    world + ground plane
    mujoco/defaults.xml joint / actuator / geom defaults

Run it again after changing anything in mini_dog.py.
"""
import os, sys, math, json
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG  = os.path.dirname(HERE)
CAD  = os.path.abspath(os.path.join(PKG, "..", "..", "3d"))
sys.path.insert(0, CAD)

import cadquery as cq
import mini_dog as md

MESHES = os.path.join(PKG, "meshes")
URDF   = os.path.join(PKG, "urdf")
MJCF   = os.path.join(PKG, "mujoco")
for d in (MESHES, URDF, MJCF):
    os.makedirs(d, exist_ok=True)

MM = 1e-3

# ---------------------------------------------------------------- mass model
# Every number here comes from the CAD's own mass block (mini_dog.py section 4).  Do not
# reintroduce local copies: this file and 3d/export_sim.py are two exporters of the same
# robot, and the last time they each kept their own servo mass they disagreed by 60 g.
RHO_PRINT = md.PRINT_RHO * 1e-3   # g/cm^3 -> g/mm^3 (tri_inertia wants g/mm^3)
RHO_TPU   = md.TPU_RHO   * 1e-3
M_SERVO   = md.SERVO_KG           # kg
M_BATTERY = md.BATTERY_KG
M_ELECTR  = md.ELECTRONICS_KG
M_LIDAR   = md.LIDAR_KG

def tri_inertia(verts, tris, density_g_mm3):
    """Exact mass properties of a closed triangle mesh (mm, g/mm^3).
    Returns mass [kg], com [mm], inertia about the com [kg*m^2]."""
    v = np.asarray(verts, dtype=float)
    t = np.asarray(tris, dtype=int)
    a, b, c = v[t[:, 0]], v[t[:, 1]], v[t[:, 2]]
    # signed tetra volumes with the origin
    vol6 = np.einsum("ij,ij->i", a, np.cross(b, c))
    vol = vol6.sum() / 6.0
    if vol < 0:                       # flip if the winding is inward
        a, b, c, vol6, vol = a, c, b, -vol6, -vol
    com = ((a + b + c) * vol6[:, None]).sum(axis=0) / (4.0 * vol6.sum())
    # covariance of a tetra (0,a,b,c) — canonical formula
    C = np.zeros((3, 3))
    Ccan = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]]) / 120.0
    for A, B, Cc, d6 in zip(a, b, c, vol6):
        M = np.stack([A, B, Cc])
        C += d6 * (M.T @ Ccan @ M)
    mass_g = vol * density_g_mm3
    # covariance about the origin, normalised, then shifted to the com
    C = C / vol                                     # mm^2 * mm^3 / mm^3
    C = C - np.outer(com, com)
    tr = np.trace(C)
    I = (np.eye(3) * tr - C) * mass_g               # g*mm^2
    return mass_g * 1e-3, com, I * 1e-9             # kg, mm, kg*m^2

def box_inertia(mass, dx, dy, dz):
    """dx,dy,dz in mm -> kg*m^2"""
    k = mass / 12.0 * 1e-6
    return np.diag([k * (dy * dy + dz * dz), k * (dx * dx + dz * dz), k * (dx * dx + dy * dy)])

class Body:
    """accumulates (mass, com, inertia) contributions in one link frame (mm)."""
    def __init__(self):
        self.m = 0.0
        self.mc = np.zeros(3)
        self.parts = []            # (mass, com_mm, I_com)
    def add(self, mass, com_mm, I_com):
        self.parts.append((mass, np.asarray(com_mm, dtype=float), I_com))
        self.m += mass
        self.mc += mass * np.asarray(com_mm, dtype=float)
    def add_solid(self, wp, rho):
        v, t = wp.val().tessellate(0.5)
        m, c, I = tri_inertia([(p.x, p.y, p.z) for p in v], t, rho)
        self.add(m, c, I)
    def add_box(self, mass, com_mm, dims_mm):
        self.add(mass, com_mm, box_inertia(mass, *dims_mm))
    @property
    def com(self):
        return self.mc / self.m if self.m > 0 else np.zeros(3)
    @property
    def inertia(self):
        """about the body com, kg*m^2"""
        I = np.zeros((3, 3))
        com = self.com
        for m, c, Ic in self.parts:
            d = (np.asarray(c) - com) * MM
            I += Ic + m * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
        return I

# ---------------------------------------------------------------- geometry
R = md                                   # shorthand for the CAD parameters
LEGS = ["fl", "fr", "rl", "rr"]
SX   = {"fl": +1, "fr": +1, "rl": -1, "rr": -1}
SY   = {"fl": +1, "fr": -1, "rl": +1, "rr": -1}

def leg_transform(leg, wp):
    """FL part -> the given leg, using the same mirrors the CAD uses."""
    if SY[leg] < 0: wp = md.mirY(wp)
    if SX[leg] < 0: wp = md.mirX(wp)
    return wp

def origin(leg, which):
    sx, sy = SX[leg], SY[leg]
    if which == "hip":   return np.array([sx * R.ROLL_X,  sy * R.ROLL_Y, R.ROLL_Z])
    if which == "thigh": return np.array([sx * R.PITCH_X, sy * R.LEG_Y,  R.PITCH_Z])
    if which == "shin":  return np.array([sx * R.PITCH_X, sy * R.LEG_Y,  R.KNEE_Z])
    raise KeyError(which)

def export_stl(wp, offset_mm, name):
    w = wp.translate((-offset_mm[0], -offset_mm[1], -offset_mm[2]))
    cq.exporters.export(w, os.path.join(MESHES, name + ".stl"),
                        tolerance=0.05, angularTolerance=0.2)
    return w

# ---------------------------------------------------------------- build links
print("building CAD parts ...")
hb, th, sh, ft = md.build()
chassis = (md.PARTS["chassis_bottom"][0]
           .union(md.PARTS["chassis_top"][0])
           .union(md.PARTS["lidar_mount"][0]))

links = {}          # name -> dict(mesh, body, extras)
SERVO_VIS = {}      # link name -> [servo mesh names]  (visual only, mass already
                    # counted in the link Body as a uniform box)

# ---- base_link -------------------------------------------------------------
print("base_link ...")
base = Body()
base.add_solid(chassis, RHO_PRINT)
for leg in LEGS:                                   # the four hip-roll servos
    o = origin(leg, "hip")
    base.add_box(M_SERVO, o + np.array([0, SY[leg] * (R.S_L / 2 - R.S_AX), 0]),
                 (R.S_H, R.S_L, R.S_W))
base.add_box(M_BATTERY, (0, 0, R.BODY_Z0 + 3 + R.BATT_H / 2), (R.BATT_L, R.BATT_W, R.BATT_H))
base.add_box(M_ELECTR,  (-22.0, 0, R.BODY_Z1 + 12.0), (100.0, 62.0, 18.0))
base.add_box(M_LIDAR,   (42.0, 0, R.BODY_Z1 + R.DECK_T + R.LIDAR_H + 30.0), (70.0, 70.0, 60.0))
export_stl(chassis, np.zeros(3), "base_link")
SERVO_VIS["base_link"] = []
for leg in LEGS:                                   # the four hip-roll servos, as they sit
    nm = f"{leg}_roll_servo"
    export_stl(leg_transform(leg, md.mv(md.servo_dummy(), md.ROLL_LOC)), np.zeros(3), nm)
    SERVO_VIS["base_link"].append(nm)
links["base_link"] = dict(mesh="base_link", body=base)

# ---- legs ------------------------------------------------------------------
for leg in LEGS:
    print(f"{leg} leg ...")
    hip_w   = leg_transform(leg, hb)
    thigh_w = leg_transform(leg, th)
    shin_w  = leg_transform(leg, sh).union(leg_transform(leg, ft))

    o_hip, o_thigh, o_shin = (origin(leg, k) for k in ("hip", "thigh", "shin"))

    b = Body(); b.add_solid(export_stl(hip_w, o_hip, f"{leg}_hip"), RHO_PRINT)
    b.add_box(M_SERVO, np.array([0, 0, R.PITCH_Z - R.ROLL_Z + R.S_L / 2 - R.S_AX]),
              (R.S_W, R.S_H, R.S_L))       # hip-pitch servo, body points up
    export_stl(leg_transform(leg, md.mv(md.servo_dummy(), md.PITCH_LOC)), o_hip,
               f"{leg}_pitch_servo")
    SERVO_VIS[f"{leg}_hip"] = [f"{leg}_pitch_servo"]
    links[f"{leg}_hip"] = dict(mesh=f"{leg}_hip", body=b)

    b = Body(); b.add_solid(export_stl(thigh_w, o_thigh, f"{leg}_thigh"), RHO_PRINT)
    b.add_box(M_SERVO, np.array([0, 0, -R.L_THIGH + R.S_L / 2 - R.S_AX]),
              (R.S_W, R.S_H, R.S_L))       # knee servo, body points up the thigh
    export_stl(leg_transform(leg, md.mv(md.servo_dummy(), md.KNEE_LOC)), o_thigh,
               f"{leg}_knee_servo")
    SERVO_VIS[f"{leg}_thigh"] = [f"{leg}_knee_servo"]
    links[f"{leg}_thigh"] = dict(mesh=f"{leg}_thigh", body=b)

    b = Body(); b.add_solid(export_stl(shin_w, o_shin, f"{leg}_shin"), RHO_PRINT)
    # shin and foot ship as one mesh but not as one material: correct the foot's share
    # by the density difference.  add_solid is linear in rho, so this is exact.
    b.add_solid(leg_transform(leg, ft).translate(tuple(-o_shin)), RHO_TPU - RHO_PRINT)
    links[f"{leg}_shin"] = dict(mesh=f"{leg}_shin", body=b)

total = sum(l["body"].m for l in links.values())
print(f"\ntotal robot mass: {total:.3f} kg")
for k, v in links.items():
    c = v["body"].com
    print(f"  {k:10s} {v['body'].m*1000:7.1f} g   com {c[0]:7.1f} {c[1]:7.1f} {c[2]:7.1f} mm")

# ---------------------------------------------------------------- joint table
J_LIM  = {"roll": 0.90, "pitch": 1.30, "knee": 1.85}     # rad, from the CAD ROM scan
MJ_MARGIN = 0.03    # MuJoCo hard stops sit INSIDE the URDF limits, so the measured
                    # position can never trip ros2_control's joint limiter
SOFT_MARGIN = 0.12  # the gait must stay this far inside the mechanical limit
J_EFF  = 3.0        # N*m, ST3215 stall 30 kg*cm
J_VEL  = 4.7        # rad/s, 0.222 s / 60 deg
STANCE = {"pitch": -0.42, "knee": 1.05}                  # nominal standing angles

def joints_of(leg):
    sy = SY[leg]
    return [
        (f"{leg}_roll",  "base_link",      f"{leg}_hip",   (0, 0, 0),
         (SX[leg]*R.ROLL_X*MM, sy*R.ROLL_Y*MM, R.ROLL_Z*MM), (1, 0, 0), J_LIM["roll"]),
        (f"{leg}_pitch", f"{leg}_hip",     f"{leg}_thigh", (0, 0, 0),
         (0.0, sy*(R.LEG_Y-R.ROLL_Y)*MM, (R.PITCH_Z-R.ROLL_Z)*MM), (0, 1, 0), J_LIM["pitch"]),
        (f"{leg}_knee",  f"{leg}_thigh",   f"{leg}_shin",  (0, 0, 0),
         (0.0, 0.0, -R.L_THIGH*MM), (0, 1, 0), J_LIM["knee"]),
    ]

JOINTS = [j for leg in LEGS for j in joints_of(leg)]
JOINT_NAMES = [j[0] for j in JOINTS]

def stance_height():
    q2, q3 = STANCE["pitch"], STANCE["knee"]
    w = -R.L_THIGH*math.cos(q2) - R.L_SHIN*math.cos(q2+q3)
    return -(R.PITCH_Z + w - R.FOOT_D/2) * MM        # base origin above the ground, m

Z0 = stance_height() + 0.003
print(f"nominal stance: base {Z0*1000:.1f} mm above ground")

# collision primitives, per link, in link coordinates (mm)
def collisions(name, leg=None):
    if name == "base_link":
        return [("box", (R.BODY_L/2, R.BODY_W/2, (R.BODY_Z1+R.DECK_T-R.BODY_Z0)/2),
                 (0, 0, (R.BODY_Z1+R.DECK_T+R.BODY_Z0)/2), None),
                ("box", (R.ROLL_X+R.SLEEVE_LEN/2, R.ROLL_Y+13.11, 15.36), (0, 0, 0), None)]
    if name.endswith("_hip"):
        sy = SY[leg]
        return [("box", (R.S_W/2+R.SLEEVE_W, R.SLEEVE_LEN/2, 25.6),
                 (0, sy*(R.LEG_Y-R.ROLL_Y), R.PITCH_Z-R.ROLL_Z-17.5), None)]
    if name.endswith("_thigh"):
        return [("box", (R.S_W/2+R.SLEEVE_W, R.SLEEVE_LEN/2, 35.0), (0, 0, -R.L_THIGH+31.0), None)]
    if name.endswith("_shin"):
        return [("capsule", (12.0,), (0, 0, -R.L_SHIN/2), (0, 0, -8.0, 0, 0, -R.L_SHIN+13.0)),
                ("sphere", (R.FOOT_D/2,), (0, 0, -R.L_SHIN), "foot")]
    return []

# ---------------------------------------------------------------- URDF
def inertial_xml(body, indent):
    I = body.inertia
    c = body.com * MM
    p = " " * indent
    return (f'{p}<inertial>\n'
            f'{p}  <origin xyz="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" rpy="0 0 0"/>\n'
            f'{p}  <mass value="{body.m:.5f}"/>\n'
            f'{p}  <inertia ixx="{I[0,0]:.8f}" ixy="{I[0,1]:.8f}" ixz="{I[0,2]:.8f}"'
            f' iyy="{I[1,1]:.8f}" iyz="{I[1,2]:.8f}" izz="{I[2,2]:.8f}"/>\n'
            f'{p}</inertial>\n')

def write_urdf():
    out = ['<?xml version="1.0"?>',
           '<!-- generated by smalldog_description/scripts/generate_model.py -->',
           '<robot name="smalldog">',
           '  <material name="grey"><color rgba="0.45 0.48 0.52 1"/></material>',
           '  <material name="orange"><color rgba="0.85 0.55 0.15 1"/></material>',
           '  <material name="servo"><color rgba="0.12 0.12 0.14 1"/></material>']
    for name, d in links.items():
        mat = "orange" if name.endswith("_hip") else "grey"
        leg = name.split("_")[0] if "_" in name and name != "base_link" else None
        out.append(f'  <link name="{name}">')
        out.append(inertial_xml(d["body"], 4).rstrip("\n"))
        out.append(f'    <visual>\n'
                   f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
                   f'      <geometry><mesh filename="package://smalldog_description/meshes/'
                   f'{d["mesh"]}.stl" scale="0.001 0.001 0.001"/></geometry>\n'
                   f'      <material name="{mat}"/>\n'
                   f'    </visual>')
        for sv in SERVO_VIS.get(name, []):          # ST3215 bodies: visual only,
            out.append(f'    <visual>\n'           # their mass is already in <inertial>
                       f'      <origin xyz="0 0 0" rpy="0 0 0"/>\n'
                       f'      <geometry><mesh filename="package://smalldog_description/'
                       f'meshes/{sv}.stl" scale="0.001 0.001 0.001"/></geometry>\n'
                       f'      <material name="servo"/>\n'
                       f'    </visual>')
        for kind, size, pos, extra in collisions(name, leg):
            p = tuple(x*MM for x in pos)
            if kind == "box":
                g = f'<box size="{size[0]*2*MM:.4f} {size[1]*2*MM:.4f} {size[2]*2*MM:.4f}"/>'
            elif kind == "sphere":
                g = f'<sphere radius="{size[0]*MM:.4f}"/>'
            else:
                g = f'<cylinder radius="{size[0]*MM:.4f}" length="{(R.L_SHIN-21)*MM:.4f}"/>'
            out.append(f'    <collision><origin xyz="{p[0]:.4f} {p[1]:.4f} {p[2]:.4f}"'
                       f' rpy="0 0 0"/><geometry>{g}</geometry></collision>')
        out.append('  </link>')
    for jn, parent, child, rpy, xyz, axis, lim in JOINTS:
        out.append(f'  <joint name="{jn}" type="revolute">\n'
                   f'    <parent link="{parent}"/>\n'
                   f'    <child link="{child}"/>\n'
                   f'    <origin xyz="{xyz[0]:.5f} {xyz[1]:.5f} {xyz[2]:.5f}" rpy="0 0 0"/>\n'
                   f'    <axis xyz="{axis[0]} {axis[1]} {axis[2]}"/>\n'
                   f'    <limit lower="{-lim:.4f}" upper="{lim:.4f}"'
                   f' effort="{J_EFF}" velocity="{J_VEL}"/>\n'
                   f'  </joint>')
    out.append('</robot>')
    with open(os.path.join(URDF, "smalldog.urdf"), "w") as f:
        f.write("\n".join(out) + "\n")
    print("wrote urdf/smalldog.urdf")

# ---------------------------------------------------------------- MJCF
def mj_inertial(body, indent):
    I = body.inertia
    I = 0.5 * (I + I.T)
    c = body.com * MM
    p = " " * indent
    return (f'{p}<inertial pos="{c[0]:.6f} {c[1]:.6f} {c[2]:.6f}" mass="{body.m:.5f}"'
            f' fullinertia="{I[0,0]:.8f} {I[1,1]:.8f} {I[2,2]:.8f}'
            f' {I[0,1]:.8f} {I[0,2]:.8f} {I[1,2]:.8f}"/>')

def mj_geoms(name, leg, indent):
    p = " " * indent
    rows = [f'{p}<geom class="visual" type="mesh" mesh="{name}"/>']
    for sv in SERVO_VIS.get(name, []):
        rows.append(f'{p}<geom class="servo" type="mesh" mesh="{sv}"/>')
    for kind, size, pos, extra in collisions(name, leg):
        q = tuple(x*MM for x in pos)
        if kind == "box":
            rows.append(f'{p}<geom class="collision" type="box"'
                        f' size="{size[0]*MM:.5f} {size[1]*MM:.5f} {size[2]*MM:.5f}"'
                        f' pos="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f}"/>')
        elif kind == "sphere":
            rows.append(f'{p}<geom name="{leg}_foot" class="foot" type="sphere"'
                        f' size="{size[0]*MM:.5f}" pos="{q[0]:.5f} {q[1]:.5f} {q[2]:.5f}"/>')
        else:
            f0 = tuple(x*MM for x in extra)
            rows.append(f'{p}<geom class="collision" type="capsule" size="{size[0]*MM:.5f}"'
                        f' fromto="{f0[0]:.5f} {f0[1]:.5f} {f0[2]:.5f}'
                        f' {f0[3]:.5f} {f0[4]:.5f} {f0[5]:.5f}"/>')
    return rows

def write_mjcf():
    o = ['<mujoco model="smalldog">',
         '  <!-- generated by smalldog_description/scripts/generate_model.py -->',
         '  <compiler angle="radian" meshdir="../meshes" autolimits="true"/>',
         '  <asset>']
    for name in links:
        o.append(f'    <mesh name="{name}" file="{name}.stl" scale="0.001 0.001 0.001"/>')
    for extras in SERVO_VIS.values():
        for sv in extras:
            o.append(f'    <mesh name="{sv}" file="{sv}.stl" scale="0.001 0.001 0.001"/>')
    o += ['  </asset>', '  <worldbody>',
          f'    <body name="base_link" pos="0 0 {Z0:.4f}" childclass="mujoco">',
          '      <freejoint name="base_freejoint"/>',
          mj_inertial(links["base_link"]["body"], 6)]
    o += mj_geoms("base_link", None, 6)
    o += ['      <site name="imu" pos="0 0 0" size="0.005"/>']
    for leg in LEGS:
        jr, jp, jk = joints_of(leg)
        o.append(f'      <body name="{leg}_hip" pos="{jr[4][0]:.5f} {jr[4][1]:.5f} {jr[4][2]:.5f}">')
        o.append(f'        <joint name="{jr[0]}" axis="1 0 0"'
                 f' range="{-jr[6]+MJ_MARGIN:.4f} {jr[6]-MJ_MARGIN:.4f}"/>')
        o.append(mj_inertial(links[f"{leg}_hip"]["body"], 8))
        o += mj_geoms(f"{leg}_hip", leg, 8)
        o.append(f'        <body name="{leg}_thigh" pos="{jp[4][0]:.5f} {jp[4][1]:.5f} {jp[4][2]:.5f}">')
        o.append(f'          <joint name="{jp[0]}" axis="0 1 0"'
                 f' range="{-jp[6]+MJ_MARGIN:.4f} {jp[6]-MJ_MARGIN:.4f}"/>')
        o.append(mj_inertial(links[f"{leg}_thigh"]["body"], 10))
        o += mj_geoms(f"{leg}_thigh", leg, 10)
        o.append(f'          <body name="{leg}_shin" pos="{jk[4][0]:.5f} {jk[4][1]:.5f} {jk[4][2]:.5f}">')
        o.append(f'            <joint name="{jk[0]}" axis="0 1 0"'
                 f' range="{-jk[6]+MJ_MARGIN:.4f} {jk[6]-MJ_MARGIN:.4f}"/>')
        o.append(mj_inertial(links[f"{leg}_shin"]["body"], 12))
        o += mj_geoms(f"{leg}_shin", leg, 12)
        o.append(f'            <site name="{leg}_foot_site" pos="0 0 {-R.L_SHIN*MM:.5f}" size="0.004"/>')
        o += ['          </body>', '        </body>', '      </body>']
    o += ['    </body>', '  </worldbody>', '  <actuator>']
    for jn in JOINT_NAMES:
        o.append(f'    <position class="mujoco" name="{jn}" joint="{jn}"/>')
    o += ['  </actuator>', '  <sensor>',
          '    <framequat name="imu_quat" objtype="site" objname="imu"/>',
          '    <gyro name="imu_gyro" site="imu"/>',
          '    <accelerometer name="imu_acc" site="imu"/>']
    for leg in LEGS:
        o.append(f'    <touch name="{leg}_contact" site="{leg}_foot_site"/>')
    o += ['  </sensor>', '</mujoco>']
    with open(os.path.join(MJCF, "robot.xml"), "w") as f:
        f.write("\n".join(o) + "\n")
    print("wrote mujoco/robot.xml")

DEFAULTS = f'''<mujoco>
  <default>
    <default class="mujoco">
      <joint frictionloss="0.02" armature="0.008" damping="0.12"/>
      <position kp="25" dampratio="1" forcerange="-{J_EFF} {J_EFF}"/>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2" rgba="0.55 0.58 0.62 1"/>
      </default>
      <default class="servo">
        <geom type="mesh" contype="0" conaffinity="0" group="2" rgba="0.13 0.13 0.15 1"/>
      </default>
      <default class="collision">
        <geom group="3" contype="4" conaffinity="1" rgba="0.9 0.2 0.2 0.3"/>
      </default>
      <default class="foot">
        <geom group="3" contype="2" conaffinity="1" friction="1.2 0.02 0.001"
              solref="0.008 1" solimp="0.95 0.99 0.001" rgba="0.15 0.15 0.15 1"/>
      </default>
    </default>
  </default>
</mujoco>
'''

SCENE = '''<mujoco model="smalldog_scene">
  <option timestep="0.001" integrator="implicitfast" solver="Newton" tolerance="1e-8"/>
  <size memory="64M"/>

  <include file="defaults.xml"/>
  <include file="robot.xml"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="140" elevation="-20" offwidth="1400" offheight="1000"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0"
             width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
             rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8"
             width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true"
              texrepeat="5 5" reflectance="0.15"/>
  </asset>

  <worldbody>
    <light pos="0 0 2.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" pos="0 0 0" type="plane" material="groundplane"
          condim="3" contype="1" conaffinity="15" friction="1.0 0.005 0.0001"/>
  </worldbody>
</mujoco>
'''

write_urdf()
write_mjcf()
open(os.path.join(MJCF, "defaults.xml"), "w").write(DEFAULTS)
open(os.path.join(MJCF, "scene.xml"), "w").write(SCENE)
print("wrote mujoco/defaults.xml, mujoco/scene.xml")

with open(os.path.join(PKG, "robot_params.json"), "w") as f:
    json.dump({
        "joint_names": JOINT_NAMES,
        "legs": LEGS,
        "sign_x": SX, "sign_y": SY,
        "hip_xyz_mm":   {l: list(map(float, origin(l, "hip"))) for l in LEGS},
        "hip_to_pitch_mm": {l: [0.0, float(SY[l]*(R.LEG_Y-R.ROLL_Y)), float(R.PITCH_Z-R.ROLL_Z)]
                            for l in LEGS},
        "l_thigh_mm": R.L_THIGH, "l_shin_mm": R.L_SHIN, "foot_r_mm": R.FOOT_D/2,
        "joint_limits_rad": J_LIM,
        "joint_soft_limits_rad": {k: round(v - SOFT_MARGIN, 4) for k, v in J_LIM.items()},
        "joint_velocity_limit": J_VEL, "stance_rad": STANCE,
        "stance_base_height_m": Z0,
        "total_mass_kg": round(total, 4),
    }, f, indent=2)
print("wrote robot_params.json")
