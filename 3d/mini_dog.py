#!/usr/bin/env python
"""
mini_dog.py - parametric 12-DOF quadruped, fully 3D-printed V1 (PETG/ASA).

Servo interface taken from the official Waveshare ST3215 CAD/drawing
(ref/ST3215-3D/ST3215.step, ref/ST3215-2D/ST3215.pdf), NOT estimated.

Joint concept (printed-first, no machined parts, no external bearings):
  proximal link ends in a rectangular SLEEVE around the servo case (form-fit,
  takes the reaction torque on 45x35 flats);
  distal link is a FORK whose two arms bolt to the two supplied aluminium hubs
  (driven 25T side + stock passive side) with 4x M2.5 each on a 14 mm bolt circle.
  The fork arms straddle the sleeve, so the servo is also captured axially.

Coordinates: +X forward, +Y left, +Z up. Origin = chassis centre at hip-roll height.
Units: mm.
"""
import math, os, json
import cadquery as cq

HERE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(HERE, "out")

# =====================================================================================
# 1. ST3215 - measured from the official Waveshare model + drawing SCS215 rev 2022/6/8
# =====================================================================================
S_L, S_W, S_H = 45.22, 24.72, 35.00   # case: length x width x height(along the axis)
S_AX          = 10.11                 # axis offset from the output-side end face
HUB_D         = 19.20                 # both aluminium hub plates
HUB_BC        = 14.00                 # 4x M2.5 clearance holes, at 0/90/180/270 deg
HUB_N         = 4
HUB_TOP_Z     = +20.30                # driven-hub outer face  (case top  +17.50)
HUB_BOT_Z     = -16.95                # passive-hub outer face (case base -17.50, recessed)
HUB_REC_D     = 25.00                 # recess in the case base around the passive hub
# hub plates as they actually are in ref/ST3215-3D/ST3215.step (axis = STEP +Y at
# x=-25.5, z=0; STEP y + 10.70 = local z).  Both plates carry 4x @2.5 THROUGH holes on
# the @14 circle at 0/90/180/270 - the screw passes through the hub, it does not thread
# into it.  Driven plate: y 7.10..9.60 -> z 17.80..20.30, central @6.2 pocket stepping to
# @3.2 at the face (the M3 output-shaft screw - a driver must reach it).  Passive plate:
# y -27.65..-25.45 -> z -16.95..-14.75, central @6.0 bore over the case's @2.6 hole.
HUB_T_TOP     = 2.50                  # driven plate thickness
HUB_T_BOT     = 2.20                  # passive plate thickness
HUB_CTR_D     = 6.10                  # central bore of both plates
HUB_SCR_D     = 3.20                  # driven plate: central screw hole at the outer face
HUB_SCR_T     = 1.50                  # ... its depth
CONN_W, CONN_H, CONN_D = 15.0, 12.0, 6.0   # cable/connector zone on the far end face

# =====================================================================================
# 2. print / fit
# =====================================================================================
CLR, ROTCLR = 0.35, 0.40
WALL, SLEEVE_W = 2.8, 3.0
SLEEVE_LEN = 33.0                     # < 2*16.95 so the fork arms never touch it
ARM_T      = 4.0
ARM_R      = 13.5
ARM_BOT_TOP = -17.90                  # bottom-arm top face (0.4 under the case base)
FORK_Y0, FORK_Y1 = ARM_BOT_TOP-ARM_T, HUB_TOP_Z+ARM_T   # fork outer faces, on the axis
SPINE_R0, SPINE_R1, SPINE_W = 23.0, 31.0, 28.0
LIGHT_L, LIGHT_D = 20.0, 12.0         # sleeve cooling window: obround, length x width
M25_CLR, M3_CLR = 1.45, 1.70
M3_NUT_AF, M3_NUT_H = 5.60, 2.70

# =====================================================================================
# 3. robot
# =====================================================================================
BODY_L, BODY_W   = 126.0, 92.0
BODY_Z0, BODY_Z1 = -25.0, 25.0
DECK_T           = 4.0
ROLL_X, ROLL_Y, ROLL_Z = 90.0, 36.0, 0.0
PITCH_X, LEG_Y, PITCH_Z = 90.0, 76.0, -30.0   # pitch axis sits below the roll fork sweep
L_THIGH, L_SHIN  = 75.0, 82.0
KNEE_Z = PITCH_Z - L_THIGH
FOOT_Z = KNEE_Z - L_SHIN
FOOT_D           = 26.0
# Shin profile, after the Waveshare DOG PRO lower leg in ref/ROBOTIC_DOG_-STEP - the one
# real quadruped link this repo actually owns.  Measured by tools/ref_ws_shin.py:
# 101.8 mm between joint centres, a CONSTANT 12 mm plate thickness, an in-plane depth that
# tapers 26.3 -> 8.9 mm (x0.34) from the knee boss to the ankle, a round boss at each end,
# and a centreline bowed 5.3 % of the length off the chord.  So it is a blade: deep in the
# plane it bends in, thin along the joint axis - and it is curved, not a straight wedge.
# Ours keeps the proportions but not the absolute sizes: PETG is ~40x less stiff than the
# aluminium that part is milled from, so the blade stays much fatter than 0.34.
#
# u = mm below the knee axis.  X = fore-aft (the knee's bending plane), Y = lateral (along
# the knee axis), bow = the X offset of the section centre, r = section corner radius.
# Interpolated with a monotone cubic: monotone means the curve cannot overshoot the table,
# which is what stops the loft spline inventing bulges the parameters never asked for.
SHIN_PROFILE = (
    #  u      X      Y    bow     r
    (25.0, 27.6, 42.6,  0.0,  1.5),   # buried in the fork spine, 0.2 inside it on every face
    (31.0, 27.6, 42.6,  0.0,  1.5),   # fork outer face - the boss
    (34.0, 27.4, 37.0, -0.8,  3.5),   # shoulder: the lateral width collapses out of the fork
    (38.0, 27.0, 28.0, -2.2,  6.5),
    (43.0, 26.0, 22.6, -3.6,  9.0),
    (49.0, 25.2, 20.6, -4.4,  9.4),   # blade: near-constant lateral, tapering fore-aft
    (55.0, 23.4, 19.8, -4.2,  9.0),
    (61.0, 21.4, 19.2, -3.0,  8.6),
    (65.0, 20.2, 19.2, -1.6,  8.4),   # waist
    (69.0, 20.0, 20.4,  0.0,  9.5),   # ankle boss, round, around the 18 mm foot spigot
)
SHIN_WALL, SHIN_RIB = 3.2, 2.6                # wall, and the central shear web
SHIN_CAV   = (36.0, 62.0)                     # cavity: closed by a bulkhead at either end
SHIN_TIE_U = (45.0, 51.0)                     # one tie anchor: in one slot, out the other
BATT_L, BATT_W, BATT_H = 72.0, 67.0, 45.0    # 6x21700 3S2P, 2 layers of 3, cells along X
BMS_L, BMS_W, BMS_H    = 64.0, 27.0, 13.0
OPI_HOLES  = (92.0, 54.0)
LIDAR_BC, LIDAR_N, LIDAR_H = 45.0, 4, 44.0
STAND_PITCH, STAND_KNEE = -22.0, 46.0

# =====================================================================================
# 4. mass / drive - the single source for fea.py, export_sim.py, the BOM AND the ROS 2
#    description generator in ../ros2/smalldog_description/scripts/generate_model.py.
#    Nothing downstream may keep its own copy of these; that is how the servo mass ended
#    up as 55 g on this side and 60 g in the ROS 2 model.
# =====================================================================================
PRINT_RHO         = 1.27 * 0.55   # g/cm3 - PETG at ~5 walls / 40 %, i.e. solid volume x this
TPU_RHO           = 1.21 * 0.35   # g/cm3 - TPU 95A at 3 walls / 25 %, i.e. the feet
SERVO_KG          = 0.060         # ST3215 incl. both hubs and the bolts - **verify**: not in
                                  # ref/, vendor figure only.  Weigh one before trusting it;
                                  # 12 of them are a third of the robot.
N_SERVO           = 12
BATTERY_KG        = 0.42          # 3S2P, 6x21700
ELECTRONICS_KG    = 0.25          # Orange Pi 5 Pro / BMS / wiring
LIDAR_KG          = 0.230         # Unitree L2 on the pedestal - **verify**
TPU_PARTS         = ("foot",)     # printed in TPU_RHO, everything else in PRINT_RHO
SERVO_STALL_NM    = 2.94          # 30 kg*cm at 12 V
SERVO_NOLOAD_RADS = 4.71          # 0.222 s / 60 deg at 12 V

# =====================================================================================
# helpers
# =====================================================================================
def W(sh): return cq.Workplane(obj=sh)
def part_rho(name): return TPU_RHO if name in TPU_PARTS else PRINT_RHO
def bxc(x0,x1,y0,y1,z0,z1):
    x0,x1 = min(x0,x1),max(x0,x1); y0,y1 = min(y0,y1),max(y0,y1); z0,z1 = min(z0,z1),max(z0,z1)
    return W(cq.Solid.makeBox(x1-x0,y1-y0,z1-z0,cq.Vector(x0,y0,z0)))
def cyl(r,h,base=(0,0,0),axis=(0,0,1)):
    if h < 0: base = tuple(b + h*a for b,a in zip(base, axis)); h = -h
    return W(cq.Solid.makeCylinder(r,h,cq.Vector(*base),cq.Vector(*axis)))
def rrect(x,y,r,c=(0,0,0)):
    """rounded-rectangle wire in XY, centred on c"""
    r = max(0.4, min(r, x/2-0.4, y/2-0.4)); hx, hy = x/2-r, y/2-r
    return (cq.Workplane("XY", origin=c)
            .moveTo(hx, y/2).lineTo(-hx, y/2).radiusArc((-x/2,  hy), -r)
            .lineTo(-x/2, -hy).radiusArc((-hx, -y/2), -r)
            .lineTo(hx, -y/2).radiusArc(( x/2, -hy), -r)
            .lineTo(x/2,  hy).radiusArc(( hx,  y/2), -r).close().wire().val())
def loft(wires): return W(cq.Solid.makeLoft(wires, False))
def mono(xs, ys):
    """Fritsch-Carlson monotone cubic through (xs, ys) -> f(x).  C1 and, unlike a plain
    spline, guaranteed not to overshoot the control points between them."""
    n = len(xs)
    h = [xs[i+1]-xs[i] for i in range(n-1)]
    d = [(ys[i+1]-ys[i])/h[i] for i in range(n-1)]
    m = [d[0]] + [0.0]*(n-2) + [d[-1]]
    for i in range(1, n-1):
        if d[i-1]*d[i] > 0:
            w1, w2 = 2*h[i]+h[i-1], h[i]+2*h[i-1]
            m[i] = (w1+w2)/(w1/d[i-1] + w2/d[i])
    def f(x):
        x = min(max(x, xs[0]), xs[-1])
        i = max(k for k in range(n-1) if xs[k] <= x) if x < xs[-1] else n-2
        t = (x-xs[i])/h[i]; t2, t3 = t*t, t*t*t
        return ((2*t3-3*t2+1)*ys[i] + (t3-2*t2+t)*h[i]*m[i]
                + (-2*t3+3*t2)*ys[i+1] + (t3-t2)*h[i]*m[i+1])
    return f
def hexn(af,h,c): return cq.Workplane("XY").polygon(6,2*af/math.sqrt(3)).extrude(h).translate(c)
def frame(o,xdir,zdir): return cq.Location(cq.Plane(origin=o,xDir=xdir,normal=zdir))
def mv(wp,loc): return W(wp.val().moved(loc))
def mirX(wp): return wp.mirror("YZ")
def mirY(wp): return wp.mirror("XZ")

# =====================================================================================
# servo primitives (servo frame: axis +Z, driven hub +Z, case body toward +X,
#                   link direction -X, axis at S_AX from the -X end face)
# =====================================================================================
def servo_case(clr=0.0):
    return bxc(-S_AX-clr, S_L-S_AX+clr, -S_W/2-clr, S_W/2+clr, -S_H/2-clr, S_H/2+clr)

def hub_plate(top=True):
    """one stock aluminium hub, in servo-local coords - the face a fork arm bolts to.
    Visualisation/interface reference only: no printed part is cut against it."""
    if top:
        z0, z1 = HUB_TOP_Z-HUB_T_TOP, HUB_TOP_Z
        h = cyl(HUB_D/2, HUB_T_TOP, (0,0,z0))
        h = h.cut(cyl(HUB_CTR_D/2, HUB_T_TOP-HUB_SCR_T, (0,0,z0)))
        h = h.cut(cyl(HUB_SCR_D/2, HUB_T_TOP+2, (0,0,z0-1)))
    else:
        z0, z1 = HUB_BOT_Z, HUB_BOT_Z+HUB_T_BOT
        h = cyl(HUB_D/2, HUB_T_BOT, (0,0,z0))
        h = h.cut(cyl(HUB_CTR_D/2, HUB_T_BOT+2, (0,0,z0-1)))
    for i in range(HUB_N):
        th = math.radians(90*i)
        h = h.cut(cyl(2.50/2, 12, (HUB_BC/2*math.cos(th), HUB_BC/2*math.sin(th), z0-1)))
    return h

def hubs():
    return hub_plate(True).union(hub_plate(False))

def servo_dummy():
    s = servo_case()
    # the @25 pocket in the case base is empty down to the passive hub face at HUB_BOT_Z;
    # only r < HUB_D/2 of it is filled, by the hub plate itself.  Modelling that pocket as
    # solid (as this did) blocks the bottom fork arm's pedestal and kills the roll ROM.
    s = s.cut(cyl(HUB_REC_D/2, HUB_BOT_Z+S_H/2, (0,0,-S_H/2)))
    s = s.union(cyl(HUB_D/2, HUB_TOP_Z-S_H/2, (0,0,S_H/2)))
    s = s.union(cyl(HUB_D/2, HUB_T_BOT, (0,0,HUB_BOT_Z)))
    s = s.union(bxc(S_L-S_AX, S_L-S_AX+CONN_D, -CONN_W/2, CONN_W/2, -CONN_H/2, CONN_H/2))
    return s

def servo_envelope(hub=True):
    """cut this from every printed part.

    hub=True  - the general case: also sweep out both hub discs and the base recess.  They
                turn with the distal link, so nothing else may sit in that volume.
    hub=False - for the ONE part whose fork bolts to this servo's hubs.  It turns with
                them, so it needs no clearance to them - it needs the opposite: material
                on the hub faces (HUB_TOP_Z / HUB_BOT_Z) and a pedestal reaching into the
                @25 base recess.  Sweeping the hubs out of that part is what left both
                fork arms with no bolt circle and no contact face at all."""
    s = servo_case(CLR)
    if hub:
        s = s.union(cyl(HUB_D/2+ROTCLR, HUB_TOP_Z+ARM_T-S_H/2, (0,0,S_H/2)))
        s = s.union(cyl(HUB_REC_D/2+ROTCLR, 8.0, (0,0,-S_H/2-4.0)))
    else:
        s = s.cut(cyl(HUB_REC_D/2-CLR, HUB_BOT_Z+S_H/2+1.0, (0,0,-S_H/2-1.0)))
    s = s.union(bxc(S_L-S_AX, S_L-S_AX+CONN_D+14, -CONN_W/2-1, CONN_W/2+1, -CONN_H/2-1, CONN_H/2+1))
    return s

def sleeve(length=SLEEVE_LEN, wall=SLEEVE_W, window=True, lighten=True):
    x0, x1 = -S_AX-wall, S_L-S_AX+wall
    y0, y1 = -S_W/2-wall, S_W/2+wall
    s = bxc(x0, x1, y0, y1, -length/2, length/2)
    s = s.cut(servo_case(CLR))
    if window:                                        # cable / connector escape
        s = s.cut(bxc(S_L-S_AX-1, x1+1, -CONN_W/2, CONN_W/2, -length/2-1, length/2+1))
    if lighten:
        # cooling windows, obround and clear of the wall edges.  The +-y walls are the
        # bending flanges of the link: a sharp 24x20 rectangle here was the peak-stress
        # site of the whole leg (fea.py, thigh_A stall -> 25.6 MPa at its top corner).
        lw, ld, xc = LIGHT_L, LIGHT_D, 14.0
        for sg in (1,-1):
            c = bxc(xc-(lw-ld)/2, xc+(lw-ld)/2, sg*(S_W/2-1), sg*(S_W/2+wall+1), -ld/2, ld/2)
            for e in (-1, 1):
                c = c.union(cyl(ld/2, wall+2, (xc+e*(lw-ld)/2, sg*(S_W/2-1), 0), axis=(0,sg,0)))
            s = s.cut(c)
    return s

def fork(spine_r0=SPINE_R0, spine_r1=SPINE_R1, spine_w=SPINE_W):
    """distal-link end; link direction is servo -X."""
    def arm(z0, z1):
        a = cyl(ARM_R, z1-z0, (0,0,z0))
        a = a.union(bxc(-spine_r1, 0.0, -spine_w/2, spine_w/2, z0, z1))
        a = a.cut(cyl(3.2, 40, (0,0,z0-10)))
        for i in range(HUB_N):
            th = math.radians(90*i)
            px, py = HUB_BC/2*math.cos(th), HUB_BC/2*math.sin(th)
            a = a.cut(cyl(M25_CLR, 40, (px,py,z0-10)))
        return a
    top = arm(HUB_TOP_Z, HUB_TOP_Z+ARM_T)
    bot = arm(ARM_BOT_TOP-ARM_T, ARM_BOT_TOP)
    bot = bot.union(cyl(HUB_REC_D/2-1.0, HUB_BOT_Z-ARM_BOT_TOP, (0,0,ARM_BOT_TOP)))
    z = ARM_BOT_TOP-ARM_T-1                       # re-drill through arm + pedestal, or the
    bot = bot.cut(cyl(3.2, 40, (0,0,z)))          # pedestal plugs the four M2.5 holes
    for i in range(HUB_N):
        th = math.radians(90*i)
        bot = bot.cut(cyl(M25_CLR, 40, (HUB_BC/2*math.cos(th), HUB_BC/2*math.sin(th), z)))
    for i in range(HUB_N):                            # nut pockets, outer faces
        th = math.radians(90*i)
        px, py = HUB_BC/2*math.cos(th), HUB_BC/2*math.sin(th)
        top = top.cut(hexn(5.3, 2.2, (px,py,HUB_TOP_Z+ARM_T-2.2)))
        bot = bot.cut(hexn(5.3, 2.2, (px,py,ARM_BOT_TOP-ARM_T)))
    spine = bxc(-spine_r1, -spine_r0, -spine_w/2, spine_w/2, ARM_BOT_TOP-ARM_T, HUB_TOP_Z+ARM_T)
    return top.union(bot).union(spine)

# =====================================================================================
# joint frames (front-left leg, zero pose = legs straight down)
# =====================================================================================
ROLL_LOC  = frame((ROLL_X,  ROLL_Y, ROLL_Z),  xdir=(0,-1,0), zdir=(1,0,0))
PITCH_LOC = frame((PITCH_X, LEG_Y,  PITCH_Z), xdir=(0,0,1),  zdir=(0,1,0))
KNEE_LOC  = frame((PITCH_X, LEG_Y,  KNEE_Z),  xdir=(0,0,1),  zdir=(0,1,0))
JOINTS = [("roll",ROLL_LOC), ("pitch",PITCH_LOC), ("knee",KNEE_LOC)]

_ENV = {}
def env_leg(no_hub=None):
    """no_hub = name of the joint whose hubs are NOT swept out (see servo_envelope)."""
    key = ("leg", no_hub)
    if key not in _ENV:
        e = None
        for nm, L in JOINTS:
            s = mv(servo_envelope(hub=(nm != no_hub)), L)
            e = s if e is None else e.union(s)
        _ENV[key] = e
    return _ENV[key]

def env_all(no_hub=None):
    key = ("all", no_hub)
    if key not in _ENV:
        e = env_leg(no_hub)
        for f in (mirY, mirX, lambda w: mirX(mirY(w))):
            e = e.union(f(env_leg(no_hub)))
        _ENV[key] = e
    return _ENV[key]

# =====================================================================================
# PART: chassis_bottom
# =====================================================================================
def roll_module():
    """front-left hip-roll cradle: sleeve + boxed neck back to the chassis front wall.
    Nothing may sit in the swept sector of the rotating fork (outboard, r<31)."""
    s = mv(sleeve(), ROLL_LOC)
    xa, xb = BODY_L/2, ROLL_X + SLEEVE_LEN/2                 # 63 .. 104.5
    xm = ROLL_X - 21.9                                       # rear fork-arm plane
    s = s.union(bxc(xa, xb, -14.0, 2.0, -15.36, 15.36)       # inboard rail
                .cut(bxc(xa-1, xb-6.0, -11.0, -1.0, -11.0, 11.0)))
    for z0, z1 in ((12.36, 15.36), (-15.36, -12.36)):
        s = s.union(bxc(xa, xm+SLEEVE_W+1.0, -14.0, 21.5, z0, z1))   # narrow past the fork arm
        s = s.union(bxc(ROLL_X-SLEEVE_LEN/2, xb, -14.0, ROLL_Y+13.11, z0, z1))
    s = s.union(bxc(xa, xm-0.6, -14.0, ROLL_Y+13.11, -15.36, 15.36)  # root gusset
                .cut(bxc(xa-1, xm-3.4, -10.0, ROLL_Y+9.0, -11.0, 11.0)))
    return s

def chassis_bottom():
    s = (bxc(-BODY_L/2, BODY_L/2, -BODY_W/2, BODY_W/2, BODY_Z0, BODY_Z1)
         .cut(bxc(-BODY_L/2+WALL, BODY_L/2-WALL, -BODY_W/2+WALL, BODY_W/2-WALL,
                  BODY_Z0+3.0, BODY_Z1+1)))
    rm = roll_module()
    for f in (lambda w: w, mirY, mirX, lambda w: mirX(mirY(w))):
        s = s.union(f(rm))
    # 6x 21700 cradle: two layers of three, cells along X, low and central
    for iy in range(4):
        y = -BATT_W/2 + iy*(BATT_W/3.0)
        s = s.union(bxc(-BATT_L/2-1, BATT_L/2+1, y-1.4, y+1.4, BODY_Z0+3.0, BODY_Z0+BATT_H))
    for x in (-BATT_L/2-3.0, BATT_L/2):                       # end stops
        s = s.union(bxc(x, x+3.0, -BATT_W/2, BATT_W/2, BODY_Z0+3.0, BODY_Z0+BATT_H))
    for x in (-24.0, 24.0):                                   # battery strap slots
        for y in (-BODY_W/2-1, BODY_W/2-WALL-1):
            s = s.cut(bxc(x-1.7, x+1.7, y, y+WALL+2, BODY_Z0+3.0, BODY_Z0+6.5))
    # BMS bay (front) and ESP32 + URT-1 bay (rear)
    for xc in (46.0, -46.0):
        s = s.union(bxc(xc-1.5, xc+1.5, -BMS_W/2-1.5, BMS_W/2+1.5, BODY_Z0+3, BODY_Z0+16))
    # deck bosses
    for x in (-52.0, -18.0, 18.0, 52.0):
        for y in (-38.0, 38.0):
            s = s.union(cyl(4.6, BODY_Z1-(BODY_Z0+3.0), (x, y, BODY_Z0+3.0)))
            s = s.cut(cyl(1.35, 15.0, (x, y, BODY_Z1-14.0)))
    for y in (-BODY_W/2-1, BODY_W/2-WALL-1):                  # vents / side cable ports
        for x in (-34.0, 0.0, 34.0):
            s = s.cut(bxc(x-11, x+11, y, y+WALL+2, -4.0, 14.0))
    s = s.cut(bxc(-BODY_L/2-1, -BODY_L/2+WALL+1, -16, 16, -10, 10))
    return s.cut(env_all())

# =====================================================================================
# PART: chassis_top / lidar_mount
# =====================================================================================
def chassis_top():
    z0, z1 = BODY_Z1, BODY_Z1+DECK_T
    s = bxc(-BODY_L/2, BODY_L/2, -BODY_W/2, BODY_W/2, z0, z1)
    for x in (-52.0, -18.0, 18.0, 52.0):
        for y in (-38.0, 38.0):
            s = s.cut(cyl(M3_CLR, 20, (x, y, z0-1))).cut(cyl(3.2, 2.2, (x, y, z1-2.2)))
    for sx in (-1, 1):                                        # Orange Pi 5 Pro standoffs
        for sy in (-1, 1):
            p = (-22.0+sx*OPI_HOLES[0]/2, sy*OPI_HOLES[1]/2, z1)
            s = s.union(cyl(4.2, 6.0, p)).cut(cyl(1.35, 8.0, p))
    for i in range(LIDAR_N):
        a = math.radians(360.0*i/LIDAR_N+45.0)
        s = s.cut(cyl(M3_CLR, 20, (42.0+LIDAR_BC/2*math.cos(a), LIDAR_BC/2*math.sin(a), z0-1)))
    s = s.cut(bxc(-16, 16, -34, 34, z0-1, z1+1)).cut(bxc(58, 60, -26, 26, z0-1, z1+1))
    for y in (-BODY_W/2, BODY_W/2-3.0):                       # stiffening lips
        s = s.union(bxc(-BODY_L/2, BODY_L/2, y, y+3.0, z1, z1+6.0))
    return s

def lidar_mount():
    cx, z0 = 42.0, BODY_Z1+DECK_T
    z1 = z0+LIDAR_H
    s = cyl(30.0, 6.0, (cx, 0, z0)).union(cyl(30.0, 7.0, (cx, 0, z1-7.0)))
    for i in range(4):
        a = math.radians(90*i+45)
        px, py = cx+21*math.cos(a), 21*math.sin(a)
        s = s.union(cyl(6.5, z1-z0, (px, py, z0)))
    s = s.cut(cyl(11.0, 200, (cx, 0, z0-10)))
    for i in range(LIDAR_N):
        a = math.radians(360.0*i/LIDAR_N+45.0)
        px, py = cx+LIDAR_BC/2*math.cos(a), LIDAR_BC/2*math.sin(a)
        s = s.cut(cyl(M3_CLR, 30, (px, py, z0-1))).cut(cyl(1.35, 11, (px, py, z1-10.0)))
    return s

# =====================================================================================
# PART: hip_bracket / thigh / shin / foot
# =====================================================================================
def hip_bracket():
    s = mv(fork(), ROLL_LOC).union(mv(sleeve(), PITCH_LOC))
    x0, x1 = PITCH_X-SPINE_W/2, PITCH_X+SPINE_W/2
    ys, ye = ROLL_Y+SPINE_R0-1.0, LEG_Y+SLEEVE_LEN/2
    # shelf: roll spine -> pitch sleeve.  CLOSED box section - as an open 3 mm U-channel
    # this was the softest path in the leg and its root never converged in FEA.
    s = s.union(bxc(x0, x1, ys, ye, ROLL_Z-17.5, ROLL_Z-9.5)
                .cut(bxc(x0+3, x1-3, ys+6.0, ye-3, ROLL_Z-14.5, ROLL_Z-12.5)))
    s = s.union(bxc(x0, x1, ys, LEG_Y-S_W/2-SLEEVE_W,                  # web down the inboard face
                    ROLL_Z-17.5, ROLL_Z+8.0))
    # ramp away the step where the shelf hangs below the fork spine (ROLL_Z-14): a sharp
    # re-entrant corner there has no converged stress, only a mesh-dependent one.
    s = s.cut(cq.Workplane("YZ")
              .polyline([(ys-0.1, ROLL_Z-13.9), (ys+9.0, ROLL_Z-13.9), (ys-0.1, ROLL_Z-18.5)])
              .close().extrude(x1-x0+2).translate((x0-1, 0, 0)))
    return s.cut(env_all("roll"))          # its fork bolts to the roll hubs

def thigh():
    s = mv(fork(), PITCH_LOC).union(mv(sleeve(), KNEE_LOC))
    x0, x1 = PITCH_X-S_W/2-SLEEVE_W, PITCH_X+S_W/2+SLEEVE_W
    y0, y1 = LEG_Y-SLEEVE_LEN/2, LEG_Y+SLEEVE_LEN/2
    s = s.union(bxc(x0, x1, y0, y1, KNEE_Z+S_L-S_AX+SLEEVE_W, PITCH_Z-SPINE_R0-3.0)  # box beam
                .cut(bxc(x0+3, x1-3, y0+3, y1-3, KNEE_Z+S_L-S_AX, PITCH_Z-SPINE_R0)))
    for z in (PITCH_Z-34.0, PITCH_Z-44.0):                            # cable-tie slots
        s = s.cut(bxc(x0-1, x0+4, LEG_Y-3, LEG_Y+3, z, z+2.6))
    return s.cut(env_all("pitch"))         # its fork bolts to the pitch hubs

_SHIN_F = []
def shin_profile(u):
    """X (fore-aft), Y (lateral), bow and corner radius at u mm below the knee axis."""
    if not _SHIN_F:
        us = [p[0] for p in SHIN_PROFILE]
        _SHIN_F.extend(mono(us, [p[k] for p in SHIN_PROFILE]) for k in (1, 2, 3, 4))
    return tuple(f(u) for f in _SHIN_F)

def shin_stations(u0, u1, step=1.8):
    """loft stations: every profile knot in range, plus an even fill, so the surface is
    driven by the table and not by where the sampling happened to land."""
    us = {u0, u1} | {p[0] for p in SHIN_PROFILE if u0 < p[0] < u1}
    us |= {u0 + i*step for i in range(int((u1-u0)/step)+1)}
    return sorted(u for u in us if u0 <= u <= u1)

def shin_wire(u, inset=0.0):
    """section at u, or its inner face `inset` in - a rounded rect inset stays a rounded
    rect, so the wall comes out uniform without a 2D offset."""
    x, y, bow, r = shin_profile(u)
    return rrect(x-2*inset, y-2*inset, r-inset, (PITCH_X+bow, LEG_Y, KNEE_Z-u))

def shin_cell_y(u):
    """centre of one cavity cell, off the shear web"""
    y = shin_profile(u)[1]
    return (SHIN_RIB + (y - 2*SHIN_WALL - SHIN_RIB)/2.0)/2.0

def shin_beam():
    """the blade: one loft, hollowed by a second loft of the same sections inset by the
    wall, split down the middle by a shear web.  The web also halves the roof span the
    slicer has to bridge - the part prints lying on its side."""
    u0, u1 = SHIN_PROFILE[0][0], SHIN_PROFILE[-1][0]
    c0, c1 = SHIN_CAV
    s = loft([shin_wire(u) for u in shin_stations(u0, u1)])
    cav = loft([shin_wire(u, SHIN_WALL) for u in shin_stations(c0, c1)])
    # The web is what is left standing when a slab is taken out of the cavity.  That slab
    # stops 0.5 mm short of the cavity's end caps on purpose: a tool that crosses those
    # faces makes the OCC boolean quietly return its own input, so the beam comes out with
    # no web at all AND with a shell that later cuts fail on just as silently.  After
    # touching any of this, check the section areas, not just isValid().
    cav = cav.cut(bxc(PITCH_X-60, PITCH_X+60, LEG_Y-SHIN_RIB/2, LEG_Y+SHIN_RIB/2,
                      KNEE_Z-c1+0.5, KNEE_Z-c0-0.5))
    return s.cut(cav)

def shin():
    zf = FOOT_Z
    s = mv(fork(), KNEE_LOC).union(shin_beam())   # the fork spine caps the box at the top
    s = s.union(cyl(9.0, 15.0, (PITCH_X, LEG_Y, zf+2.0)))             # foot spigot
    s = s.cut(cyl(1.35, 22.0, (PITCH_X, LEG_Y, zf)))
    for u in SHIN_TIE_U:      # through the LATERAL wall into the near cell: that face is
        x, y, bow, _ = shin_profile(u)    # the neutral axis for the bending the knee does,
        xc, yw = PITCH_X + bow, LEG_Y + y/2.0   # the fore-aft faces are the extreme fibre
        s = s.cut(bxc(xc-1.7, xc+1.7, yw-SHIN_WALL-1.0, yw+1.0,
                      KNEE_Z-u-1.2, KNEE_Z-u+1.2))
    return s.cut(env_all("knee"))          # its fork bolts to the knee hubs

def foot():
    zf = FOOT_Z
    dome = (cq.Workplane("XY").sphere(FOOT_D/2).translate((PITCH_X, LEG_Y, zf))
            .cut(bxc(PITCH_X-20, PITCH_X+20, LEG_Y-20, LEG_Y+20, zf, zf+20)))
    s = cyl(FOOT_D/2, 12.0, (PITCH_X, LEG_Y, zf)).union(dome)
    s = s.cut(cyl(9.15, 13.5, (PITCH_X, LEG_Y, zf+2.0)))
    s = s.cut(cyl(M3_CLR, 24, (PITCH_X, LEG_Y, zf-6)))
    s = s.cut(cyl(3.4, 3.6, (PITCH_X, LEG_Y, zf-1)))
    return s

def servo_gauge():
    g = sleeve(length=18.0, window=False)
    g = g.cut(bxc(-S_AX-4, S_L-S_AX+4, 0.0, S_W, -20, 20))            # half sleeve: quick print
    a = cyl(ARM_R, ARM_T, (0,0,HUB_TOP_Z)).union(
        bxc(-26.0, 0.0, -SPINE_W/2, SPINE_W/2, HUB_TOP_Z, HUB_TOP_Z+ARM_T))
    a = a.cut(cyl(3.2, 20, (0,0,HUB_TOP_Z-1)))
    for i in range(HUB_N):
        th = math.radians(90*i)
        a = a.cut(cyl(M25_CLR, 20, (HUB_BC/2*math.cos(th), HUB_BC/2*math.sin(th), HUB_TOP_Z-1)))
    return g.union(a.translate((0, 46.0, -HUB_TOP_Z)))

# =====================================================================================
# pose / assembly / export
# =====================================================================================
def rot_pitch(w, a): return w.rotate((PITCH_X, LEG_Y, PITCH_Z), (PITCH_X, LEG_Y+1, PITCH_Z), a)
def rot_knee(w, a, p):
    r = math.radians(p)
    kx, kz = PITCH_X - L_THIGH*math.sin(r), PITCH_Z - L_THIGH*math.cos(r)
    return w.rotate((kx, LEG_Y, kz), (kx, LEG_Y+1, kz), a)
def posed(w, kind, p=STAND_PITCH, k=STAND_KNEE):
    if kind == "hip": return w
    w = rot_pitch(w, p)
    return w if kind == "thigh" else rot_knee(w, k, p)

def rom_scan(moving, static, loc_pt, axis=(0,1,0), lo=-150, hi=150, step=10):
    """coarse swept-interference scan about `axis` through loc_pt; returns the free range."""
    free = []
    for a in range(lo, hi+1, step):
        m = moving.rotate(loc_pt, tuple(p+d for p, d in zip(loc_pt, axis)), a)
        try:
            v = m.val().intersect(static.val()).Volume()
        except Exception:
            v = -1.0
        if v < 1.0: free.append(a)
    if not free: return (0, 0)
    best = cur = [free[0], free[0]]
    for a in free[1:]:
        if a - cur[1] <= step: cur[1] = a
        else:
            if cur[1]-cur[0] > best[1]-best[0]: best = cur
            cur = [a, a]
    if cur[1]-cur[0] > best[1]-best[0]: best = cur
    return tuple(best)

PARTS, REPORT = {}, {}
def build():
    hb, th, sh, ft = hip_bracket(), thigh(), shin(), foot()
    PARTS["chassis_bottom"] = (chassis_bottom(), 1, "PETG/ASA, 4 walls, 30% gyroid")
    PARTS["chassis_top"]    = (chassis_top(),    1, "PETG/ASA, 4 walls, 25%")
    PARTS["lidar_mount"]    = (lidar_mount(),    1, "PETG/ASA, 4 walls, 30%")
    PARTS["hip_bracket_A"]  = (hb,       2, "PETG/ASA/PA-CF, 5 walls, 40% - FL+RR")
    PARTS["hip_bracket_B"]  = (mirY(hb), 2, "PETG/ASA/PA-CF, 5 walls, 40% - FR+RL")
    PARTS["thigh_A"]        = (th,       2, "PETG/ASA/PA-CF, 5 walls, 40% - FL+RR")
    PARTS["thigh_B"]        = (mirY(th), 2, "PETG/ASA/PA-CF, 5 walls, 40% - FR+RL")
    PARTS["shin_A"]         = (sh,       2, "PETG/ASA/PA-CF, 5 walls, 40% - FL+RR")
    PARTS["shin_B"]         = (mirY(sh), 2, "PETG/ASA/PA-CF, 5 walls, 40% - FR+RL")
    PARTS["foot"]           = (ft,       4, "TPU 95A, 3 walls, 25%")
    PARTS["servo_gauge"]    = (servo_gauge(), 1, "TEST PRINT FIRST - checks the ST3215 fit")
    return hb, th, sh, ft

def assembly(hb, th, sh, ft):
    a = cq.Assembly(name="mini_dog")
    grey, dark = cq.Color(0.42,0.45,0.50), cq.Color(0.12,0.12,0.14)
    a.add(PARTS["chassis_bottom"][0], name="chassis_bottom", color=grey)
    a.add(PARTS["chassis_top"][0],    name="chassis_top",    color=grey)
    a.add(PARTS["lidar_mount"][0],    name="lidar_mount",    color=grey)
    srv = [mv(servo_dummy(), L) for _, L in JOINTS]
    hub = [mv(hubs(), L) for _, L in JOINTS]
    posed_parts = [("hip_bracket", hb, "hip"), ("thigh", th, "thigh"), ("shin", sh, "shin"),
                   ("foot", ft, "shin")]
    legs = {"FL": lambda w: w, "FR": mirY, "RL": mirX, "RR": lambda w: mirX(mirY(w))}
    for tag, f in legs.items():
        for nm, w, kind in posed_parts:
            a.add(f(posed(w, kind)), name=f"{nm}_{tag}",
                  color=cq.Color(0.85,0.55,0.15) if nm == "hip_bracket" else
                        (dark if nm == "foot" else cq.Color(0.78,0.79,0.82)))
        for i, kind in enumerate(("hip", "thigh", "shin")):
            kind = "hip" if i == 0 else ("thigh" if i == 1 else "shin")
            a.add(f(posed(srv[i], kind)), name=f"servo_{tag}_{JOINTS[i][0]}", color=dark)
            a.add(f(posed(hub[i], kind)), name=f"hub_{tag}_{JOINTS[i][0]}",
                  color=cq.Color(0.66,0.70,0.76))          # stock aluminium hubs
    return a

PRINT_ORIENT = {"chassis_bottom": ((1,0,0),0), "chassis_top": ((1,0,0),180),
                "lidar_mount": ((1,0,0),0), "hip_bracket_A": ((0,1,0),90),
                "hip_bracket_B": ((0,1,0),90), "thigh_A": ((1,0,0),90),
                "thigh_B": ((1,0,0),90), "shin_A": ((0,1,0),90), "shin_B": ((0,1,0),90),
                "foot": ((1,0,0),180), "servo_gauge": ((1,0,0),0)}

def main():
    for d in ("step", "stl"): os.makedirs(os.path.join(OUT, d), exist_ok=True)
    hb, th, sh, ft = build()
    rows = []
    print("\n  part                qty   volume   est.mass    print bbox (mm)")
    for name, (wp, qty, note) in PARTS.items():
        shp = wp.val()
        ok = shp.isValid()
        cq.exporters.export(wp, os.path.join(OUT, "step", f"{name}.step"))
        ax, ang = PRINT_ORIENT[name]
        pw = wp.rotate((0,0,0), ax, ang) if ang else wp
        bb = pw.val().BoundingBox()
        pw = pw.translate((-bb.xmin, -bb.ymin, -bb.zmin))
        cq.exporters.export(pw, os.path.join(OUT, "stl", f"{name}.stl"),
                            tolerance=0.02, angularTolerance=0.15)
        bb = pw.val().BoundingBox()
        v = shp.Volume()/1000.0
        m = v*part_rho(name)
        rows.append({"part": name, "qty": qty, "volume_cm3": round(v,1),
                     "est_mass_g": round(m,1),
                     "print_bbox_mm": [round(bb.xlen,1), round(bb.ylen,1), round(bb.zlen,1)],
                     "valid": ok, "note": note})
        print(f"  {name:18s} x{qty}  {v:7.1f} cm3 {m:7.1f} g   "
              f"{bb.xlen:6.1f} x {bb.ylen:6.1f} x {bb.zlen:6.1f}{'' if ok else '   !! INVALID'}")
    a = assembly(hb, th, sh, ft)
    a.save(os.path.join(OUT, "mini_dog_assembly.step"))
    tm = sum(r["est_mass_g"]*r["qty"] for r in rows)
    carried = (N_SERVO*SERVO_KG + BATTERY_KG + ELECTRONICS_KG + LIDAR_KG)*1000.0
    print(f"\n  printed mass  ~{tm:.0f} g   + {N_SERVO} servos {N_SERVO*SERVO_KG*1000:.0f} g"
          f" + 3S2P pack ~{BATTERY_KG*1000:.0f} g"
          f" + Orange Pi/BMS/wiring ~{ELECTRONICS_KG*1000:.0f} g"
          f" + LiDAR ~{LIDAR_KG*1000:.0f} g"
          f"  ->  ~{(tm+carried)/1000:.2f} kg")
    print("  ROM scan (coarse, 10 deg steps, real solids):")
    rom = {}
    rom["hip_roll"] = rom_scan(hip_bracket(), chassis_bottom().union(mv(servo_dummy(), ROLL_LOC)),
                               (ROLL_X, ROLL_Y, ROLL_Z), axis=(1,0,0), lo=-90, hi=90)
    rom["hip_pitch"] = rom_scan(thigh(), hip_bracket().union(mv(servo_dummy(), PITCH_LOC)),
                                (PITCH_X, LEG_Y, PITCH_Z))
    rom["knee"] = rom_scan(shin(), thigh().union(mv(servo_dummy(), KNEE_LOC)),
                           (PITCH_X, LEG_Y, KNEE_Z))
    for k, v in rom.items():
        print(f"    {k:10s} free {v[0]:+4d} .. {v[1]:+4d} deg  (0 = leg straight down)")
    with open(os.path.join(OUT, "bom.json"), "w") as f:
        json.dump({"parts": rows, "rom_deg": rom,
                   "stance": {"wheelbase": 2*PITCH_X, "track": 2*LEG_Y,
                              "thigh": L_THIGH, "shin": L_SHIN}}, f, indent=2)

if __name__ == "__main__":
    main()
