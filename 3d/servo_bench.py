#!/usr/bin/env python
"""
servo_bench.py - the printed hardware for ../robot/bench, plus a leg rig.

WHAT THIS IS
------------
`../robot/README.md`, "The bench, in order", specifies a servo identification bench and
then says **"Build it first. The parts list is in the project notes"**.  This is that
parts list, as CAD: the frame that holds one ST3215, the two arms its trajectories need,
and the numbers `../robot/bench/sweep.py` has to be told about them.

The protocol it serves, from that README and from `sweep.py`'s own docstring:

    --check                     preflight, nothing moves
    --traj rock                 torque off, rock the arm by hand: does Present Position
                                move by the backlash?  Decides whether the encoder sits
                                after the gearbox, which rl/actuator.py assumes
    --traj freeswing            torque off, released from 1.2 rad, log the decay.  The one
                                direct measurement of J_m.  TWO DIFFERENT ARMS - one
                                cannot separate inertia from Coulomb friction
    --traj hold                 heavy arm, three supply voltages, steady state
    --traj step/triangle/reversal/chirp    same arm, same three voltages

Three things in that protocol fix this geometry, and nothing else about the bench is free:

  1. **q = 0 is the arm hanging straight down.**  `rl/actuator.pendulum_load` is
     `-m*g*r*sin(q)` with `q_zero_is_down=True`, and `fit_bam.py` regresses measured
     current against `m*g*r*sin(q)`.  So the joint axis is horizontal and the arm is a
     pendulum.  A bench that put q = 0 at the horizontal would need cos, and every torque
     constant off it would be wrong.

  2. **The arm has to swing about that.**  `sweep.py --qmax` defaults to 1.4 rad and
     aborts at `|q| > qmax + 0.15`, so the mechanical travel has to reach +-89 deg about
     straight down before anything is in the way, or the stand becomes the limit instead
     of the software.  That is what makes this a portal: the arm swings BETWEEN two legs,
     which is the only place to put structure when the whole lower half plane out to
     r = 180 belongs to the arm.

  3. **Two arms, and the short one carries as little as will still move the joint.**
     `fit_bam.py` gets J_m as `J_tot - J_load`, a difference of two numbers, and refuses
     to run at all when `m*g*r*sin(q0) <= tau_c` ("this arm cannot move the joint. Use a
     longer one").  Both bounds are checked in main() against the servo's own priors in
     `../rl/params/st3215.json`, so the arms are sized against the servo rather than by
     eye.

THE NUMBER THIS FILE EXISTS TO PRODUCE
--------------------------------------
`sweep.py` takes `--mass`, `--radius` and `--arm-inertia` and `fit_bam.py` uses them as

    mgr    = mass * g * radius                  (the gravity torque)
    J_load = arm_inertia + mass * radius^2      (the load inertia)

Note what is NOT in `mgr`: the printed arm's own weight.  It is a pendulum too, and on
these arms it is worth 20-30 % of the payload's first moment - a bias that lands straight
on the torque constant and that no amount of data can see, because the fit is told the
torque rather than measuring it.

Both inputs are free, so both can be made exactly right, and main() prints them:

    mass   := m_payload + m_arm * r_com / r_station     (an EFFECTIVE mass: same torque)
    arm_inertia := J_total_about_axis - mass * r_station^2      (whatever is left over)

`arm_inertia` can come out negative and that is not a bug: the arm's own mass sits at a
smaller radius than the payload, so it buys more gravity torque per unit of inertia than a
point mass at the station does, and the correction has to give the inertia back.  What
matters is that `mgr` and `J_load` both come out right, and they do.

It does mean the argument has to be written `--arm-inertia=-1.0e-04` and not with a space:
argparse reads a leading minus as the start of the next option and fails with "expected
one argument".  main() prints the `=` form for that reason - tested against sweep.py's own
--dry-run, both signs.

RELATIONSHIP TO mini_dog.py
---------------------------
Every servo dimension, fit, clearance, nut pocket, density and mass here is imported from
`mini_dog`.  Nothing is re-typed.  That is not tidiness: the bench holds the servo in
`md.sleeve()` driving `md.fork()` on the two stock aluminium hubs, so what it identifies is
this robot's joint - the printed compliance and the printed backlash included - and not a
servo in a vice.  A bench with its own copy of S_L would drift away from the robot and
start identifying a different machine.

There is also a leg rig here (`leg_tower`, `foot_plate`, `cell_riser`).  It is NOT part of
`../robot/bench`'s protocol, which is single-servo throughout; it is a whole leg at the
robot's own geometry over a load cell, for checking a fitted actuator against a real leg
later.  Build it after the servo bench, or not at all.

Bench frame: +X forward, +Y = the joint axis, +Z up.  Origin on the base plate's
underside, under the joint axis.  q = 0 hangs along -Z; +q swings toward +X.  Units mm.
"""
import math, os, json, argparse
import cadquery as cq

import mini_dog as md
import fea                       # materials only - E and the inter-layer allowable
from export_sim import MP        # mass properties; nothing here reimplements them

OUT  = os.path.join(md.OUT, "bench")
PETG = fea.MATERIALS["PETG"]
G    = 9.80665                   # the value fit_bam.py uses, to the digit

# The servo's own priors, from the tree that will consume this bench's output.  They are
# vendor figures (`"fitted": false`) and that is exactly the point - the bench exists to
# replace them - but they are the only estimate of tau_c and J_m there is, and the arms
# have to be sized against something.  Missing file: fall back and say so.
_PRIOR = {"tau_c": 0.05, "J_m": 0.008, "stall_Nm": md.SERVO_STALL_NM, "src": "built-in"}
_PJSON = os.path.join(os.path.dirname(md.HERE), "rl", "params", "st3215.json")
try:
    with open(_PJSON) as _f:
        _d = json.load(_f)
    _PRIOR.update(tau_c=_d["tau_c"], J_m=_d["J_m"],
                  src=f"../rl/params/st3215.json ({_d.get('source', '?')})")
except Exception:
    pass

# From ../robot/bench/sweep.py.  These live twice, like the LiDAR scan pattern in
# mini_dog.py's note: nothing in 3d/ can import robot/ (it pulls in pyserial and the bus),
# and nothing in robot/ can import CadQuery.  If they change there, change them here.
SWEEP_QMAX   = 1.40                       # --qmax default, rad
SWEEP_ABORT  = 0.15                       # ... and the travel abort's margin over it
SWEEP_START  = 1.20                       # traj_freeswing(start=), the release angle
NEED_DEG     = math.degrees(SWEEP_QMAX + SWEEP_ABORT)      # 88.8 - the travel to clear

# =====================================================================================
# 1. bench parameters
# =====================================================================================
# Hole naming follows mini_dog's: a *_CLR is a RADIUS and goes straight into cyl(), a
# *_D or *_HEAD is a DIAMETER and is halved at the point of use.  Getting that backwards
# is a hole at twice or half its size that still builds and still looks right.
M2_CLR, M3_CLR, M4_CLR, M6_CLR = 1.15, md.M3_CLR, 2.30, 3.40

# The fork's own outermost point - the spine corner at (SPINE_R1, SPINE_W/2).  mini_dog
# rounds this to "34" in prose; here it is computed, because everything above the axis
# is spaced off it.
CLEAR_R   = math.hypot(md.SPINE_R1, md.SPINE_W/2)          # 34.01

# WHY THE AXIS IS THIS HIGH, AND WHY IT IS A PORTAL.
# The arm is a pendulum that has to reach +-NEED_DEG about straight down before the stand
# stops it, and the long arm plus its payload reaches ARM_REACH from the axis.  So the
# whole lower half plane out to that radius belongs to the arm, and there is nowhere to
# put a support except ABOVE the axis - which is exactly where the servo's own case
# already points, and where the sleeve's far end sits at r = 38.1, outside CLEAR_R.
# Two legs outboard of the fork in y carry that head down to the plate, straddling the
# arm.  An L-shaped stand cannot do this: its upright would have to stand in the arm's
# own swept disc.
# AXIS_Z is not free: the long arm plus its payload carrier reaches ARM_REACH from the
# axis (computed below, once both are defined) and has to clear the base plate at the
# bottom of its swing.  main() checks it rather than trusting this number.
AXIS_Z    = 208.0
ARM_CLEAR = 20.0                          # ... the air left under the arm at q = 0
HEAD_Z    = AXIS_Z + md.S_L - md.S_AX + md.SLEEVE_W        # the sleeve's top face
HEAD_T    = 12.0                          # ... and the beam that hangs it
LEG_Y0, LEG_T = 42.0, 9.0                 # each leg: a PLATE, outboard of everything that
LEG_Y1    = LEG_Y0 + LEG_T                # turns.  A plate and not a box because the
                                          # joint's torque bends these IN their own plane,
                                          # which is the stiff direction of a plate and
                                          # the reason a shelled box here is 300 g of PETG
                                          # buying a safety factor that was already 160.
LEG_X0, LEG_X1 = -34.0, 34.0              # at the head; they flare to LEG_FLARE at the foot
LEG_FLARE = 18.0
LEG_LIGHT = 44.0                          # holes down the legs' webs
LEG_LIGHT_Z = (62.0, 114.0, 166.0, 218.0)
BASE_T    = 8.0
BASE_X0, BASE_X1 = -98.0, 98.0
BASE_HY   = 66.0
BASE_R    = 6.0                           # plate corner radius
BASE_WIN  = (62.0, 34.0)                  # the plate is a ring: nothing needs the middle,
                                          # and the arm swings over it
BOLT_AT   = ((-84.0, 56.0), (-84.0, -56.0), (84.0, 56.0), (84.0, -56.0),
             (0.0, 58.0), (0.0, -58.0))
M4_HEAD   = 10.0                          # sized for a WASHER, not the head: 4 mm of
BASE_CBORE = 4.0                          # plate under a bare M4 head is not bearing area

# ARMS.  Two of them, because fit_bam.py cannot separate inertia from Coulomb friction
# from one fall.  The stations are the radii sweep.py is told about; the short arm's are
# small so its J_load stays a modest share of J_m, the long arm's are the 0.10 and 0.15
# that ../robot/README.md's own examples use.
ARM_SHORT_ST = (50.0, 90.0)
ARM_LONG_ST  = (100.0, 150.0)
ARM_TIP      = 12.0                       # beam runs this far past the last station
ARM_W0, ARM_H0 = md.SPINE_W, 22.0         # root section: as wide as the fork's spine
ARM_W1, ARM_H1 = 13.0, 13.0               # tip section
ARM_PAD      = 3.0                        # flat pads either side of a station's bolt
ARM_LIGHTEN  = 9.0                        # holes down the long arm's web: it is stiffness
ARM_LIGHT_R  = (36.0, 62.0, 88.0, 114.0)  # that arm needs, not mass on a moment arm

# The payload carrier.  Its mouth faces along +-Y, i.e. along the joint axis, which is
# horizontal at every swing angle - so loose shot or nuts cannot fall out however far the
# arm goes over, and no lid is load-bearing.  It straddles the beam and its bolt is on
# the station, so its centre of mass IS the station radius and `--radius` is exact.
CUP_D, CUP_HY = 56.0, 23.0                # outer, and half its length along the axis
CUP_WALL      = 3.0
CUP_SLOT      = 1.0                       # slide fit over the beam, per side
CUP_MOUTH     = -1                        # which way the mouth faces in y (-y: away from
                                          # the encoder bridge, so both can be fitted)

# Encoder - OPTIONAL, and not needed by ../robot/bench at all: every quantity that
# protocol fits comes off the servo's own telemetry, which is the point of fitting what
# the robot can actually observe.  This pair is here for the one question the servo cannot
# answer about itself - `--traj rock` asks whether Present Position moves with the
# backlash, and if it does not, only an external angle says whether the JOINT moved.
MAG_D, MAG_H = 6.0, 2.5                   # standard N35, DIAMETRICALLY magnetised - an
MAG_FIT    = -0.10                        # axial one reads as a constant and looks broken
ENC_GAP    = 1.5                          # cap face -> bridge face; main() reports the
ENC_PCB    = (12.7, 12.7, 1.6)            # magnet-to-IC distance that follows  **verify**
ENC_PCB_HOLES = 10.0                      # ... 2 x M2 at this pitch              **verify**
ENC_LEDGE  = 1.0                          # plastic between the board and the magnet
ENC_WIN    = 9.0                          # ... with a window this wide for the IC itself
ENC_SLOT   = 1.5                          # the bridge's mounting slots run +-this in y,
                                          # which is how the gap is actually set: fit it,
                                          # then read the AS5600's AGC register and slide
                                          # the bridge until the gain sits mid-range
ENC_BOLT_Z = (AXIS_Z - 30.0, AXIS_Z + 20.0)        # into the +y leg's inner face
CAP_R      = md.ARM_R                     # magnet cap: the fork arm's own disc
CAP_T      = MAG_H                        # magnet flush both ways, trapped by the arm
CAP_SCREW  = max(L for L in (6, 8, 10, 12) if L <= CAP_T + md.ARM_T + md.HUB_T_TOP)
EYE_D      = 26.0                         # the bore every part that caps a fork arm needs,
                                          # or the four M2.5 under it cannot be reached

# LEG RIG (not part of ../robot/bench - see the module docstring).
TOWER_T    = 10.0                         # flange plate, bolted to a board or an extrusion
TOWER_HY, TOWER_Z = (-46.0, 70.0), 46.0
TOWER_BOLT = ((-36.0, 36.0), (-36.0, -36.0), (58.0, 36.0), (58.0, -36.0))
TOWER_BOLT_R = 40.0                       # ... the radius their nuts must clear
# The cell is a bending beam and it only reads if its two ends are carried by two bodies
# that do not touch: the FREE end hangs off the platform on a spacer, the FIXED end stands
# on the riser, and the air between them is what the beam bends into.
#
#      platform  ---------------------------------  z 0 .. PLATE_T
#      spacer      [====]                           z -PLATE_GAP .. 0     (free end, M4)
#      cell        [==============================] z -PLATE_GAP-CELL_H .. -PLATE_GAP
#      riser                              [======]  down to the board     (fixed end, M5)
#
CELL_L, CELL_W, CELL_H = 80.0, 12.7, 12.7 # a 5 kg straight bar                  **verify**
CELL_END, CELL_PITCH = 6.0, 15.0          # 2 holes per end: this far in, this apart
CELL_M4_D, CELL_M5_D = 3.30, 5.30         # M4 at the free end, M5 at the fixed    **verify**
CELL_M4_HEAD, CELL_M5_HEAD = 7.4, 9.4     # ... sunk flush in the plate / the riser
PLATE_XY, PLATE_T = (96.0, 96.0), 8.0
PLATE_GAP  = 6.0                          # the cell has to be able to bend: this much air
RISER_H    = 16.0                         # riser, from the board up to the cell's underside


# =====================================================================================
# 2. helpers - the bench frame, and the servo in it
# =====================================================================================
# The servo, placed by the SAME frame() mini_dog uses for its own joints.  local +Z (the
# driven hub) -> bench +Y, so the axis is horizontal; local +X (the case body) -> bench +Z,
# so the case stands UP out of the way and the fork's spine - local -X - hangs DOWN.  That
# last mapping is the whole bench: it puts q = 0 at the hanging pendulum that
# rl/actuator.pendulum_load and fit_bam.py both assume.
BENCH_LOC = md.frame((0.0, 0.0, AXIS_Z), xdir=(0, 0, 1), zdir=(0, 1, 0))

def at_bench(wp):  return md.mv(wp, BENCH_LOC)

def prism_y(pts, y0, y1):
    """polygon (x, z) in the bench side view, extruded from y0 to y1."""
    return md.W(cq.Workplane("XZ").polyline(pts).close()
                .extrude(y1 - y0).val()).translate((0, y1, 0))

def disc_y(r, y0, y1, c=(0.0, AXIS_Z)):
    return md.cyl(r, y1 - y0, (c[0], y0, c[1]), axis=(0, 1, 0))

def polar(r, a):
    """a point at radius r, a deg from straight DOWN toward +X - the bench's own q."""
    return (r*math.sin(math.radians(a)), AXIS_Z - r*math.cos(math.radians(a)))

def taper(u):
    """0 at an arm's root, 1 at its tip -> (half-width in y, half-depth in x)."""
    return (ARM_W0 + (ARM_W1-ARM_W0)*u)/2.0, (ARM_H0 + (ARM_H1-ARM_H0)*u)/2.0

def beam_down(r0, r1, n=14):
    """the tapered arm beam, lofted along -Z from radius r0 to r1 on the joint axis.

    Lofted in the XY plane along +Z and then turned over, because md.rrect() draws in XY -
    the same detour md.shin_beam() takes for the same reason.  Sections are rounded
    rectangles, so the taper cannot invent a bulge between them."""
    secs = []
    for i in range(n+1):
        u = i/n
        hy, hx = taper(u)
        secs.append(md.rrect(2*hx, 2*hy, min(hx, hy)*0.45, (0, 0, r0 + (r1-r0)*u)))
    return md.loft(secs).rotate((0, 0, 0), (1, 0, 0), 180).translate((0, 0, AXIS_Z))


# =====================================================================================
# 3. PART: bench_frame  (the servo bench)
# =====================================================================================
def bench_frame(hollow=True):
    """base plate + two legs + head + the robot's own sleeve, one printed part.

    hollow=False returns the same frame with the legs' lightening holes filled.  That is
    the SILHOUETTE, and it is what the swing check runs against: a hole is a void the arm
    cannot reach through, but a per-angle interference scan is static, so against the real
    part the arm can appear inside one and the scan reports travel that does not exist.

    The sleeve is md.sleeve() unmodified - full length, cable window, thrust-clamp lug and
    both nut channels - so the servo is clamped here exactly as it is clamped in a leg,
    down to the two M3 jack screws that take the 0.35 mm of case play out in thrust.
    Identifying a joint held any other way identifies a different joint."""
    # base plate, as a ring: the arm swings over the middle and nothing needs it there.
    plate = md.W(cq.Workplane("XY").add(
        cq.Face.makeFromWires(md.rrect(BASE_X1-BASE_X0, 2*BASE_HY, BASE_R,
                                       ((BASE_X0+BASE_X1)/2, 0, 0))))
        .wires().toPending().extrude(BASE_T).val())
    wx, wy = BASE_WIN
    plate = plate.cut(md.W(cq.Workplane("XY").add(
        cq.Face.makeFromWires(md.rrect(2*wx, 2*wy, BASE_R, (0, 0, -1))))
        .wires().toPending().extrude(BASE_T+2).val()))
    for bx, by in BOLT_AT:
        plate = plate.cut(md.cyl(M4_CLR, BASE_T+2, (bx, by, -1)))
        plate = plate.cut(md.cyl(M4_HEAD/2, BASE_CBORE, (bx, by, BASE_T-BASE_CBORE)))
    s = plate

    # The two legs.  Plates, in the plane the joint's torque bends them: a plate is stiff
    # in plane and this one only has to be stiffer than a PETG joint, which main() reports
    # it is by two orders of magnitude.  They flare toward the foot so the moment gets
    # into the plate over a long root rather than at a corner.
    prof = [(LEG_X0, HEAD_Z+HEAD_T), (LEG_X1, HEAD_Z+HEAD_T),
            (LEG_X1+LEG_FLARE, BASE_T), (LEG_X0-LEG_FLARE, BASE_T)]
    for sg in (+1, -1):
        y0, y1 = sg*LEG_Y0, sg*LEG_Y1
        leg = prism_y(prof, min(y0, y1), max(y0, y1))
        if hollow:
            for lz in LEG_LIGHT_Z:
                leg = leg.cut(disc_y(LEG_LIGHT/2.0, min(y0, y1)-1.0, max(y0, y1)+1.0,
                                     (0.0, lz)))
        s = s.union(leg)

    # the head, and the sleeve hanging from it.  Everything above HEAD_Z is out of the
    # arm's reach by construction - the arm never rises above the axis - and the sleeve's
    # own top face IS HEAD_Z, so the joint hangs straight off the beam with no bracket.
    s = s.union(md.bxc(LEG_X0, LEG_X1, -LEG_Y1, LEG_Y1, HEAD_Z, HEAD_Z+HEAD_T))
    s = s.union(at_bench(md.sleeve()))
    s = s.cut(at_bench(md.servo_case(md.CLR)))            # re-open the bore
    # the connector pokes up into the head; the lead comes out of the top
    s = s.cut(md.bxc(-md.CONN_W/2-2.0, md.CONN_W/2+2.0, -md.CONN_H/2-2.0, md.CONN_H/2+2.0,
                     AXIS_Z+md.S_L-md.S_AX-1.0, HEAD_Z+HEAD_T+1.0))
    # the encoder bridge's two M3, into the +y leg's inner face.  Nut channels open on the
    # leg's OUTER face, which is bare air whatever else is fitted.
    for bz in ENC_BOLT_Z:
        s = s.cut(md.cyl(M3_CLR, LEG_T+2.0, (0.0, LEG_Y0-1.0, bz), axis=(0, 1, 0)))
        s = s.cut(md.nut_slot((0.0, LEG_Y1, bz), (0, 0, 1), up=(0, -1, 0), run=LEG_T-2.0))
    return s


# =====================================================================================
# 4. PARTS: arm_short / arm_long  (the two pendulums)
# =====================================================================================
def arm(stations, lighten=False):
    """md.fork() + a tapered beam hanging down, with a bolt station at each radius.

    Each station is an M6 cross hole through a flat pad, so the payload carrier seats
    square whatever the taper is doing there and its bolt axis - which is its centre of
    mass - lands exactly on the radius sweep.py is told about."""
    r0, r1 = md.SPINE_R0, stations[-1] + ARM_TIP
    s = at_bench(md.fork()).union(beam_down(r0, r1))
    for r in stations:
        hy, _ = taper((r - r0)/(r1 - r0))
        cx, cz = polar(r, 0.0)
        s = s.union(md.bxc(cx-9.0, cx+9.0, -hy-ARM_PAD, hy+ARM_PAD, cz-9.0, cz+9.0))
        s = s.cut(md.cyl(M6_CLR, 2*(hy+ARM_PAD)+4.0, (cx, -hy-ARM_PAD-2.0, cz),
                         axis=(0, 1, 0)))
    if lighten:
        # Only the long arm, and only between the stations: what comes off here is mass on
        # a moment arm, which is the one thing this part must not carry more of than it
        # has to - every gram of it is a correction main() has to make to `--mass`.
        for r in ARM_LIGHT_R:
            if r >= stations[-1]: continue
            cx, cz = polar(r, 0.0)
            s = s.cut(md.cyl(ARM_LIGHTEN/2.0, 2*ARM_W0, (cx, -ARM_W0, cz), axis=(0, 1, 0)))
    return s

def arm_short(): return arm(ARM_SHORT_ST)
def arm_long():  return arm(ARM_LONG_ST, lighten=True)


# =====================================================================================
# 5. PART: mass_cup  (the payload carrier)
# =====================================================================================
def mass_cup(station=ARM_LONG_ST[-1]):
    """a cup that straddles the beam on one M6, mouth along the joint axis.

    The mouth faces along +-Y, and Y is horizontal at every swing angle, so shot or nuts
    cannot fall out however far the arm goes over - the lid is a dust cover, not a
    retainer.  Modelled at the outer long station; it is the same part at any of them, and
    what it weighs is what you weigh, not what this file says."""
    cx, cz = polar(station, 0.0)
    hy = taper(1.0)[0] + ARM_PAD + CUP_SLOT
    s = disc_y(CUP_D/2.0, -CUP_HY, CUP_HY, (cx, cz))
    s = s.cut(disc_y(CUP_D/2.0-CUP_WALL, -CUP_HY+CUP_WALL if CUP_MOUTH > 0 else -CUP_HY-1.0,
                     CUP_HY+1.0 if CUP_MOUTH > 0 else CUP_HY-CUP_WALL, (cx, cz)))
    # the slot the beam drops into, and the M6 across it
    s = s.cut(md.bxc(cx-CUP_D, cx+CUP_D, -hy, hy, cz, cz+CUP_D))
    s = s.cut(md.cyl(M6_CLR, 2*CUP_HY+4.0, (cx, -CUP_HY-2.0, cz), axis=(0, 1, 0)))
    for sg in (-1, 1):                                   # a cable tie over the mouth
        s = s.cut(md.bxc(cx+sg*(CUP_D/2-6.0)-1.6, cx+sg*(CUP_D/2-6.0)+1.6,
                         CUP_MOUTH*(CUP_HY-CUP_WALL-2.0), CUP_MOUTH*(CUP_HY+1.0),
                         cz-CUP_D, cz+CUP_D))
    return s


# =====================================================================================
# 6. PARTS: enc_magnet_cap / enc_bridge  (optional - see the parameter block)
# =====================================================================================
def enc_magnet_cap():
    """the diametric magnet, on the joint axis, on the fork's +y arm.

    Held by the same four M2.5 that hold that arm to the driven hub, longer by exactly the
    cap's thickness - CAP_SCREW, derived, because past the hub's own thread the screw
    bottoms on the case and the vendor FAQ says that burns the servo.  The magnet is a
    through pocket loaded from the fork side, and the fork arm is what stops it coming
    back out, so nothing relies on glue and there is no plastic in the field path."""
    y0 = md.HUB_TOP_Z + md.ARM_T
    s = disc_y(CAP_R, y0, y0 + CAP_T)
    s = s.cut(disc_y((MAG_D + MAG_FIT)/2.0, y0-1.0, y0+CAP_T+1.0))
    for i in range(md.HUB_N):
        th = math.radians(90*i)
        c = (md.HUB_BC/2*math.cos(th), AXIS_Z + md.HUB_BC/2*math.sin(th))
        s = s.cut(disc_y(md.M25_CLR, y0-1.0, y0+CAP_T+1.0, c))
    # a flat, so it is visible which way round a diametric magnet went in
    s = s.cut(md.bxc(-CAP_R-1, -CAP_R+2.0, y0-1.0, y0+CAP_T+1.0, AXIS_Z-4.0, AXIS_Z+4.0))
    return s

def enc_y0():
    """the bridge's inboard face: clear of the rotating cap by ENC_GAP."""
    return md.HUB_TOP_Z + md.ARM_T + CAP_T + ENC_GAP

def mag_to_ic():
    """magnet face -> the AS5600's own face, which is what its datasheet window is on."""
    return ENC_GAP + ENC_LEDGE

def enc_bridge():
    """the AS5600's stator: off the +y leg's inner face, over the axis, nothing touching
    the table.

    Referenced to the LEG rather than to the base plate, so what the frame does under load
    is largely common mode between the sensor and the servo case and the reading stays a
    joint angle.  Its two mounting holes are slots in y: the gap is set at assembly by
    sliding it until the AS5600's AGC register sits mid-range, which is the only honest
    way to set a magnetic air gap.  The inboard end of the slot lands it on the cap, which
    is a hard stop and an obvious one."""
    y0 = enc_y0()
    y1 = y0 + ENC_LEDGE + ENC_PCB[2] + 2.0
    s = md.bxc(-20.0, 20.0, y0, y1, ENC_BOLT_Z[0]-12.0, ENC_BOLT_Z[1]+12.0)
    s = s.union(md.bxc(-14.0, 14.0, y0, LEG_Y0, ENC_BOLT_Z[0]-12.0, ENC_BOLT_Z[0]+2.0))
    s = s.union(md.bxc(-14.0, 14.0, y0, LEG_Y0, ENC_BOLT_Z[1]-2.0, ENC_BOLT_Z[1]+12.0))
    pw, ph, _ = ENC_PCB
    s = s.cut(md.bxc(-pw/2, pw/2, y0+ENC_LEDGE, y1+1.0, AXIS_Z-ph/2, AXIS_Z+ph/2))
    s = s.cut(disc_y(ENC_WIN/2.0, y0-1.0, y1+1.0))
    for sx in (-1, 1):
        s = s.cut(disc_y(M2_CLR, y0-1.0, y1+1.0, (sx*ENC_PCB_HOLES/2.0, AXIS_Z)))
    for bz in ENC_BOLT_Z:                                # slotted in y: that is the gap
        slot = md.bxc(-M3_CLR, M3_CLR, y0-1.0, LEG_Y0+1.0, bz-ENC_SLOT, bz+ENC_SLOT)
        for e in (-1, 1):
            slot = slot.union(md.cyl(M3_CLR, LEG_Y0-y0+2.0, (0.0, y0-1.0, bz+e*ENC_SLOT),
                                     axis=(0, 1, 0)))
        s = s.cut(slot)
    # ... and it stops under the head.  Everything above HEAD_Z is frame.
    s = s.cut(md.bxc(-60.0, 60.0, y0-1.0, LEG_Y1+1.0, HEAD_Z-2.0, HEAD_Z+HEAD_T+2.0))
    return s


# =====================================================================================
# 7. PARTS: leg_tower / foot_plate / cell_riser  (the leg rig - not robot/bench's)
# =====================================================================================
# Tower frame: the ROBOT's own coordinates, translated so the chassis' front face
# (x = BODY_L/2, which is where roll_module() joins the body) lands on x = 0.
TOWER_DX = -md.BODY_L/2.0

def leg_tower():
    """md.roll_module(), unmodified, on a flange.

    The cradle is not a bench copy of the robot's hip - it IS the robot's hip, moved.  A
    leg bolted to it hangs at the real ROLL_Y / LEG_Y / PITCH_Z geometry.  Bolt the flange
    to a board or a 2020 upright with the leg hanging past the edge; how far the roll axis
    has to be above the foot plate is main()'s to report, and it is measured off the posed
    leg rather than off FOOT_Z."""
    s = md.roll_module().translate((TOWER_DX, 0, 0))
    y0, y1 = TOWER_HY
    s = s.union(md.bxc(-TOWER_T, 0.0, y0, y1, -TOWER_Z, TOWER_Z))
    for by, bz in TOWER_BOLT:            # through only - the nut lands on the far face
        s = s.cut(md.cyl(M6_CLR, TOWER_T+2.0, (-TOWER_T-1.0, by, bz), axis=(1, 0, 0)))
    return s

def tower_bolt_margin():
    """how far each flange bolt sits outside the hip-roll fork's swept circle, mm.  A nut
    standing proud of the flange's inboard face inside that circle is a leg that jams at
    one end of the roll travel and nowhere else - which mini_dog's own ROM scan cannot
    see, because the flange is not in the robot's model."""
    return min(math.hypot(by - md.ROLL_Y, bz - md.ROLL_Z) - CLEAR_R
               for by, bz in TOWER_BOLT)

def cell_holes(sx):
    """the two bolt centres at one end of the cell.  sx = -1 free end, +1 fixed end."""
    return [sx*(CELL_L/2 - CELL_END - i*CELL_PITCH) for i in (0, 1)]

def foot_plate():
    """the platform half of a straight-bar load cell, with the cell's free end under it.
    Sized on fea.py's own landing case - 36 N on one foot - so a 5 kg (49 N) bar covers it
    and not much more."""
    px, py = PLATE_XY
    s = md.bxc(-px/2, px/2, -py/2, py/2, 0.0, PLATE_T)
    xs = cell_holes(-1)
    x0, x1 = min(xs)-9.0, max(xs)+9.0
    s = s.union(md.bxc(x0, x1, -CELL_W/2-3.0, CELL_W/2+3.0, -PLATE_GAP, 0.0))
    for cx in xs:
        s = s.cut(md.cyl(CELL_M4_D/2.0, PLATE_T+PLATE_GAP+2.0, (cx, 0.0, -PLATE_GAP-1.0)))
        s = s.cut(md.cyl(CELL_M4_HEAD/2.0, 4.5, (cx, 0.0, PLATE_T-4.5)))   # head, flush
    s = s.cut(md.bxc(min(xs)+8.0, px/2+1.0, -CELL_W/2-2.0, CELL_W/2+2.0,
                     -PLATE_GAP-CELL_H-1.0, -PLATE_GAP))
    # a witness slot: it is how you see, with the leg standing on it, that the platform is
    # still floating and has not settled onto the riser.
    s = s.cut(md.bxc(px/2-14.0, px/2+1.0, -1.5, 1.5, -1.0, PLATE_T+1.0))
    return s

def cell_riser():
    """the fixed end of the same cell, down onto the board.  A separate part on purpose:
    the two ends of a bending-beam cell must not share a body, or it reads nothing."""
    xs = cell_holes(+1)
    x0, x1 = min(xs)-9.0, max(xs)+9.0
    z1 = -PLATE_GAP - CELL_H
    s = md.bxc(x0, x1, -CELL_W/2-5.0, CELL_W/2+5.0, z1-RISER_H, z1)
    for cx in xs:
        s = s.cut(md.cyl(CELL_M5_D/2.0, RISER_H+2.0, (cx, 0.0, z1-RISER_H-1.0)))
        s = s.cut(md.cyl(CELL_M5_HEAD/2.0, 5.0, (cx, 0.0, z1-RISER_H-0.5)))
    for sy in (-1, 1):
        s = s.cut(md.cyl(M4_CLR, RISER_H+2.0,
                         (max(xs)+4.5, sy*(CELL_W/2+2.6), z1-RISER_H-1.0)))
    return s


# =====================================================================================
# 8. purchased BOM
# =====================================================================================
BUY = [
    ("ST3215 bus servo + both stock aluminium hubs", "1 (3+ to get a spread)", "bench",
     "the device under test.  Every fitted parameter is per servo and they differ; the "
     "policy's margin has to cover the worst one on the robot"),
    ("Waveshare URT-1, or an ESP32 + a TTL half-duplex driver", 1, "bench",
     "the bus robot/feetech drives.  Set the FTDI latency timer to 1 - it ships at 16 ms "
     "and that alone eats the control tick (robot/README.md)"),
    ("ADJUSTABLE bench supply, 9-13 V, >= 3 A, with a current readout", 1, "bench",
     "one of the two things robot/README.md says decides the quality of the fit.  At ONE "
     "voltage the back-EMF damping and the viscous friction are the same column of the "
     "regressor and no data separates them; the runs go at three.  >= 3 A because a "
     "stalled ST3215 draws 2.7 A.  Power the servo from it DIRECTLY, not through the "
     "URT-1 and not through USB - share the ground and nothing else"),
    ("known payload, ~0.1 kg and ~0.4 kg, or shot/nuts and a scale", "2", "bench",
     "what goes in mass_cup.  It is weighed, never assumed: --mass and --radius are what "
     "anchor the whole torque scale"),
    ("scale, 1 g or better", 1, "bench",
     "for the payloads AND for both printed arms - main() prints an estimate from the "
     "solid volume, and a fill factor is not a measurement"),
    ("M6 x 60 + M6 nut + 2 washers (payload -> arm station)", 2, "bench", ""),
    ("M2.5 x 6 / x 7 (fork -> the two hubs, as the robot)", "4 / 4", "bench+leg",
     "per joint"),
    ("M3 x 10 + M3 nut (sleeve thrust clamp)", 2, "bench+leg", "per joint, and not longer"),
    ("M4 x 30 + M4 nut + washer (base plate -> board)", 6, "bench",
     "the frame is stable on its own weight at full load - main() reports the margin - "
     "but a free swing is a dynamic reaction and it should be held down anyway"),
    ("plywood or MDF board, >= 400 x 250 x 18, + 2 G-clamps", 1, "bench+leg",
     "the stand is printed, the ground is not"),
    ("cable tie, 2.5 mm", 6, "bench+leg",
     "servo lead off the head, and one over the cup's mouth"),
    (f"M2.5 x {CAP_SCREW:.0f} (magnet cap + fork arm -> driven hub)", 4, "encoder",
     f"OPTIONAL.  Replaces the robot's x 6 on the +y arm only: {CAP_T:.1f} of cap and "
     f"{md.ARM_T:.1f} of arm to clear, {md.HUB_T_TOP:.1f} of hub to thread into.  Do not "
     f"go longer - past the hub it bottoms on the case"),
    ("AS5600 breakout + N35 magnet 6 x 2.5 DIAMETRICALLY magnetised", 1, "encoder",
     "OPTIONAL - robot/bench fits everything off the servo's own telemetry.  This is for "
     "the one question it cannot answer about itself, --traj rock.  Board outline is "
     "**verify**; an axially magnetised magnet reads as a constant and looks like a dead "
     "sensor"),
    ("M3 x 16 + M3 nut (encoder bridge -> the +y leg)", 2, "encoder", "OPTIONAL"),
    ("straight-bar load cell, 5 kg, + HX711", 1, "leg",
     "outline is **verify** - CELL_* is the common 80 x 12.7 x 12.7 bar"),
    ("M4 x 20 (load cell free end -> foot plate)", 2, "leg",
     f"{PLATE_T:.0f} of plate + {PLATE_GAP:.0f} of spacer, then into the cell's own M4 "
     f"thread.  No nut - **verify** that thread against the bar you buy"),
    ("M5 x 16 (load cell fixed end -> riser)", 2, "leg", "into the cell's own M5 thread"),
    ("M4 x 25 + M4 nut (riser -> board)", 2, "leg", ""),
    ("M6 x 35 + M6 nut + washer (leg tower -> board)", 4, "leg",
     "for an 18 mm board.  The nuts land on the flange's inboard face, where the roll "
     "fork sweeps; main() reports how far the bolt centres clear it"),
]


# =====================================================================================
# 9. build / checks
# =====================================================================================
PARTS, SILHOUETTE = {}, None
STATIC = ("bench_frame", "enc_bridge")
SWINGING = ("arm_short", "arm_long")

def build():
    global SILHOUETTE
    SILHOUETTE = bench_frame(hollow=False)
    PARTS["bench_frame"]    = (bench_frame(),    1, "bench",   "PETG/ASA, 5 walls, 30% gyroid")
    PARTS["arm_short"]      = (arm_short(),      1, "bench",   "PETG/ASA, 5 walls, 40% - WEIGH IT")
    PARTS["arm_long"]       = (arm_long(),       1, "bench",   "PETG/ASA, 5 walls, 40% - WEIGH IT")
    PARTS["mass_cup"]       = (mass_cup(),       1, "bench",   "PETG/ASA, 4 walls, 40%")
    PARTS["enc_magnet_cap"] = (enc_magnet_cap(), 1, "encoder", "PETG/ASA, 4 walls, 60%")
    PARTS["enc_bridge"]     = (enc_bridge(),     1, "encoder", "PETG/ASA, 4 walls, 40%")
    PARTS["leg_tower"]      = (leg_tower(),      1, "leg",     "PETG/ASA, 5 walls, 40%")
    PARTS["foot_plate"]     = (foot_plate(),     1, "leg",     "PETG/ASA, 4 walls, 40%")
    PARTS["cell_riser"]     = (cell_riser(),     1, "leg",     "PETG/ASA, 4 walls, 60%")
    return PARTS

PRINT_ORIENT = {
    # base plate on the bed.  The head bridges only 22.6 mm on either side of the sleeve
    # before it reaches a leg, and the sleeve bore is the same 25 mm ceiling the robot's
    # own roll cradle prints, so none of it wants support.
    "bench_frame":    ((1, 0, 0), 0),
    # joint axis vertical, i.e. layers in the beam's own bending plane - the same reason
    # md.PRINT_ORIENT does this to thigh_A.  An arm is a cantilever in bending; printed
    # the other way the load is inter-layer tension at the root.
    "arm_short":      ((1, 0, 0), 90),
    "arm_long":       ((1, 0, 0), 90),
    "mass_cup":       ((1, 0, 0), 90),
    "enc_magnet_cap": ((1, 0, 0), 90),
    "enc_bridge":     ((1, 0, 0), 90),
    "leg_tower":      ((0, 1, 0), 90),
    "foot_plate":     ((1, 0, 0), 180),
    "cell_riser":     ((1, 0, 0), 0),
}

def rho(name): return md.part_rho(name)      # bench parts are PETG; unknown fill ->
                                             # md.PRINT_FILL_MEAN, and main() says so

def mp_of(name):
    return MP.of(PARTS[name][0], rho(name)*1e-6)

def about_axis(mp):
    """(mass kg, com radius m, J about the joint axis kg*m^2) for a body in bench coords."""
    d2 = (mp.c[0])**2 + (mp.c[2] - AXIS_Z)**2
    return mp.m, math.sqrt(d2)/1000.0, (mp.I[1][1] + mp.m*d2) * 1e-6

def interference():
    """the static parts share one rigid frame; nothing else checks that they do not share
    solid.  Same test, same tolerance as mini_dog.interference()."""
    bad = []
    for i, a in enumerate(STATIC):
        for b in STATIC[i+1:]:
            try:    v = PARTS[a][0].val().intersect(PARTS[b][0].val()).Volume()
            except Exception: v = 0.0
            if v > md.INTERF_TOL: bad.append((a, b, v))
    return bad

def sweep(name, static_names, step=5, lo=-150, hi=150):
    """the free swing of an arm about the joint axis, against the frame it runs in.  This
    is the check that the portal actually bought the travel it exists for: robot/bench
    aborts at NEED_DEG and the stand has to reach further, or the stand is the limit.

    Runs against bench_frame(hollow=False) - see its docstring.  A positive angle here is
    a positive q: rom_scan rotates about +y, which takes the hanging arm toward +x."""
    stat = None
    for n in static_names:
        w = SILHOUETTE if n == "bench_frame" else PARTS[n][0]
        stat = w if stat is None else stat.union(w)
    return md.rom_scan(PARTS[name][0], stat, (0.0, 0.0, AXIS_Z), axis=(0, 1, 0),
                       lo=lo, hi=hi, step=step)

def cup_volume():
    """the carrier's usable internal volume, cm3 - what a filler mass has to fit in."""
    solid = disc_y(CUP_D/2.0, -CUP_HY, CUP_HY, polar(ARM_LONG_ST[-1], 0.0))
    try:
        return (solid.val().Volume() - PARTS["mass_cup"][0].val().Volume())/1000.0
    except Exception:
        return 0.0

def sweep_args(arm_name, station, payload_kg, traj):
    """what to type after `python bench/sweep.py`, for one arm and one payload.

    fit_bam.py builds  mgr = mass*g*radius  and  J_load = arm_inertia + mass*radius^2,
    and `mass` there is the PAYLOAD only - the printed arm's own weight is a pendulum it
    never hears about.  Both inputs are free, so both are made exact: an effective mass
    that reproduces the true first moment at this radius, and whatever inertia is left
    over once that mass has taken its share.

    The carrier and its filling ride at the station, so they go in as part of the payload;
    the filling's own inertia about its centre is modelled as a solid cylinder of the
    cup's bore, which is 1-2 % of m*r^2 and free to include."""
    m_arm, r_arm, J_arm = about_axis(mp_of(arm_name))
    r = station/1000.0
    cup = mp_of("mass_cup")
    m_cup, I_cup = cup.m, cup.I[1][1]*1e-6            # about its own centre, kg*m^2
    r_bore = (CUP_D/2.0 - CUP_WALL)/1000.0
    m_pay = payload_kg + m_cup
    J_tot = (J_arm + I_cup + m_cup*r*r                # carrier, at the station
             + payload_kg*(r*r + r_bore*r_bore/2.0))  # filling, ditto, as a cylinder
    m_eff = m_pay + m_arm*r_arm/r                     # same gravity torque at this radius
    return dict(arm=arm_name, station_mm=station, payload_kg=payload_kg, traj=traj,
                mass=m_eff, radius=r, arm_inertia=J_tot - m_eff*r*r,
                J_load=J_tot, mgr=m_eff*G*r, m_arm=m_arm, r_arm=r_arm, J_arm=J_arm,
                m_cup=m_cup)

def frame_flex(T_nm):
    """Euler-Bernoulli rotation of the sleeve about the joint axis, integrated over the
    frame's real section as it goes.  Slices the solid, so it follows the shell, the two
    legs and the cable window rather than a nominal rectangle.

    First order, and not a substitute for fea.py: no shear, no stress concentration.  It
    answers one question - is the frame an order of magnitude stiffer than the joint it is
    identifying - and the encoder, when fitted, is referenced to a leg so that whatever is
    left of this is common mode."""
    s = PARTS["bench_frame"][0].val()
    E = PETG["E"]
    z0, z1, n = BASE_T + 4.0, HEAD_Z - 4.0, 24
    th, worst = 0.0, None
    for i in range(n):
        zc = z0 + (z1-z0)*(i+0.5)/n
        t = 2.0
        try:
            sl = s.intersect(md.bxc(-300, 300, -300, 300, zc-t/2, zc+t/2).val())
            a = sl.Volume()/t
            if a < 1e-6: continue
            I = cq.Shape.matrixOfInertia(sl)[1][1]/t                  # about y, mm^4
            c = sl.Center()
            bb = sl.BoundingBox()
            fib = max(bb.xmax - c.x, c.x - bb.xmin)
        except Exception:
            continue
        M = T_nm*1000.0                                               # a pure couple
        th += M/(E*I) * ((z1-z0)/n)
        # printed base-down, the layers are horizontal and this bending stress is normal
        # to them, so it is the inter-layer allowable that applies - the call fea.py makes
        # on the leg parts.
        sf = PETG["s_z"] / (M * fib / I) if I > 0 else 0.0
        if worst is None or sf < worst[0]: worst = (sf, zc, a, I)
    return math.degrees(th), worst

def tipover(cfg):
    """(restoring N*m from the frame's own weight, N*m demanded with the arm out at
    NEED_DEG).  The portal hangs its load rather than cantilevering it, so unlike an
    L-shaped stand this comes out positive - but a free swing is a dynamic reaction."""
    mp = MP.of(PARTS["bench_frame"][0], rho("bench_frame")*1e-6)
    edge = BASE_X1
    restore = mp.m*G*(edge - mp.c[0])/1000.0
    reach = cfg["radius"]*1000.0*math.sin(math.radians(NEED_DEG))
    demand = cfg["mass"]*G*max(0.0, reach - edge)/1000.0
    return restore, demand


def stance():
    """where the foot plate goes under the tower: the contact patch of the leg in the
    robot's OWN stance pose, taken off the posed solid rather than off FOOT_Z."""
    if not md.PARTS: md.build()
    bb = md.posed(md.PARTS["foot"][0], "shin").translate((TOWER_DX, 0, 0)).val().BoundingBox()
    return ((bb.xmin+bb.xmax)/2.0, (bb.ymin+bb.ymax)/2.0, bb.zmin - PLATE_T)

def scenes():
    grey, wht, blk = (0.62, 0.65, 0.70), (0.85, 0.86, 0.88), (0.16, 0.16, 0.18)
    alu, org = (0.72, 0.76, 0.82), (0.90, 0.58, 0.16)
    srv = (at_bench(md.servo_dummy()).val(), blk)
    hub = (at_bench(md.hubs()).val(), alu)
    frame = (PARTS["bench_frame"][0].val(), grey)
    brg = (PARTS["enc_bridge"][0].val(), wht)
    cap = (PARTS["enc_magnet_cap"][0].val(), blk)
    out = [("short", [frame, srv, hub, brg, cap, (PARTS["arm_short"][0].val(), org)]),
           ("long", [frame, srv, hub, brg, cap, (PARTS["arm_long"][0].val(), org),
                     (PARTS["mass_cup"][0].val(), blk)])]
    hb, th, sh, ft = md.build()
    dx = (TOWER_DX, 0.0, 0.0)
    leg = [(PARTS["leg_tower"][0].val(), grey)]
    for w, kind, col in ((hb, "hip", org), (th, "thigh", wht), (sh, "shin", wht),
                         (ft, "shin", blk)):
        leg.append((md.posed(w, kind).translate(dx).val(), col))
    for i, kind in enumerate(("hip", "thigh", "shin")):
        leg.append((md.posed(md.mv(md.servo_dummy(), md.JOINTS[i][1]), kind)
                    .translate(dx).val(), blk))
        leg.append((md.posed(md.mv(md.hubs(), md.JOINTS[i][1]), kind)
                    .translate(dx).val(), alu))
    leg.append((PARTS["foot_plate"][0].translate(stance()).val(), grey))
    leg.append((PARTS["cell_riser"][0].translate(stance()).val(), grey))
    out.append(("leg", leg))
    return out

def render_all():
    from render import render
    for name, sc in scenes():
        f = (0, 0, AXIS_Z-60) if name in ("short", "long") else (TOWER_DX, md.LEG_Y, -100)
        for view, cam in (("iso", (460, -620, 300)), ("side", (10, -900, 20))):
            render(sc, os.path.join(OUT, f"view_{name}_{view}.png"),
                   tuple(c + o for c, o in zip(cam, f)))


def main(do_render=False):
    for d in ("step", "stl"): os.makedirs(os.path.join(OUT, d), exist_ok=True)
    build()
    rows = []
    print("\n  part               qty  for       volume   est.mass    print bbox (mm)")
    for name, (wp, qty, grp, note) in PARTS.items():
        shp = wp.val()
        ok = shp.isValid()
        cq.exporters.export(wp, os.path.join(OUT, "step", f"{name}.step"))
        ax, ang = PRINT_ORIENT[name]
        pw = wp.rotate((0, 0, 0), ax, ang) if ang else wp
        bb = pw.val().BoundingBox()
        pw = pw.translate((-bb.xmin, -bb.ymin, -bb.zmin))
        cq.exporters.export(pw, os.path.join(OUT, "stl", f"{name}.stl"),
                            tolerance=0.02, angularTolerance=0.15)
        bb = pw.val().BoundingBox()
        v = shp.Volume()/1000.0
        m = v*rho(name)
        rows.append({"part": name, "qty": qty, "for": grp, "volume_cm3": round(v, 1),
                     "est_mass_g": round(m, 1),
                     "print_bbox_mm": [round(bb.xlen, 1), round(bb.ylen, 1), round(bb.zlen, 1)],
                     "valid": ok, "note": note})
        print(f"  {name:17s} x{qty}  {grp:8s} {v:7.1f} cm3 {m:7.1f} g   "
              f"{bb.xlen:6.1f} x {bb.ylen:6.1f} x {bb.zlen:6.1f}"
              f"{'' if ok else '   !! INVALID'}")

    bad = interference()
    for na, nb, v in bad:
        print(f"  !! INTERFERENCE  {na} x {nb}  {v:.1f} mm3")
    if not bad:
        print(f"  frame clear: {' / '.join(STATIC)} share no solid")
    try:
        v = PARTS["foot_plate"][0].val().intersect(PARTS["cell_riser"][0].val()).Volume()
    except Exception:
        v = 0.0
    if v > md.INTERF_TOL:
        print(f"  !! CELL BRIDGED  foot_plate x cell_riser share {v:.1f} mm3 - the cell"
              f" would read a fraction of the force, and look like it works")
    else:
        print(f"  cell floats: platform clears the riser by {PLATE_GAP+CELL_H:.1f} mm at"
              f" its underside, {CELL_H:.1f} mm at the spacer - which is the cell itself")

    # the travel, against what robot/bench actually needs
    print(f"  swing (5 deg steps, real solids; 0 = hanging, + = toward +x."
          f"  sweep.py aborts at +-{NEED_DEG:.0f}):")
    sw = {}
    for nm in SWINGING:
        sw[nm] = sweep(nm, ("bench_frame", "enc_bridge"))
        lo, hi = sw[nm]
        ok = (lo <= -NEED_DEG and hi >= NEED_DEG)
        print(f"    {nm:10s} free {lo:+4d} .. {hi:+4d} deg"
              f"{'' if ok else f'   !! the stand is the limit, not the software'}")

    if PARTS.get("enc_bridge"):
        g = mag_to_ic()
        flag = "" if 0.5 <= g <= 3.0 else "   !! outside the AS5600's 0.5..3.0 mm window"
        print(f"  encoder:     magnet face -> IC {g:.1f} mm, slotted +-{ENC_SLOT:.1f} mm"
              f" ({ENC_GAP:.1f} air + {ENC_LEDGE:.1f} ledge){flag}   [optional]")

    # ------------------------------------------------------------------ the arms
    print(f"\n  arms, and what to tell ../robot/bench/sweep.py."
          f"  Priors from {_PRIOR['src']}:")
    print(f"    tau_c {_PRIOR['tau_c']:.3f} N*m, J_m {_PRIOR['J_m']:.2e} kg*m2,"
          f" release at {SWEEP_START:.2f} rad")
    print(f"  {'cm3 in the cup':>16s}  {cup_volume():.0f}"
          f"   (~{cup_volume()*4.7/1000:.2f} kg of steel shot,"
          f" ~{cup_volume()*7.0/1000:.2f} kg of lead)")
    cfgs = [sweep_args("arm_short", ARM_SHORT_ST[0], 0.25, "freeswing"),
            sweep_args("arm_short", ARM_SHORT_ST[1], 0.25, "freeswing"),
            sweep_args("arm_long",  ARM_LONG_ST[0],  0.25, "hold+step+chirp"),
            sweep_args("arm_long",  ARM_LONG_ST[1],  0.40, "hold+step+chirp")]
    for c in cfgs:
        print(f"\n    {c['arm']} @ {c['station_mm']:.0f} mm, {c['payload_kg']:.2f} kg in "
              f"the cup ({c['m_cup']*1000:.0f} g) -> --traj {c['traj']}")
        # --arm-inertia is emitted in the `=` form on purpose: it can be negative (see the
        # module docstring), and argparse reads a leading minus as the next option.
        # `--arm-inertia -1.0e-04` fails with "expected one argument"; this does not.
        print(f"      --mass {c['mass']:.4f} --radius {c['radius']:.3f} "
              f"--arm-inertia={c['arm_inertia']:.3e}")
        drive = c["mgr"]*math.sin(SWEEP_START) - _PRIOR["tau_c"]
        share = c["J_load"]/(c["J_load"] + _PRIOR["J_m"])
        # The drive bound is fit_bam.py's own: it declines the free swing outright when
        # m*g*r*sin(q0) <= tau_c ("Use a longer one").  2x is the margin over a prior
        # that is itself a guess.
        f1 = "" if drive > _PRIOR["tau_c"] else "   !! under 2x tau_c - it may not swing"
        print(f"      m*g*r {c['mgr']:.4f} N*m -> drive at release {drive:+.4f} N*m{f1}")
        # ... and the inertia bound applies to the free swing alone: J_m = J_tot - J_load
        # is a difference of two numbers there.  A hold run wants a big load and does not
        # care, which is the whole reason there are two arms.
        f2 = ("" if c["traj"] != "freeswing" or share < 0.5 else
              "   !! J_load swamps J_m - the difference is noise")
        print(f"      J_load {c['J_load']:.3e} kg*m2, {100*share:.0f} % of"
              f" J_load+J_m{f2}")
    m_arm_s, r_s, J_s = about_axis(mp_of("arm_short"))
    m_arm_l, r_l, J_l = about_axis(mp_of("arm_long"))
    print(f"\n    the printed arms themselves: short {m_arm_s*1000:.0f} g at r ="
          f" {r_s*1000:.0f} mm, J = {J_s:.2e} kg*m2;"
          f" long {m_arm_l*1000:.0f} g at r = {r_l*1000:.0f} mm, J = {J_l:.2e} kg*m2")
    print(f"    WEIGH BOTH - those are solid volume x md.PRINT_FILL_MEAN, and the arm's"
          f" own first moment is {100*m_arm_l*r_l/(cfgs[-1]['mass']*cfgs[-1]['radius']):.0f}"
          f" % of the long arm's total at its outer station")

    th, worst = frame_flex(md.SERVO_STALL_NM)
    print(f"\n  frame:       {th:.3f} deg at stall ({md.SERVO_STALL_NM:.2f} N*m),"
          f" E = {PETG['E']:.0f} MPa")
    if worst:
        print(f"               thinnest section at z = {worst[1]:.0f}: A = {worst[2]:.0f} mm2,"
              f" I = {worst[3]/1000:.0f} x10^3 mm4, inter-layer SF ~{worst[0]:.1f}")
    restore, demand = tipover(cfgs[-1])
    tag = "" if restore > 2*demand else "  !! bolt it down"
    print(f"  tip-over:    restores {restore:.2f} N*m against {demand:.2f} N*m with the"
          f" long arm out at {NEED_DEG:.0f} deg{tag}")

    # AXIS_Z is only right if the long arm and its payload clear the plate at the bottom
    # of the swing.  Measured off the real solids, both of them, rather than assumed.
    bb = PARTS["arm_long"][0].val().BoundingBox()
    reach = max(AXIS_Z - bb.zmin, math.hypot(CUP_D/2.0, 0.0) + ARM_LONG_ST[-1])
    gap = (AXIS_Z - reach) - BASE_T
    if gap < ARM_CLEAR:
        print(f"  !! ARM CLEARANCE  the long arm reaches {reach:.0f} mm and leaves"
              f" {gap:.0f} mm over the plate - raise AXIS_Z")
    else:
        print(f"  arm clearance: {gap:.0f} mm under the long arm and its carrier at q = 0"
              f" (reach {reach:.0f} mm, axis at {AXIS_Z:.0f})")

    sx, sy, sz = stance()
    print(f"  leg tower:   roll axis {-sz-PLATE_T:.0f} mm over the foot plate's top face,"
          f" x {sx:+.0f} y {sy:+.0f} from the flange   [not robot/bench's]")
    m = tower_bolt_margin()
    if m < TOWER_BOLT_R - CLEAR_R:
        print(f"  !! TOWER BOLTS  nearest is {m:+.1f} mm outside the roll fork's sweep")
    else:
        print(f"               flange bolts {m:+.1f} mm clear of the roll fork's"
              f" {CLEAR_R:.1f} mm sweep, so their nuts miss the leg")

    per = {}
    for r_ in rows:
        per[r_["for"]] = per.get(r_["for"], 0.0) + r_["est_mass_g"]*r_["qty"]
    print("\n  printed mass  " + ",  ".join(f"{k} ~{v:.0f} g" for k, v in per.items()))

    print("\n  buy:")
    for item, qty, grp, note in BUY:
        print(f"    [{grp:8s}] {str(qty):>6s}  {item}")
        if note: print(f"                        {note}")

    with open(os.path.join(OUT, "bom.json"), "w") as f:
        json.dump({"parts": rows,
                   "buy": [{"item": i, "qty": q, "for": g, "note": n}
                           for i, q, g, n in BUY],
                   "swing_deg": {k: list(v) for k, v in sw.items()},
                   "sweep_args": [{k: v for k, v in c.items()} for c in cfgs],
                   "bench": {"axis_z": AXIS_Z, "need_deg": round(NEED_DEG, 1),
                             "cup_cm3": round(cup_volume(), 1),
                             "frame_flex_deg": round(th, 4),
                             "encoder_gap_mm": round(mag_to_ic(), 2)},
                   "priors": _PRIOR},
                  f, indent=2)
    if do_render:
        render_all()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true",
                    help="also write out/bench/view_*.png - and look at them")
    main(do_render=ap.parse_args().render)
