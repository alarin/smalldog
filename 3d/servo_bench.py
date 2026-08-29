#!/usr/bin/env python
"""
servo_bench.py - printed test stands that MEASURE the ST3215, so a motion search can
replace the analytic IK.

WHY THIS EXISTS
---------------
`../ros2/smalldog_walker` plans with a closed-form leg IK: a foot target goes in, three
joint angles come out, and the servo is assumed to follow.  Swapping that for an
enumeration - generate candidate joint trajectories, score them, keep the feasible ones -
changes what the model has to know.  IK needs link lengths, which this repo measures to
0.1 mm.  A search needs the *actuator*, and every actuator number in this repo is a
catalogue figure or a guess:

    md.SERVO_STALL_NM    = 2.94    # "30 kg*cm at 12 V", vendor
    md.SERVO_NOLOAD_RADS = 4.71    # "0.222 s / 60 deg at 12 V", vendor, unloaded
    md.SERVO_KG          = 0.060   # marked **verify** in mini_dog.py - vendor figure
    export_sim.MJ_KP / MJ_DAMPING / MJ_ARMATURE / MJ_FRICTIONLOSS
                                   # "Not measured - the ST3215 gearbox is a black box"

A search filters every candidate against exactly those numbers, so a search built on them
is a search over a robot that does not exist.  Worse, two of the quantities it needs are
not in the model at all: the joint's compliance (the printed fork and sleeve are PETG, and
the thrust-clamp note in mini_dog.py estimates +-0.9 deg of knock per joint - a
calculation from the print clearance, never measured), and the map from the servo's own
reported load to real newton-metres, which is what `smalldog_walker/contact.py` wants to
use as a foot-contact signal.

These stands produce those numbers.  Three of them, nine printed parts:

  A  torque bench   bench_base + torque_arm + protractor (+ the encoder pair)
                    static: stall torque, joint compliance, backlash, load-report
                    calibration, thermal duty
  B  inertia bench  bench_base + inertia_disc (+ the encoder pair)
                    dynamic: no-load speed, step response at known J, tracking lag
  C  leg rig        leg_tower + foot_plate + cell_riser
                    a whole leg at its real geometry over a load cell: does the motion
                    the search picked actually do what it said?

WHAT EACH MEASUREMENT IS FOR
----------------------------
  #   stand  measurement                              lands in
  1   A      stall torque at 12 V, per servo          md.SERVO_STALL_NM (**verify** today)
  2   A      angle vs applied torque -> joint rate    new: the search's deflection model.
                                                      Includes the printed fork+sleeve,
                                                      which is the point - the gearbox
                                                      alone is not what the foot hangs off
  3   A      load-reversal loop -> backlash+deadband  the search's angular resolution.
                                                      Enumerating below it is enumerating
                                                      noise
  4   A+C    reported load counts vs real N*m         smalldog_walker/contact.py's
                                                      contact_threshold - masses on A,
                                                      the load cell under a real foot on C
  5   A      hold torque vs case temperature vs time  the duty-cycle cost term.  Nothing
                                                      in this repo models it, and a search
                                                      will happily pick a gait that cooks
  6   B      no-load speed at 12 V                    md.SERVO_NOLOAD_RADS
  7   B      step response at 3+ known J              export_sim.MJ_KP, MJ_DAMPING,
                                                      MJ_ARMATURE, MJ_FRICTIONLOSS
  8   B      goal-position tracking lag at 100 Hz     the enumeration's time step - the
                                                      walker publishes at 100 Hz and
                                                      assumes the joint is there
  9   A+B    the torque-speed line between 1 and 6    the feasibility filter itself
  10  C      candidate trajectory -> foot path+force  end-to-end check of 1..9

Numbers 1..9 are per-servo, and they are not the same servo to servo; run at least three
and quote a spread, because the search's margin has to cover the worst one on the robot.

WHAT THIS FILE IS NOT
---------------------
It is not the planner.  Nothing here generates or scores motions; it produces the CAD and
the BOM for the hardware that measures the constants such a planner needs, and the
measurement -> parameter map above.  The planner is a separate change and it should not
start before there are numbers to feed it.

RELATIONSHIP TO mini_dog.py
---------------------------
Every servo dimension, fit, clearance, nut pocket, sleeve, fork, density and mass here is
imported from `mini_dog`.  Nothing is re-typed.  That is not tidiness: the whole value of
these stands is that they hold the servo the way the ROBOT holds it - the same sleeve, the
same thrust clamp, the same fork on the same stock aluminium hubs - so what they measure
is this robot's joint and not a servo in a vice.  A stand with its own copy of S_L would
drift away from the robot and start measuring a different joint, silently.

Bench frame (stands A and B): +X = the arm at its zero, horizontal; +Y = the joint axis;
+Z = up.  Origin on the base plate's underside, under the joint axis.  Units mm, like
everything else here.
"""
import math, os, json, argparse
import cadquery as cq

import mini_dog as md
import fea                       # materials only - E and the inter-layer allowable
from export_sim import MP        # mass properties; nothing here reimplements them

OUT = os.path.join(md.OUT, "bench")

PETG = fea.MATERIALS["PETG"]
G    = 9.81

# =====================================================================================
# 1. bench parameters
# =====================================================================================
# The joint axis height.  Set by the arm, not chosen: a 2 kg mass at the outermost load
# station has to be exactly full-scale torque (see ARM_R below), the arm has to be able to
# swing that station past straight-down, and the tip has to clear the base plate.
AXIS_Z    = 175.0

# THE ONE CLEARANCE RULE ON THIS BENCH.  mini_dog's invariant "nothing goes into
# 23 < r < 34 of a joint axis over the sleeve's length" is the distal fork's spine
# sweeping that annulus, and it applies here exactly as it does on the robot - the fork
# IS md.fork().  On the robot the cradle gets around it by coming in from a sector the
# roll fork never reaches.  This bench needs a large sweep in every direction the arm can
# point, so it uses the other exit: a plane at x = -(SPINE_R1 + margin) is outside r = 34
# at EVERY height, so anything behind it is unconditionally clear whatever the arm does.
# That plane is the front face of the buttress and it is why the stand is an L and not a
# yoke.  A yoke cannot exist: the sleeve ends at |y| = 16.5 and the fork arms start at
# |y| = 17.9, so there is nowhere for a second bearing wall to reach the sleeve from.
# The fork's own outermost point, which is what has to get past: the spine's corner at
# (SPINE_R1, SPINE_W/2).  mini_dog rounds this to "34" in prose; here it is computed,
# because 2 mm of clearance on it is the entire margin the front face has.
CLEAR_R   = math.hypot(md.SPINE_R1, md.SPINE_W/2)          # 34.01
POST_X    = -(CLEAR_R + 2.0)              # buttress front face.  A plane at this x is
                                          # tangent to the fork's sweep circle at the axis
                                          # height and further from it everywhere else, so
                                          # everything behind it is clear at every angle.
POST_HY   = 22.0                          # buttress half-width.  < 28, so the inertia
                                          # disc at |y| >= 28 clears it (stand B).
POST_TOP  = AXIS_Z + md.S_W/2 + md.SLEEVE_W        # level with the sleeve's top face
# The front face runs straight down to the plate and does NOT rake forward under the
# joint, which it structurally would like to.  A rake from the axis height out to the
# plate's front edge costs the arm its downward sweep: the tip is at r = 156 and it fouls
# such a face at -74 deg, against the +-100 the protractor is drawn for.  The footprint
# that a rake would have bought comes from the base plate instead, which is 8 mm of PETG
# doing three jobs - footprint, bolt pattern and ballast.
POST_BACK_TOP, POST_BACK_BOT = -74.0, -122.0       # the rear face's rake
POST_WALL = 4.5                           # shell wall.  The cavity is also the cable run:
                                          # the servo's connector points into it.

# Hole naming follows mini_dog's: a *_CLR is a RADIUS and goes straight into cyl(), a
# *_D or *_HEAD is a DIAMETER and is halved at the point of use.  Getting that backwards
# is a hole at twice or half its size that still builds and still looks right.
BASE_T    = 8.0                           # base plate
BASE_X0, BASE_X1 = -132.0, 62.0
BASE_HY   = 65.0
BASE_R    = 6.0                           # plate corner radius
M2_CLR    = 1.15                          # the AS5600's own two screws
M4_CLR    = 2.30                          # bolting the plate down - see the tip-over
M4_HEAD   = 10.0                          # check in main(); this stand MUST be held down.
                                          # Sized for a WASHER, not the head: 4 mm of
                                          # plate under a bare M4 head is not bearing area
BASE_CBORE = 4.0                          # ... how deep, leaving BASE_T - this as floor
BOLT_AT   = ((-118.0, 52.0), (-118.0, -52.0), (44.0, 52.0), (44.0, -52.0),
             (-36.0, 56.0), (-36.0, -56.0))
BALLAST   = (-126.0, -62.0, 26.0)         # x0, x1, wall height of the ballast tray - the
                                          # alternative to bolting down, sized in main()
BALLAST_Y = -22.0                         # ... and its -y wall stops here rather than at
                                          # the plate's edge, because the protractor's
                                          # strut lands on the plate at y = -30, right
                                          # through where a symmetric tray would be

# Load arm.  ARM_R is derived, not styled: it is the radius at which one 2 kg mass is
# exactly the catalogue stall torque, so the outermost station is full scale by
# construction and the operator has a round number to hang.
ARM_MASS_KG = 2.0
ARM_R       = md.SERVO_STALL_NM / (ARM_MASS_KG * G) * 1000.0        # -> 149.8 mm
ARM_STATIONS = (ARM_R/3.0, 2.0*ARM_R/3.0, ARM_R)                    # 50 / 100 / 150 mm
ARM_W0, ARM_H0 = md.SPINE_W, 24.0         # root section: as wide as the fork spine
ARM_W1, ARM_H1 = 14.0, 14.0               # tip section
ARM_HOLE    = md.M3_CLR                   # each station is an M3 cross hole: an S-hook on
ARM_PAD     = 3.0                         # a bolt through it seats the same way every time
ARM_TIP     = 6.0                         # beam runs ARM_TIP past the last station

# Protractor.  A backup for the encoder and the thing you set zero against with no
# electronics on the bench.  It lives entirely outboard of the fork's -y arm, so it MAY
# cross the swept annulus - it is a flat plate in a plane the fork never reaches, which is
# also why it prints flat in one piece with no support.
#
# It stands on the base plate rather than on the buttress, and that is a deliberate step
# down in rigour from the encoder bridge: at PROT_R1 one degree is about a millimetre, so
# it reads to a quarter of a degree at best and the 0.03 deg the stand flexes under full
# load (main() computes it) is far below its resolution.  The encoder is the instrument;
# this is the zero reference and the check that the encoder has not lost a turn.
PROT_Y0, PROT_T = -32.0, 4.0              # -32 .. -28; the fork's -y face is at -21.9
PROT_R1     = 62.0                        # 1 deg is 1.08 mm at the outer edge
PROT_SWEEP  = 90.0                        # +- deg.  Checked against the real sweep in main()
PROT_TICK   = (0.6, 4.0, 6.0, 9.0)        # width, len(1 deg), len(5 deg), len(10 deg)
PROT_HUB    = 40.0                        # hub disc: ties the scale sector to the strut
# Both parts that live outboard of the fork's -y arm have to be bored on the axis, or the
# four M2.5 that hold that arm to the passive hub cannot be reached: their heads are on
# the arm's -y face and the driver comes in along -y, straight through here.  @14 bolt
# circle, M2.5 heads out to r = 9.25, so anything over @21 clears - and it is the same
# hole in both parts for the same reason.
EYE_D       = 26.0
PROT_LIGHT  = 13.0                        # lightening holes in the web,
PROT_LIGHT_R = (22.0, 38.0)               # ... on these two rings
# THE POINTER READS FROM OUTSIDE, AND THAT IS WHAT SHAPES THIS PART.
# The ticks are engraved on the plate's OUTBOARD face, so the pointer has to be outboard
# of the plate or it is behind the thing it is pointing at.  Outboard means its bracket
# has to cross the plate's plane, and the only radius where it can is outside the plate's
# own rim - so the bracket hooks over at r > PROT_R1 and the blade comes back inward over
# the scale.  That in turn is why the plate cannot be held up by a leg under the joint:
# the bracket sweeps the annulus PROT_R1..POINT_R[1] over the whole +-PROT_SWEEP.  The
# strut instead leaves in the wedge the arm never reaches, down and back past the
# buttress, which it can do because this whole plane is outboard of everything.
PROT_STRUT  = (-120.0, -106.0)            # the strut's angular window, deg from +x.  The
                                          # pointer bracket's outer corner reaches -100.3
                                          # at full down-sweep; this leaves 5.7 deg.
PROT_FOOT_T = 7.0                         # the foot's thickness, standing on the plate
PROT_BOLT   = (-58.0, -86.0)              # 2 x M3 down into the plate's own nut slots

POINT_Y     = (-37.0, -34.5)              # pointer blade: 2.5 mm thick, 2.0 mm outboard
POINT_R     = (56.0, 78.0)                # of the plate's face - tip inside the ticks'
POINT_BR    = 66.0                        # inner end, bracket hooking over the rim at
POINT_TIP   = 0.8                         # POINT_BR, which is 4 mm outside it

# Encoder.  An AS5600 over a diametric magnet on the axis, and it is the only instrument
# here that is not a mass or a ruler.  It exists because the number that matters most -
# joint compliance - is the difference between what the servo's own encoder says and where
# the link actually is, so the servo cannot be asked to measure it.
#
# The magnet rides on the fork's +y arm; the sensor is on a bridge off the buttress.  The
# bridge is deliberately NOT referenced to the table: it grows off the buttress within
# ~40 mm of the sleeve, so whatever the stand does under load, the sensor and the servo
# case do most of it together and the reading stays a joint angle rather than a stand
# angle.  That is what lets a printed stand measure a printed joint at all.
MAG_D, MAG_H = 6.0, 2.5                   # standard N35, DIAMETRICALLY magnetised - an
                                          # axial one reads as a constant and looks broken
MAG_FIT    = -0.10                        # light press; the fork arm backs it up
ENC_GAP    = 1.5                          # cap face -> bridge face.  The magnet-to-IC
                                          # distance that follows is reported in main()
                                          # and has to land in the AS5600's 0.5..3 mm.
ENC_PCB    = (12.7, 12.7, 1.6)            # a bare AS5600 breakout                **verify**
ENC_PCB_HOLES = 10.0                      # ... 2 x M2 at this pitch               **verify**
ENC_LEDGE  = 1.0                          # plastic between the board and the magnet
ENC_WIN    = 9.0                          # ... with a window this wide for the IC itself
ENC_SLOT   = 1.5                          # the bridge's mounting slots run +-this in y,
                                          # which is how ENC_GAP is actually set: fit it,
                                          # then read the AS5600's AGC register and slide
                                          # the bridge until the gain sits mid-range.  At
                                          # the inboard end of the slot the bridge touches
                                          # the cap, which is a hard stop and an obvious
                                          # one; main() reports the window it leaves.
ENC_FOOT_T = 5.0                          # the bridge's foot, flat on the buttress' pad
ENC_BOLT   = ((-44.0, 12.0), (-56.0, 12.0))        # x, y of its two M3 - both clear of the
                                          # +-8 mm cable channel that runs up the pad
PAD_T      = 12.0                         # the pad itself: the buttress' shell capped
                                          # solid, so those two nuts have somewhere to sit
PAD_L      = 26.0
CAP_R      = md.ARM_R                     # magnet cap: the fork arm's own disc
CAP_T      = MAG_H                        # magnet flush both ways, trapped by the arm
# The cap is bolted on by the SAME four M2.5 that hold the fork arm to the driven hub, so
# they get longer by exactly the cap's thickness.  How much longer is not a free choice:
# past the hub's own thread the screw bottoms out on the servo case, and the vendor FAQ
# says that stalls and burns the servo - the same rule that fixes the robot's M2.5 x 6.
# Clearance to pass is CAP_T + ARM_T; thread available is HUB_T_TOP.
CAP_SCREW  = max(L for L in (6, 8, 10, 12) if L <= CAP_T + md.ARM_T + md.HUB_T_TOP)

# Inertia rotor (stand B).  A disc outboard of the fork's -y arm - the same plane the
# protractor uses, and the two stands never run at once.  Its own inertia is the baseline;
# M8 bolts on the pocket circle step it up in known increments.  Weighed, not assumed:
# main() reports J for the printed disc from the real solid and the increment per bolt
# from a weighed bolt mass.
DISC_Y0, DISC_T = -34.0, 6.0
DISC_R     = 75.0
DISC_HUB_R = 20.0                         # inside CLEAR_R's inner edge (23) - the hub
                                          # crosses the fork's own plane, so it must be
DISC_BC, DISC_N = 120.0, 8                # M8 pockets, filled symmetrically only
DISC_HOLE  = 8.6
DISC_LIGHT, DISC_LIGHT_BC = 22.0, 84.0    # ... and the web between hub and pockets, gone:
                                          # what is left of the rotor's own inertia sits
                                          # in its rim, where the fill factor matters least
BOLT_M8_G  = 17.0                         # M8x20 + nut, nominal - **weigh yours**

# Leg rig (stand C).  Not a second servo bench: it is the whole leg at the robot's own
# geometry, because the thing a motion search gets wrong is never one joint - it is the
# knee carrying the shin's inertia while the hip is somewhere the single-joint bench never
# put it.  The cradle is md.roll_module() itself, moved, so the leg hangs off exactly the
# structure it hangs off on the robot.
TOWER_T    = 10.0                         # flange plate, bolted to a board or an extrusion
TOWER_HY, TOWER_Z = (-46.0, 70.0), 46.0   # flange outline, around the cradle's own root
M6_CLR     = 3.40
# The four bolts go through the flange from the board's far side, with their nuts on the
# flange's INBOARD face - so each nut stands ~8 mm proud into a space the hip-roll fork
# sweeps.  That fork's arms come to within 5.1 mm of the flange and sweep r <= 34 about
# the roll axis, so the bolt centres have to stand outside that circle with room for the
# nut.  main() computes the margin off the real roll axis rather than trusting these.
TOWER_BOLT = ((-36.0, 36.0), (-36.0, -36.0), (58.0, 36.0), (58.0, -36.0))
TOWER_BOLT_R = 40.0                       # ... the radius they must clear

# Foot plate: the platform half of a 5 kg straight-bar load cell, plus the riser that
# takes its fixed end down to the board.  The cell is what turns the servo's load counts
# into newtons (measurement #4) - the one number contact.py needs and nobody has.
# The cell is a bending beam and it only reads if its two ends are carried by two bodies
# that do not touch: the FREE end hangs off the platform on a spacer, the FIXED end stands
# on the riser, and the air between the platform's underside and everything below it is
# what the beam bends into.  Get that wrong - bridge the two, or let the platform land on
# the riser - and it reads a fraction of the force, or nothing, and looks like it works.
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
# 2. helpers - bench frame, and the servo in it
# =====================================================================================
# The servo, placed by the SAME frame() mini_dog uses for its own joints.  local +Z (the
# driven hub) -> bench +Y, so the axis is +Y; local -X (the link direction) -> bench +X, so
# the fork's spine and everything bolted to it point along +X at zero.  The case therefore
# lies behind the axis with its connector at x = -41.1, pointing into the buttress' cavity.
BENCH_LOC = md.frame((0.0, 0.0, AXIS_Z), xdir=(-1, 0, 0), zdir=(0, 1, 0))

def at_bench(wp):  return md.mv(wp, BENCH_LOC)

def prism_y(pts, y0, y1):
    """polygon (x, z) in the bench side view, extruded from y0 to y1."""
    return md.W(cq.Workplane("XZ").polyline(pts).close()
                .extrude(y1 - y0).val()).translate((0, y1, 0))

def disc_y(r, y0, y1, c=(0.0, AXIS_Z)):
    return md.cyl(r, y1 - y0, (c[0], y0, c[1]), axis=(0, 1, 0))

def arc_sector(r0, r1, a0, a1, y0, y1, step=1.0):
    """annular sector about the bench axis, between angles measured from +X toward +Z."""
    n = max(2, int(round((a1 - a0) / step)))
    out = [(r1*math.cos(math.radians(a0 + (a1-a0)*i/n)),
            AXIS_Z + r1*math.sin(math.radians(a0 + (a1-a0)*i/n))) for i in range(n+1)]
    inn = ([(0.0, AXIS_Z)] if r0 <= 1e-9 else
           [(r0*math.cos(math.radians(a1 - (a1-a0)*i/n)),
             AXIS_Z + r0*math.sin(math.radians(a1 - (a1-a0)*i/n))) for i in range(n+1)])
    return prism_y(out + inn, y0, y1)

def polar(r, a):
    """a point at radius r, angle a deg from +X toward +Z, in the bench side view."""
    return (r*math.cos(math.radians(a)), AXIS_Z + r*math.sin(math.radians(a)))

def taper(u):
    """0 at the arm's root, 1 at its tip -> (half-width in y, half-height in z)."""
    return (ARM_W0 + (ARM_W1-ARM_W0)*u)/2.0, (ARM_H0 + (ARM_H1-ARM_H0)*u)/2.0

def beam_x(r0, r1, n=14):
    """the tapered load beam, lofted along +X from radius r0 to r1 on the joint axis.

    Lofted in the XY plane along +Z and then turned into the bench frame, because
    md.rrect() draws in XY - the same detour md.shin_beam() takes for the same reason.
    Sections are rounded rectangles, so the taper cannot invent a bulge between them."""
    secs = []
    for i in range(n+1):
        u = i/n
        hy, hz = taper(u)
        secs.append(md.rrect(2*hz, 2*hy, min(hy, hz)*0.45, (0, 0, r0 + (r1-r0)*u)))
    return md.loft(secs).rotate((0, 0, 0), (0, 1, 0), 90).translate((0, 0, AXIS_Z))


# =====================================================================================
# 3. PART: bench_base  (stands A and B)
# =====================================================================================
def bench_base(hollow=True):
    """base plate + buttress + the robot's own sleeve, one printed part.

    hollow=False returns the same stand with the buttress' cavity and its ballast tray
    filled in.  That is the SILHOUETTE, and it is what the sweep check runs against: the
    cavity is a void the arm cannot reach through, but a per-angle interference scan is
    static, so against the real hollow part the arm tunnels into it and the scan reports a
    sweep that does not exist.  It reported -110 deg that way, off a joint that stops at
    -105.

    The sleeve is md.sleeve() unmodified - full length, cable window, thrust-clamp lug and
    both nut channels - so the servo is clamped on this bench exactly as it is clamped in
    a leg, down to the two M3 jack screws that take the 0.35 mm of case play out in
    thrust.  Measuring a joint held any other way measures a different joint.

    The buttress meets it only behind POST_X, which is outside the fork's swept annulus at
    every height (see CLEAR_R), and its front face stays there all the way down to the
    plate - see the note on POST_BACK_TOP for what raking it forward would cost."""
    # base plate, with a rounded outline and the bolt-down pattern
    plate = md.W(cq.Workplane("XY").add(
        cq.Face.makeFromWires(md.rrect(BASE_X1-BASE_X0, 2*BASE_HY, BASE_R,
                                       ((BASE_X0+BASE_X1)/2, 0, 0))))
        .wires().toPending().extrude(BASE_T).val())
    for bx, by in BOLT_AT:
        plate = plate.cut(md.cyl(M4_CLR, BASE_T+2, (bx, by, -1)))
        plate = plate.cut(md.cyl(M4_HEAD/2, BASE_CBORE, (bx, by, BASE_T-BASE_CBORE)))
    # ballast tray: the alternative to bolting down.  main() says how much it needs.
    bx0, bx1, bh = BALLAST
    tray = md.bxc(bx0, bx1, BALLAST_Y, BASE_HY-3.0, BASE_T, BASE_T+bh)
    if hollow:
        tray = tray.cut(md.bxc(bx0+4.0, bx1-4.0, BALLAST_Y+4.0, BASE_HY-7.0,
                               BASE_T-1, BASE_T+bh+1))
    s = plate.union(tray)

    # buttress: an outer prism minus an inner one, i.e. a closed box section.  Closed
    # matters - this is the member that carries the joint torque to the plate, and an open
    # channel of the same outline is an order of magnitude softer in the direction the
    # torque twists it.
    out = [(POST_X, POST_TOP), (POST_X, BASE_T),
           (POST_BACK_BOT, BASE_T), (POST_BACK_TOP, POST_TOP)]
    inn = [(POST_X-POST_WALL, POST_TOP+1), (POST_X-POST_WALL, BASE_T+POST_WALL),
           (POST_BACK_BOT+13.0, BASE_T+POST_WALL), (POST_BACK_TOP+11.0, POST_TOP+1)]
    post = prism_y(out, -POST_HY, POST_HY)
    if hollow:
        post = post.cut(prism_y(inn, -POST_HY+POST_WALL, POST_HY-POST_WALL))
        # one internal rib on the axis' own plane, where the torque comes in
        post = post.union(prism_y(out, -3.0, 3.0))
    s = s.union(post)

    # the sleeve, and its collar into the buttress' front face.  x <= POST_X is
    # unconditionally clear of the fork, so the collar may wrap the sleeve there.
    slv = at_bench(md.sleeve())
    # It thickens the shell's front wall from POST_WALL to ~10 mm right around the
    # sleeve's rear end, which is where the joint's whole moment enters the buttress.  It
    # stops at POST_TOP: the pad's top face has to be one flat plane for the encoder
    # bridge to sit on, and a collar standing 6 mm proud of it is what the interference
    # check caught the first time this was built.
    collar = md.bxc(-md.S_L+md.S_AX-md.SLEEVE_W-8.0, POST_X,
                    -md.S_W/2-md.SLEEVE_W-6.0, md.S_W/2+md.SLEEVE_W+6.0,
                    AXIS_Z-md.S_W/2-md.SLEEVE_W-6.0, POST_TOP)
    s = s.union(collar).union(slv)
    # The pad: the buttress' shell capped solid over its top PAD_T, which is what gives
    # the encoder bridge's two M3 nuts somewhere to sit.  A 4.5 mm wall cannot hold a
    # nut_slot - it needs the nut's across-flats plus two walls, 9.5 mm - and nothing on
    # this stand threads into plastic any more than anything on the robot does.
    s = s.union(md.bxc(POST_X-PAD_L, POST_X, -POST_HY, POST_HY, POST_TOP-PAD_T, POST_TOP))
    s = s.cut(at_bench(md.servo_case(md.CLR)))            # re-open the bore through it
    # the cable window and the connector, straight through into the cavity
    s = s.cut(md.bxc(-160.0, md.S_AX-md.S_L+1.0,
                     -md.CONN_H/2-2.0, md.CONN_H/2+2.0,
                     AXIS_Z-md.CONN_W/2-2.0, AXIS_Z+md.CONN_W/2+2.0))
    # Nothing is added forward of POST_X, so the sleeve's own two thrust bolts stay
    # reachable from +x with the stand assembled - which is what makes the assembly order
    # here the same as the robot's: nuts in the lug, servo in, both bolts, then the fork.
    #
    # The encoder bridge's two M3, down into the pad.  Their nut channels open on the pad's
    # +y face, which is bare air whatever else is fitted; the bolt itself is buried, so
    # the nut goes in first and the bridge comes off without disturbing it.
    for bx, by in ENC_BOLT:
        s = s.cut(md.cyl(md.M3_CLR, PAD_T+2.0, (bx, by, POST_TOP-PAD_T-1.0)))
        s = s.cut(md.nut_slot((bx, by, POST_TOP-6.0), (0, 1, 0), up=(0, 0, -1),
                              run=POST_HY-by+3.0))
    # ... and the protractor's two, down into the plate.  Their channels open on the
    # plate's own -y edge, which is the far side of the stand from everything else, so
    # they go in first and stay in whichever stand is fitted.
    py = PROT_Y0 + PROT_T/2.0
    for bx in PROT_BOLT:
        s = s.cut(md.cyl(md.M3_CLR, BASE_T+2.0, (bx, py, -1.0)))
        s = s.cut(md.nut_slot((bx, py, 3.0), (0, -1, 0), up=(0, 0, 1),
                              run=BASE_HY+py+4.0))
    return s


# =====================================================================================
# 4. PART: torque_arm  (stand A)
# =====================================================================================
def torque_arm():
    """md.fork() + a tapered beam with three load stations, and a pointer over the
    protractor.

    The beam is one-sided, so its own weight is a torque about the joint and it is in
    every reading.  That is a tare, not a defect - it is a rigid printed part, so main()
    reports its mass, the radius of its centre of mass and the moment that follows, and
    the protocol subtracts m*g*r*cos(theta).  A balanced beam would be better and cannot
    exist here: the counterweight would have to swing through the buttress.

    WEIGH THIS PART before using it.  Its mass enters every torque on stand A, and a
    printed mass is a fill factor away from its solid volume - the estimate main() prints
    uses md.PRINT_FILL_MEAN, which is a mean over other parts."""
    r0, r1 = md.SPINE_R0, ARM_STATIONS[-1] + ARM_TIP
    s = at_bench(md.fork())
    s = s.union(beam_x(r0, r1))
    # load stations: an M3 cross hole through the beam, on a flat pad either side so a
    # hook bolt and its nut seat square whatever the taper is doing there.
    for r in ARM_STATIONS:
        hy, _ = taper((r - r0)/(r1 - r0))
        s = s.union(md.bxc(r-6.0, r+6.0, -hy-ARM_PAD, hy+ARM_PAD,
                           AXIS_Z-6.0, AXIS_Z+6.0))
        s = s.cut(md.cyl(ARM_HOLE, 2*(hy+ARM_PAD)+4.0, (r, -hy-ARM_PAD-2.0, AXIS_Z),
                         axis=(0, 1, 0)))
    # pointer: a blade in the protractor's plane, carried out on a web off the beam's
    # -y face.  Both stay outboard of nothing and inboard of the plate by ENC-sized air;
    # rom_scan() in main() is what actually proves they clear the buttress.
    py0, py1 = POINT_Y
    pr0, pr1 = POINT_R
    hy = taper((POINT_BR - r0)/(r1 - r0))[0]
    # the bracket: over the plate's rim at POINT_BR, out to the blade's plane
    s = s.union(md.bxc(POINT_BR, pr1, py1, -hy+0.5, AXIS_Z-6.0, AXIS_Z+6.0))
    # the blade, back inward over the scale, tapering to POINT_TIP at the ticks' inner end
    s = s.union(md.bxc(pr0, pr1, py0, py1, AXIS_Z-5.0, AXIS_Z+5.0))
    for sg in (1, -1):
        s = s.cut(md.bxc(pr0-1.0, POINT_BR, py0-1.0, py1+1.0,
                         AXIS_Z+sg*POINT_TIP/2, AXIS_Z+sg*9.0))
    return s


# =====================================================================================
# 5. PART: protractor  (stand A)
# =====================================================================================
def protractor():
    """+-PROT_SWEEP deg of scale, 1 deg ticks, in the plane just outboard of the fork's
    -y arm.  It is not the instrument - the AS5600 is, at 0.088 deg - but it is what you
    set zero against, what you read with no electronics powered, and the check that says
    the encoder has not lost a whole turn.

    One flat plate, one plane, no support: sector, web, leg and foot are all in the
    y = PROT_Y0 slab, because that slab is the one volume around this joint that nothing
    else can reach.  It shares that slab with stand B's inertia disc, which is why the
    two stands are never assembled at once and why sweep() checks the arm against the
    protractor and the disc against the bridge, but never all four together."""
    a, y0, y1 = PROT_SWEEP, PROT_Y0, PROT_Y0 + PROT_T
    s = arc_sector(0.0, PROT_R1, -a, a, y0, y1)          # the scale's own disc
    s = s.union(disc_y(PROT_HUB, y0, y1))                # ... and the hub that ties it on
    s = s.cut(disc_y(EYE_D/2.0, y0-1.0, y1+1.0))         # ... bored for the fork's screws
    s0, s1 = PROT_STRUT                                  # the strut, out to the plate
    s = s.union(arc_sector(PROT_HUB-2.0, 260.0, s0, s1, y0, y1, step=4.0)
                .intersect(md.bxc(BASE_X0+6.0, BASE_X1, y0-1.0, y1+1.0,
                                  BASE_T, POST_TOP)))
    fx0, fx1 = min(PROT_BOLT)-16.0, max(PROT_BOLT)+16.0
    s = s.union(md.bxc(fx0, fx1, y0, y1, BASE_T, BASE_T+PROT_FOOT_T))
    for bx in PROT_BOLT:                       # down into the base plate's own nut slots
        s = s.cut(md.cyl(md.M3_CLR, PROT_FOOT_T+2.0, (bx, y0-1.0, BASE_T-1.0)))
    # lighten the web.  The scale is a rim on a plate and the plate is only there to carry
    # it, so most of the disc can go; the ring of holes keeps the rim continuous.
    for i in range(-3, 4):
        for r in PROT_LIGHT_R:
            th = i*25.0 + (12.5 if r == PROT_LIGHT_R[0] else 0.0)
            if abs(th) > a - 10.0: continue
            cx, cz = polar(r, th)
            s = s.cut(md.cyl(PROT_LIGHT/2.0, PROT_T+2.0, (cx, y0-1.0, cz), axis=(0, 1, 0)))
    w, l1, l5, l10 = PROT_TICK
    ticks = []
    for i in range(-int(a), int(a)+1):
        ln = l10 if i % 10 == 0 else (l5 if i % 5 == 0 else l1)
        t = md.bxc(PROT_R1-ln, PROT_R1+1.0, y0-1.0, y0+PROT_T*0.55,
                   AXIS_Z-w/2, AXIS_Z+w/2)
        ticks.append(t.val().moved(cq.Location(cq.Vector(0, 0, AXIS_Z),
                                               cq.Vector(0, 1, 0), -i)))
    s = s.cut(md.W(cq.Compound.makeCompound(ticks)))
    return s


# =====================================================================================
# 6. PARTS: enc_magnet_cap / enc_bridge  (stands A and B)
# =====================================================================================
def enc_magnet_cap():
    """the diametric magnet, on the joint axis, on the fork's +y arm.

    Held by the same four M2.5 that hold the arm to the driven hub - M2.5 x 10 instead of
    the robot's x 6, which still leaves the screw 2 mm into the hub's thread and nowhere
    near the case behind it.  The magnet is a through pocket: it goes in from the fork
    side and the fork arm is what stops it coming back out, so nothing here relies on
    glue and there is no plastic in the field path."""
    y0 = md.HUB_TOP_Z + md.ARM_T                       # the fork arm's outer face, +y
    s = disc_y(CAP_R, y0, y0 + CAP_T)
    s = s.cut(disc_y((MAG_D + MAG_FIT)/2.0, y0-1.0, y0+CAP_T+1.0))
    for i in range(md.HUB_N):
        th = math.radians(90*i)
        c = (md.HUB_BC/2*math.cos(th), AXIS_Z + md.HUB_BC/2*math.sin(th))
        s = s.cut(disc_y(md.M25_CLR, y0-1.0, y0+CAP_T+1.0, c))
    # a flat, so the operator can see which way round a diametric magnet went in
    s = s.cut(md.bxc(-CAP_R-1, -CAP_R+2.0, y0-1.0, y0+CAP_T+1.0, AXIS_Z-4.0, AXIS_Z+4.0))
    return s

def enc_y0():
    """the bridge's inboard face: clear of the rotating cap by ENC_GAP."""
    return md.HUB_TOP_Z + md.ARM_T + CAP_T + ENC_GAP

def mag_to_ic():
    """magnet face -> the AS5600's own face, which is what its datasheet window is on."""
    return ENC_GAP + ENC_LEDGE

def enc_bridge():
    """the AS5600's stator: off the buttress, over the axis, nothing touching the table.

    Three things about it are deliberate.  It is referenced to the BUTTRESS and not to the
    base plate, so what the stand does under load is largely common mode between the
    sensor and the servo case and the reading stays a joint angle rather than a stand
    angle.  It sits on the pad at the top of the buttress, which is the closest flat to
    the joint that is out of the fork's sweep in every direction.  And its two mounting
    holes are SLOTS in y: ENC_GAP is set at assembly by sliding it until the AS5600's AGC
    register sits mid-range, which is the only honest way to set a magnetic air gap.  The
    inboard end of the slot lands the bridge on the magnet cap - a hard stop, and an
    obvious one."""
    y0 = enc_y0()
    y1 = y0 + ENC_LEDGE + ENC_PCB[2] + 2.0
    x0 = POST_X - PAD_L
    s = md.bxc(x0, 20.0, y0, y1, AXIS_Z-20.0, POST_TOP+ENC_FOOT_T)      # the plate
    s = s.cut(md.bxc(x0-1.0, -24.0, y0-1.0, y1+1.0, AXIS_Z-21.0, 180.0))  # ... as an L
    s = s.union(md.bxc(x0, POST_X, 6.0, y1, POST_TOP, POST_TOP+ENC_FOOT_T))   # the foot
    # the board pocket, loaded from +y, and the window the IC looks through
    pw, ph, _ = ENC_PCB
    s = s.cut(md.bxc(-pw/2, pw/2, y0+ENC_LEDGE, y1+1.0, AXIS_Z-ph/2, AXIS_Z+ph/2))
    s = s.cut(disc_y(ENC_WIN/2.0, y0-1.0, y1+1.0))
    for sx in (-1, 1):                                   # 2 x M2 through the board
        s = s.cut(disc_y(M2_CLR, y0-1.0, y1+1.0, (sx*ENC_PCB_HOLES/2.0, AXIS_Z)))
    # the slots.  They run in y, which is the direction the gap is in; the nuts they pull
    # against are in the pad, and their channels open on its +y face.
    for bx, by in ENC_BOLT:
        slot = md.bxc(bx-md.M3_CLR, bx+md.M3_CLR, by-ENC_SLOT, by+ENC_SLOT,
                      POST_TOP-1.0, POST_TOP+ENC_FOOT_T+1.0)
        for e in (-1, 1):
            slot = slot.union(md.cyl(md.M3_CLR, ENC_FOOT_T+2.0,
                                     (bx, by+e*ENC_SLOT, POST_TOP-1.0)))
        s = s.cut(slot)
    return s


# =====================================================================================
# 7. PART: inertia_disc  (stand B)
# =====================================================================================
def inertia_disc():
    """md.fork() + a flywheel in the plane outboard of the fork's -y arm.

    Its own J is the baseline and is computed from the real solid in main(); M8 bolts in
    the pocket circle step it up.  Fill the pockets symmetrically only - 0, 2, 4, 6 or 8,
    opposite pairs.  An unbalanced rotor at the no-load speed is not dangerous on this
    scale, but it puts a once-per-rev disturbance into exactly the step response the
    stand exists to measure."""
    s = at_bench(md.fork())
    a0 = md.ARM_BOT_TOP - md.ARM_T                       # the fork arm's outer face, -y
    s = s.union(disc_y(DISC_HUB_R, DISC_Y0 + DISC_T, a0))
    s = s.union(disc_y(DISC_R, DISC_Y0, DISC_Y0 + DISC_T))
    # bored on the axis, for the same reason the protractor is: the rotor caps the fork's
    # -y arm, and those four M2.5 have to go in through it.  It also takes 8 g off the
    # one place on the rotor where inertia is cheapest, which is why it costs nothing.
    s = s.cut(disc_y(EYE_D/2.0, DISC_Y0-1.0, a0+1.0))
    for i in range(DISC_N):
        th = math.radians(360.0*i/DISC_N + 180.0/DISC_N)
        c = (DISC_BC/2*math.cos(th), AXIS_Z + DISC_BC/2*math.sin(th))
        s = s.cut(disc_y(DISC_HOLE/2.0, DISC_Y0-1.0, DISC_Y0+DISC_T+1.0, c))
    # lighten the web between the hub and the pocket circle, so the disc's OWN inertia is
    # mostly in its rim where it is least sensitive to the fill factor being off
    for i in range(DISC_N):
        th = math.radians(360.0*i/DISC_N)
        c = (DISC_LIGHT_BC/2*math.cos(th), AXIS_Z + DISC_LIGHT_BC/2*math.sin(th))
        s = s.cut(disc_y(DISC_LIGHT/2.0, DISC_Y0-1.0, DISC_Y0+DISC_T+1.0, c))
    return s


# =====================================================================================
# 8. PARTS: leg_tower / foot_plate / cell_riser  (stand C)
# =====================================================================================
# Tower frame: the ROBOT's own coordinates, translated so the chassis' front face
# (x = BODY_L/2, which is where roll_module() joins the body) lands on x = 0.  Everything
# outboard of that plane is the real robot; everything at x < 0 is the flange.
TOWER_DX = -md.BODY_L/2.0

def leg_tower():
    """md.roll_module(), unmodified, on a flange.

    The cradle is not a bench copy of the robot's hip - it IS the robot's hip, moved.  A
    leg bolted to it hangs at the real ROLL_Y / LEG_Y / PITCH_Z geometry, so the shin's
    inertia about the knee and the hip's moment arm are the ones the search will have to
    respect.  Bolt the flange to a board or a 2020 upright with the leg hanging past the
    edge; how far the roll axis has to be above the foot plate is main()'s to report, and
    it is measured off the posed leg rather than off FOOT_Z."""
    s = md.roll_module().translate((TOWER_DX, 0, 0))
    y0, y1 = TOWER_HY
    s = s.union(md.bxc(-TOWER_T, 0.0, y0, y1, -TOWER_Z, TOWER_Z))
    for by, bz in TOWER_BOLT:            # through only - the nut lands on the far face
        s = s.cut(md.cyl(M6_CLR, TOWER_T+2.0, (-TOWER_T-1.0, by, bz), axis=(1, 0, 0)))
    return s

def tower_bolt_margin():
    """how far each flange bolt sits outside the hip-roll fork's swept circle, mm.  The
    fork sweeps r <= CLEAR_R about the roll axis, and a nut standing proud of the flange's
    inboard face inside that circle is a leg that jams at some roll angle and nowhere
    else - which is exactly the kind of thing rom_scan cannot see here, because the flange
    is not in the robot's own model."""
    return min(math.hypot(by - md.ROLL_Y, bz - md.ROLL_Z) - CLEAR_R
               for by, bz in TOWER_BOLT)

def cell_holes(sx):
    """the two bolt centres at one end of the cell.  sx = -1 free end, +1 fixed end."""
    return [sx*(CELL_L/2 - CELL_END - i*CELL_PITCH) for i in (0, 1)]

def foot_plate():
    """the platform half of a straight-bar load cell, with the cell's free end under it.

    This is the only force anywhere in this repo that is measured rather than derived, and
    it is what closes measurement #4: the servo reports a load in counts, the cell reports
    newtons, and contact.py's threshold is the line between them.  Sized on fea.py's own
    landing case - 36 N on one foot - so a 5 kg (49 N) bar covers it and not much more."""
    px, py = PLATE_XY
    s = md.bxc(-px/2, px/2, -py/2, py/2, 0.0, PLATE_T)
    # the free-end spacer: it hangs the platform off that end of the cell and nothing else
    xs = cell_holes(-1)
    x0, x1 = min(xs)-9.0, max(xs)+9.0
    s = s.union(md.bxc(x0, x1, -CELL_W/2-3.0, CELL_W/2+3.0, -PLATE_GAP, 0.0))
    for cx in xs:
        s = s.cut(md.cyl(CELL_M4_D/2.0, PLATE_T+PLATE_GAP+2.0, (cx, 0.0, -PLATE_GAP-1.0)))
        s = s.cut(md.cyl(CELL_M4_HEAD/2.0, 4.5, (cx, 0.0, PLATE_T-4.5)))   # head, flush
    # a witness slot down the centre line: it is how you see, with the leg standing on it,
    # that the platform is still floating and has not settled onto the riser.
    s = s.cut(md.bxc(px/2-14.0, px/2+1.0, -1.5, 1.5, -1.0, PLATE_T+1.0))
    return s

def cell_riser():
    """the fixed end of the same cell, down onto the board.  A separate part on purpose:
    the two ends of a bending-beam cell must not share a body, or it reads nothing."""
    xs = cell_holes(+1)
    x0, x1 = min(xs)-9.0, max(xs)+9.0
    z1 = -PLATE_GAP - CELL_H
    s = md.bxc(x0, x1, -CELL_W/2-5.0, CELL_W/2+5.0, z1-RISER_H, z1)
    for cx in xs:                                    # M5 up into the cell's own threads
        s = s.cut(md.cyl(CELL_M5_D/2.0, RISER_H+2.0, (cx, 0.0, z1-RISER_H-1.0)))
        s = s.cut(md.cyl(CELL_M5_HEAD/2.0, 5.0, (cx, 0.0, z1-RISER_H-0.5)))
    for sy in (-1, 1):                               # ... and down onto the board, M4+nut
        s = s.cut(md.cyl(M4_CLR, RISER_H+2.0,
                         (max(xs)+4.5, sy*(CELL_W/2+2.6), z1-RISER_H-1.0)))
    return s


# =====================================================================================
# 9. purchased BOM - everything the stands need that is not printed
# =====================================================================================
# (item, qty, stands, note).  Marked **verify** where the dimension is in the parameter
# block above but has not been read off a real part.
BUY = [
    ("ST3215 bus servo + both stock aluminium hubs", "1 (3+ to get a spread)", "A B C",
     "the device under test.  Measurements 1..9 are per servo and they differ; the "
     "search's margin has to cover the worst one on the robot"),
    ("Waveshare URT-1 (or an ESP32 + a TTL half-duplex driver)", 1, "A B C",
     "the servo bus.  Everything the servo reports - position, speed, load, voltage, "
     "current, temperature - comes over this, and calibrating those reports is the point"),
    ("12 V bench supply, >= 5 A, current readout", 1, "A B C",
     "5 A because a stalled ST3215 is the load case; the readout is half of measurement 9"),
    ("INA226 (or equivalent) current/voltage module", 1, "A B",
     "only if the supply cannot log.  The torque-speed line needs current against time, "
     "not a panel meter"),
    ("AS5600 breakout board", 1, "A B",
     "the external angle.  Outline is **verify** - ENC_PCB in this file is a generic "
     "12.7 mm square"),
    ("N35 magnet, 6 x 2.5 mm, DIAMETRICALLY magnetised", 1, "A B",
     "an axially magnetised one of the same size reads as a constant and looks like a "
     "dead sensor"),
    ("straight-bar load cell, 5 kg, + HX711", 1, "C",
     "outline is **verify** - CELL_* here is the common 80 x 12.7 x 12.7 bar"),
    ("calibrated masses 0.5 / 1 / 2 kg + S-hooks", "2 each", "A",
     "the primary standard on this bench.  2 kg at the outer station is exactly the "
     "catalogue stall torque by construction"),
    ("kitchen/jeweller's scale, 1 g or better", 1, "A B",
     "for the masses, the arm's tare and every M8 bolt that goes in the rotor"),
    ("M8 x 20 bolt + nut + 2 washers", 8, "B", "the rotor's inertia increments - weigh them"),
    ("M2.5 x 6 / x 7 (fork -> hubs, as the robot)", "4 / 4", "A B C", "per joint"),
    (f"M2.5 x {CAP_SCREW:.0f} (magnet cap + fork arm -> driven hub)", 4, "A B",
     f"replaces the robot's x 6 on the +y arm only: {CAP_T:.1f} of cap and "
     f"{md.ARM_T:.1f} of arm to clear, {md.HUB_T_TOP:.1f} of hub to thread into.  Do not "
     f"go longer - past the hub it bottoms on the case"),
    ("M3 x 10 + M3 nut (sleeve thrust clamp)", 2, "A B C", "per joint, and not longer"),
    ("M3 x 16 + M3 nut (encoder bridge -> the buttress' pad, protractor -> the plate)",
     4, "A B", "the bridge's two go through slots - that is how the encoder gap is set"),
    ("M4 x 30 + M4 nut + washer (base plate -> board)", 6, "A B",
     "or fill the ballast tray - main() prints how much it needs.  Not optional: the "
     "stand tips over at full scale on its own weight"),
    (f"M4 x 20 (load cell free end -> foot plate)", 2, "C",
     f"{PLATE_T:.0f} of plate + {PLATE_GAP:.0f} of spacer, then into the cell's own M4 "
     f"thread.  No nut - **verify** that thread against the bar you buy"),
    ("M5 x 16 (load cell fixed end -> riser)", 2, "C",
     "up through the riser into the cell's own M5 thread - **verify** it too"),
    ("M4 x 25 + M4 nut (riser -> board)", 2, "C", ""),
    ("M6 x 35 + M6 nut + washer (leg tower -> board)", 4, "C",
     "for an 18 mm board.  The nuts land on the flange's inboard face, where the roll "
     "fork sweeps; main() reports how far the bolt centres clear it"),
    ("plywood or MDF board, >= 300 x 200 x 18, + 2 G-clamps", 1, "A B C",
     "the stand is printed, the ground is not.  This is what makes the tip-over check "
     "in main() pass"),
    ("2020 extrusion upright, >= 400 mm, + a right-angle bracket", 1, "C",
     "alternative to the board for the tower; the leg has to hang past an edge"),
    ("cable tie, 2.5 mm", 6, "A B C", "servo lead strain relief off the buttress"),
]


# =====================================================================================
# 10. build / checks
# =====================================================================================
PARTS, SILHOUETTE = {}, None
STATIC = ("bench_base", "enc_bridge", "protractor")
MOVING = ("torque_arm", "inertia_disc", "enc_magnet_cap")

def build():
    global SILHOUETTE
    SILHOUETTE = bench_base(hollow=False)
    PARTS["bench_base"]     = (bench_base(),     1, "A B", "PETG/ASA, 5 walls, 30% gyroid")
    PARTS["torque_arm"]     = (torque_arm(),     1, "A",   "PETG/ASA, 5 walls, 40% - WEIGH IT")
    PARTS["protractor"]     = (protractor(),     1, "A",   "PETG/ASA, 4 walls, 25%")
    PARTS["inertia_disc"]   = (inertia_disc(),   1, "B",   "PETG/ASA, 5 walls, 40%")
    PARTS["enc_bridge"]     = (enc_bridge(),     1, "A B", "PETG/ASA, 4 walls, 40%")
    PARTS["enc_magnet_cap"] = (enc_magnet_cap(), 1, "A B", "PETG/ASA, 4 walls, 60%")
    PARTS["leg_tower"]      = (leg_tower(),      1, "C",   "PETG/ASA, 5 walls, 40%")
    PARTS["foot_plate"]     = (foot_plate(),     1, "C",   "PETG/ASA, 4 walls, 40%")
    PARTS["cell_riser"]     = (cell_riser(),     1, "C",   "PETG/ASA, 4 walls, 60%")
    return PARTS

PRINT_ORIENT = {
    # base plate on the bed: the buttress' front rake is self-supporting and the sleeve
    # bore comes out horizontal, which is how the robot's own cradle prints too.
    "bench_base":     ((1, 0, 0), 0),
    # joint axis vertical, i.e. layers in the beam's own bending plane - the same reason
    # md.PRINT_ORIENT does this to thigh_A.  The arm is a cantilever in bending; printed
    # the other way the load is inter-layer tension at the root.
    "torque_arm":     ((1, 0, 0), 90),
    "inertia_disc":   ((1, 0, 0), 90),
    "enc_magnet_cap": ((1, 0, 0), 90),
    "enc_bridge":     ((1, 0, 0), 90),
    "protractor":     ((1, 0, 0), 90),
    # flange face down: the one big flat, and the cradle's shelves then print the way
    # chassis_bottom's do.
    "leg_tower":      ((0, 1, 0), 90),
    "foot_plate":     ((1, 0, 0), 180),
    "cell_riser":     ((1, 0, 0), 0),
}

def rho(name): return md.part_rho(name)          # bench parts are PETG; unknown fill ->
                                                 # md.PRINT_FILL_MEAN, and main() says so

def mass_g(name):
    return PARTS[name][0].val().Volume()/1000.0 * rho(name)

def interference():
    """the static parts share one rigid stand; nothing else checks that they do not share
    solid.  Same test, same tolerance as mini_dog.interference()."""
    bad = []
    for i, a in enumerate(STATIC):
        for b in STATIC[i+1:]:
            try:    v = PARTS[a][0].val().intersect(PARTS[b][0].val()).Volume()
            except Exception: v = 0.0
            if v > md.INTERF_TOL: bad.append((a, b, v))
    return bad

def sweep(name, static_names, step=5, lo=-150, hi=150):
    """the free sweep of a moving part about the bench axis, against the stand it runs on.
    This is the check that the CLEAR_R rule actually held: the buttress is the only thing
    in the way, and how much arm it leaves is not something to take on trust.

    Runs against bench_base(hollow=False) - see its docstring.  Everything else is thin
    enough to have no reachable interior."""
    stat = None
    for n in static_names:
        w = SILHOUETTE if n == "bench_base" else PARTS[n][0]
        stat = w if stat is None else stat.union(w)
    return md.rom_scan(PARTS[name][0], stat, (0.0, 0.0, AXIS_Z), axis=(0, 1, 0),
                       lo=lo, hi=hi, step=step)

def rotor_J(name):
    """kg*m^2 about the bench axis, from the real solid, by the parallel-axis theorem.
    Uses export_sim.MP - the same mass properties the URDF and the MJCF are built from."""
    mp = MP.of(PARTS[name][0], rho(name)*1e-6)
    d = (mp.c[0] - 0.0, mp.c[2] - AXIS_Z)                       # offset in the axis' plane
    return (mp.I[1][1] + mp.m*(d[0]**2 + d[1]**2)) * 1e-6, mp.m

def arm_tare():
    """(mass kg, com radius mm, N*m at the horizontal).  Subtract this from every static
    reading on stand A."""
    mp = MP.of(PARTS["torque_arm"][0], rho("torque_arm")*1e-6)
    r = math.hypot(mp.c[0], mp.c[2]-AXIS_Z)
    return mp.m, r, mp.m*G*r/1000.0

def buttress_flex(T_nm, F_n):
    """Euler-Bernoulli rotation of the sleeve about the joint axis, integrated over the
    buttress' own section as it actually tapers.  Slices the real solid, so it follows the
    shell, the rib and the cable window rather than a nominal rectangle.

    It is first order and it is not a substitute for fea.py: no shear, no stress
    concentration, no inter-layer plane.  It does not have to be exact.  It is here to
    answer one question - is the stand an order of magnitude stiffer than the joint it is
    measuring - and the encoder is referenced to the buttress precisely so that whatever
    is left of this is common mode."""
    s = PARTS["bench_base"][0].val()
    E = PETG["E"]
    z0, z1, n = BASE_T + 4.0, AXIS_Z - md.S_W/2 - md.SLEEVE_W - 2.0, 24
    th, worst = 0.0, None
    for i in range(n):
        zc = z0 + (z1-z0)*(i+0.5)/n
        t = 2.0
        try:
            sl = s.intersect(md.bxc(-300, 300, -300, 300, zc-t/2, zc+t/2).val())
            a = sl.Volume()/t
            if a < 1e-6: continue
            I = cq.Shape.matrixOfInertia(sl)[1][1]/t                 # about the y axis, mm^4
            c = sl.Center()
            bb = sl.BoundingBox()
            fib = max(bb.xmax - c.x, c.x - bb.xmin)                   # extreme fibre, mm
        except Exception:
            continue
        M = T_nm*1000.0 + F_n*abs(c.x)                               # N*mm at this height
        th += M/(E*I) * ((z1-z0)/n)
        # printed base-down, the layers are horizontal and this bending stress is normal
        # to them, so it is the inter-layer allowable that applies - the same call
        # fea.py makes on the leg parts.
        sf = PETG["s_z"] / (M * fib / I) if I > 0 else 0.0
        if worst is None or sf < worst[0]: worst = (sf, zc, a, I)
    return math.degrees(th), worst

def tipover():
    """(restoring N*m from the stand's own weight, N*m demanded at full scale, kg of
    ballast needed).  A printed stand loaded to 2.94 N*m at 150 mm tips: this is not a
    warning, it is a requirement to bolt it down."""
    m_stand = sum(mass_g(n) for n in ("bench_base",))/1000.0
    mp = MP.of(PARTS["bench_base"][0], rho("bench_base")*1e-6)
    edge = BASE_X1
    restore = m_stand*G*(edge - mp.c[0])/1000.0
    W = ARM_MASS_KG*G
    demand = W*(ARM_STATIONS[-1] - edge)/1000.0 + W*0.0
    arm_m, arm_r, _ = arm_tare()
    demand += arm_m*G*max(0.0, arm_r - edge)/1000.0
    bx0, bx1, bh = BALLAST
    lever = (edge - (bx0+bx1)/2.0)/1000.0
    need = max(0.0, (2.0*demand - restore))/(G*lever)                # 2x margin
    return restore, demand, need


def stance():
    """where the foot plate goes under the tower: the contact patch of the leg in the
    robot's OWN stance pose, taken off the posed solid rather than off FOOT_Z.  Those two
    are not the same number - the stance pose swings the foot forward and up, and reading
    the nominal one puts the plate 20 mm out and leaves the foot hanging in air."""
    if not md.PARTS: md.build()
    bb = md.posed(md.PARTS["foot"][0], "shin").translate((TOWER_DX, 0, 0)).val().BoundingBox()
    return ((bb.xmin+bb.xmax)/2.0, (bb.ymin+bb.ymax)/2.0, bb.zmin - PLATE_T)

def scenes():
    """(name, [(shape, colour)]) per stand, with the servo and its stock hubs in place -
    the three things you would have on the bench, assembled."""
    grey, wht, blk = (0.62, 0.65, 0.70), (0.85, 0.86, 0.88), (0.16, 0.16, 0.18)
    alu, org = (0.72, 0.76, 0.82), (0.90, 0.58, 0.16)
    srv = (at_bench(md.servo_dummy()).val(), blk)
    hub = (at_bench(md.hubs()).val(), alu)
    base = (PARTS["bench_base"][0].val(), grey)
    brg = (PARTS["enc_bridge"][0].val(), wht)
    out = [("a", [base, srv, hub, brg,
                  (PARTS["protractor"][0].val(), wht),
                  (PARTS["torque_arm"][0].val(), org),
                  (PARTS["enc_magnet_cap"][0].val(), blk)]),
           ("b", [base, srv, hub, brg,
                  (PARTS["inertia_disc"][0].val(), org),
                  (PARTS["enc_magnet_cap"][0].val(), blk)])]
    # stand C: the tower with a whole leg on it, in the robot's own pose, and the foot
    # plate under it - which is also the check that the 200 mm of drop is real.
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
    out.append(("c", leg))
    return out

def render_all():
    from render import render
    for name, sc in scenes():
        f = (0, 0, AXIS_Z) if name in ("a", "b") else (TOWER_DX, md.LEG_Y, -100)
        for view, cam in (("iso", (420, -560, 380)), ("side", (10, -900, 40))):
            render(sc, os.path.join(OUT, f"view_{name}_{view}.png"),
                   tuple(c + o for c, o in zip(cam, f)))

def main(do_render=False):
    for d in ("step", "stl"): os.makedirs(os.path.join(OUT, d), exist_ok=True)
    build()
    rows = []
    print("\n  part               qty  stand   volume   est.mass    print bbox (mm)")
    for name, (wp, qty, stand, note) in PARTS.items():
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
        rows.append({"part": name, "qty": qty, "stand": stand, "volume_cm3": round(v, 1),
                     "est_mass_g": round(m, 1),
                     "print_bbox_mm": [round(bb.xlen, 1), round(bb.ylen, 1), round(bb.zlen, 1)],
                     "valid": ok, "note": note})
        print(f"  {name:17s} x{qty}  {stand:5s} {v:7.1f} cm3 {m:7.1f} g   "
              f"{bb.xlen:6.1f} x {bb.ylen:6.1f} x {bb.zlen:6.1f}"
              f"{'' if ok else '   !! INVALID'}")

    # Stand C's own invariant, and it is the one that fails silently: a bending-beam cell
    # only reads if its two ends are carried by two bodies that do not touch.
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

    bad = interference()
    for na, nb, v in bad:
        print(f"  !! INTERFERENCE  {na} x {nb}  {v:.1f} mm3")
    if not bad:
        print(f"  stand clear: {' / '.join(STATIC)} share no solid")

    # the sweep each stand actually has, against the stand it runs on
    # Sign: rom_scan rotates about +y, and a positive rotation about +y takes the arm
    # from +x toward -z, i.e. DOWNWARD.  Checked, not assumed - at +90 the arm's bbox
    # bottoms out at z = 19, one arm's length below the axis.
    print("  sweep (5 deg steps, real solids; 0 = arm horizontal +x, + = down):")
    sw = {}
    sw["torque_arm"]   = sweep("torque_arm", ("bench_base", "protractor", "enc_bridge"))
    sw["inertia_disc"] = sweep("inertia_disc", ("bench_base", "enc_bridge"))
    for k, v in sw.items():
        need = PROT_SWEEP if k == "torque_arm" else 90.0
        ok = (v[0] <= -need and v[1] >= need)
        print(f"    {k:13s} free {v[0]:+4d} .. {v[1]:+4d} deg"
              f"{'' if ok else f'   !! wanted +-{need:.0f}'}")

    # the encoder gap, which is the one dimension a wrong AS5600 outline breaks
    g = mag_to_ic()
    flag = "" if 0.5 <= g <= 3.0 else "   !! outside the AS5600's 0.5..3.0 mm window"
    print(f"  encoder:     magnet face -> IC {g:.1f} mm, slotted +-{ENC_SLOT:.1f} mm"
          f" ({ENC_GAP:.1f} air + {ENC_LEDGE:.1f} ledge){flag}")

    # stand A's numbers
    m, r, tare = arm_tare()
    print(f"  torque arm:  tare {m*1000:.0f} g at r = {r:.1f} mm -> {tare:.3f} N*m at the"
          f" horizontal (SUBTRACT IT; weigh the part)")
    print("    station     1 kg      2 kg      full scale needs")
    for st in ARM_STATIONS:
        print(f"    {st:5.0f} mm  {1.0*G*st/1000:6.3f}   {2.0*G*st/1000:6.3f} N*m   "
              f"{md.SERVO_STALL_NM/(G*st/1000.0):5.2f} kg")

    th, worst = buttress_flex(md.SERVO_STALL_NM, ARM_MASS_KG*G)
    print(f"  buttress:    {th:.3f} deg at full scale ({md.SERVO_STALL_NM:.2f} N*m + "
          f"{ARM_MASS_KG*G:.0f} N), E = {PETG['E']:.0f} MPa")
    if worst:
        print(f"               thinnest section at z = {worst[1]:.0f}: A = {worst[2]:.0f} mm2,"
              f" I = {worst[3]/1000:.0f} x10^3 mm4, inter-layer SF ~{worst[0]:.1f}")

    restore, demand, need = tipover()
    print(f"  tip-over:    stand alone restores {restore:.2f} N*m against {demand:.2f} N*m"
          f" at full scale")
    print(f"               -> BOLT IT DOWN (6 x M4), or {need:.1f} kg in the ballast tray"
          f" for 2x margin")

    # stand B's numbers
    J0, mdisc = rotor_J("inertia_disc")
    dJ = DISC_N//2 * 2 * (BOLT_M8_G/1000.0) * (DISC_BC/2000.0)**2
    print(f"  rotor:       {mdisc*1000:.0f} g, J = {J0*1e6:.0f} g*cm2 bare"
          f" (= {J0:.2e} kg*m2)")
    print(f"    bolts   0        2        4        6        8   (M8 x 20 + nut,"
          f" {BOLT_M8_G:.0f} g each, at r = {DISC_BC/2:.0f} mm)")
    line = "    J g*cm2 " + "".join(
        f"{(J0 + n*(BOLT_M8_G/1000.0)*(DISC_BC/2000.0)**2)*1e6:8.0f} " for n in (0, 2, 4, 6, 8))
    print(line)
    tau_free = J0 * (md.SERVO_NOLOAD_RADS/0.05)
    print(f"               bare rotor wants {tau_free:.3f} N*m to reach no-load speed in"
          f" 50 ms - {100*tau_free/md.SERVO_STALL_NM:.0f}% of stall, so the step response"
          f" is torque-limited and readable")

    # stand C's one geometric requirement
    sx, sy, sz = stance()
    print(f"  leg tower:   roll axis {-sz-PLATE_T:.0f} mm over the foot plate's top face,"
          f" x {sx:+.0f} y {sy:+.0f} from the flange" )
    print(f"               (the leg in md.posed() stance, off the real solid - the nominal"
          f" FOOT_Z is {abs(md.FOOT_Z):.0f})")
    m = tower_bolt_margin()
    if m < TOWER_BOLT_R - CLEAR_R:
        print(f"  !! TOWER BOLTS  nearest is {m:+.1f} mm outside the roll fork's sweep -"
              f" its nut is in the way of the leg")
    else:
        print(f"               flange bolts {m:+.1f} mm clear of the roll fork's"
              f" {CLEAR_R:.1f} mm sweep, so their nuts miss the leg")

    tm = sum(r_["est_mass_g"]*r_["qty"] for r_ in rows)
    per = {}
    for r_ in rows:
        for s in r_["stand"].split():
            per[s] = per.get(s, 0.0) + r_["est_mass_g"]*r_["qty"]
    print(f"\n  printed mass ~{tm:.0f} g total"
          f"   (A ~{per.get('A',0):.0f} g, B ~{per.get('B',0):.0f} g, C ~{per.get('C',0):.0f} g;"
          f" the shared parts count in both A and B)")

    print("\n  buy:")
    for item, qty, stands, note in BUY:
        print(f"    [{stands:5s}] {str(qty):>6s}  {item}")
        if note: print(f"                     {note}")

    with open(os.path.join(OUT, "bom.json"), "w") as f:
        json.dump({"parts": rows,
                   "buy": [{"item": i, "qty": q, "stands": s.split(), "note": n}
                           for i, q, s, n in BUY],
                   "sweep_deg": {k: list(v) for k, v in sw.items()},
                   "bench": {"axis_z": AXIS_Z, "arm_r": round(ARM_R, 1),
                             "arm_stations": [round(v, 1) for v in ARM_STATIONS],
                             "arm_tare_nm": round(tare, 4),
                             "rotor_J_kgm2": J0,
                             "rotor_J_per_bolt_kgm2": (BOLT_M8_G/1000.0)*(DISC_BC/2000.0)**2,
                             "encoder_gap_mm": round(g, 2),
                             "buttress_flex_deg": round(th, 4),
                             "ballast_kg": round(need, 1)},
                   "servo": {"stall_nm": md.SERVO_STALL_NM,
                             "noload_rads": md.SERVO_NOLOAD_RADS,
                             "mass_kg": md.SERVO_KG,
                             "source": "vendor catalogue - what these stands exist to replace"}},
                  f, indent=2)
    if do_render:
        render_all()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--render", action="store_true",
                    help="also write out/bench/view_{a,b,c}_{iso,side}.png - and look at them")
    main(do_render=ap.parse_args().render)
