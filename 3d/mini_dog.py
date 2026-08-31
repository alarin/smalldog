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
M25_NUT_AF, M25_NUT_H = 5.00, 2.00
M3_NUT_AF, M3_NUT_H = 5.60, 2.70
NUT_CLR    = 0.25                     # slide fit on a nut pocket's flats
# Sleeve thrust clamp.  The case sits in the sleeve on CLR alone and nothing holds it:
# 0.35 mm on the 45 mm flats is ~+-0.9 deg of knock per joint, ~+-2.5 deg at the foot, and
# PETG that wears a little more every time the gait reverses.  Two M3s through the -x end
# wall push the case onto the two legs the cable window leaves standing in the +x end wall,
# and that pins it: turn the case either way and one thrust bolt blocks it on the -x side
# while the opposite +x leg blocks it on the other, so the play is closed by geometry and
# not by friction.  One bolt is not enough - it only blocks one direction.  Both bolts
# thread into M3 nuts in side-loaded channels in the lug; nothing threads into plastic.
#
# Why not slit the sleeve into a C and squeeze it, which is the obvious clamp?  Because the
# -x end wall is the tube's ONLY crossing of y=0 - the cable window has already eaten the
# +x one - so slitting it opens the whole sleeve-plus-link box section.  That is measured,
# not guessed: the C costs thigh_A 1.1 -> 0.6 inter-layer SF at stall and 2.65 -> 12.2 mm
# of deflection, and tying the +x wall back across the window does not buy it back (0.5).
# The clamp that does not cut the ring wins.
#
# Everything here stays inside r < SPINE_R0 of the joint axis: the distal fork's spine
# sweeps the annulus SPINE_R0..~34 over the sleeve's whole length, and the hip bracket's
# inboard web already comes to r = 22.0, so the lug corner is held at 21.6.  Fit M3x10 and
# no longer - the head has to stay inside that radius too.
THRUST_Y    = 5.50                    # the two bolts, either side of the axis
THRUST_YL   = 10.00                   # lug half-width; the nut channels open on its faces
THRUST_Z    = 7.00                    # lug half-height
THRUST_L    = 6.00                    # how far the lug stands off the -x end wall
THRUST_SEAT = 3.00                    # lug behind each nut - this is what takes the preload

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
SPIGOT_R, SPIGOT_Z0, SPIGOT_H = 9.0, 2.0, 15.0   # the ankle spigot the TPU foot presses on
FOOT_FIT   = 0.15                     # TPU bore over the spigot
FOOT_NUT_Z = 15.0                     # foot-bolt nut slot, floor this far above FOOT_Z:
                                      # clear of the foot's top face, so the nut goes in
                                      # (and comes out) with the foot fitted
# The foot bolt spans FOOT_D/2 below FOOT_Z (the sole) to the nut FOOT_NUT_Z above it, so
# it is ~28 mm of span before the head is even seated - the length is geometry, not taste,
# and foot_bolt_check() below recomputes it.  It used to be specified as M3 x 16, which no
# placement of the head can reach: the head pocket and the clearance hole were both cut
# from zf-6/zf-1 as if the dome's radius were 6, not FOOT_D/2 = 13, so the hole opened
# 7 mm INSIDE the solid and never broke through the sole at all.
FOOT_CB_R  = 3.2                      # head pocket in the sole, D6.4 over an M3 socket
                                      # head's D5.5 - TPU prints holes tight
FOOT_CB_Z  = 8.0                      # pocket ceiling this far BELOW FOOT_Z.  That annulus
                                      # is the head's bearing face, and the 5 mm of pocket
                                      # under it recesses the head ~2 mm above the sole so
                                      # the metal never reaches the ground.
FOOT_BOLT_L = 30.0                    # M3 x 30, the next standard length over the span
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
# 3S2P pack: 6 x 21700, two layers of three, cells along X.  The cradle is DERIVED from
# the cell, not styled.  At the old literal 67 mm the four fins left a 19.5 mm channel for
# a 21.3 mm cell - the pack it was drawn for did not exist.  CELL_D and CELL_L are the
# datasheet MAXIMA of a Molicel INR21700-P42A over its wrap; a cell that measures bigger is
# a cell to re-measure, not a number to shave here.
CELL_D, CELL_L = 21.3, 70.2           # 21700, at its maximum over the wrap
CELL_FIT       = 0.4                  # slip fit, per cell, into its channel
BATT_FIN       = 2.4                  # printed separator between cells
BATT_TAB       = 4.0                  # welded nickel + insulation, at each cell end
BATT_PITCH     = CELL_D + CELL_FIT + BATT_FIN
BATT_L = CELL_L + 2*BATT_TAB          # clear length between the end stops
BATT_W = 3*BATT_PITCH + BATT_FIN      # fin-cell-fin-cell-fin-cell-fin, overall
BATT_H = 3.0 + 2*(CELL_D+CELL_FIT)    # tray floor + two layers
BMS_L, BMS_W, BMS_H    = 64.0, 27.0, 13.0
DECK_BOSS_R  = 5.8                    # tray boss: fat enough to swallow an M3 nut slot
DECK_NUT_DZ  = 6.0                    # ... its floor, below the boss top
# Deck screws as (x, |y|), once: chassis_bottom grows a boss under each, chassis_top drills
# each.  The mid pair sits at |y| = 41 and the corners at 38, and that is not a style: the
# corner bosses stand beyond the pack's end stops, but the mid pair stands BESIDE the pack,
# and at |y| = 38 a DECK_BOSS_R boss reaches y = 32.2, i.e. 2.7 mm inside the outer cell.
# There is nowhere else to put it - the tray floor between the cradle and the side wall is
# 5.9 mm, an M3 nut needs 9.5 - so the boss is pushed out until it clears the cell and
# clipped back to the body's own face, and its nut channel opens along +x, because inboard
# is the battery bay.  It blocks the side cable channel at x = +-18; cables cross over the
# pack (3.6 mm under the deck) or out through the side ports.
DECK_SCREWS  = ((-52.0, 38.0), (-18.0, 41.0), (18.0, 41.0), (52.0, 38.0))
# Rear connector panel: the pack's three ways out of the tray, around the bus window.
#   XT60  master disconnect / bench supply, on the pack's fused P+ ;
#   XT30  charge, and it is deliberately the SMALLER XT - a charger physically cannot be
#         plugged into the bus, which is the whole reason for two different shells;
#   a plain pass-through for the 3S JST-XH balance lead, which lives outside the tray so
#         the pack can be metered without opening the robot.
# Each XT sits in a pocket in a locally thickened wall and goes in from INSIDE, before the
# deck.  The outer PANEL_LIP_T of wall is left as a PANEL_LIP lip all round, and that lip -
# not glue - is what takes the unplug force, which on an XT60 is the big one.  The mating
# half therefore stands PANEL_LIP_T proud of the wall; XT pins are ~7 mm long against ~5 mm
# of engagement, so it still seats.  Nothing threads into plastic here either.
# The XT60 sits BELOW the window rather than beside it, and that is geometry, not taste:
# the clear strip between the window's edge at |y| = 16 and the rear deck boss's inboard
# face at |y| = 32.2 is 16.2 mm, and an XT60 needs 16.5.
PANEL_XT60   = (16.5, 8.5)            # body over the moulding, + fit         **verify**
PANEL_XT30   = (12.0, 6.6)            #                                       **verify**
PANEL_BAL    = (13.0, 6.0)            # JST-XH 3S plug, passing through       **verify**
PANEL_T      = 8.0                    # pocket depth, from the wall's outer face
PANEL_LIP    = 0.8                    # ... the lip left at the outer face, all round
PANEL_LIP_T  = 1.5                    # ... and how thick that lip is
PANEL_AT     = ((0.0, -16.0, PANEL_XT60),     # (y, z, size) - XT60 under the window,
                (25.0, 0.0, PANEL_XT30))      # XT30 in the strip beside it
PANEL_BAL_AT = (-24.0, 0.0)           # ... balance lead, in the strip on the other side
OPI_X        = -22.0                  # Orange Pi 5 Pro board centre on the deck
OPI_HOLES    = (92.0, 54.0)
OPI_STAND_R, OPI_STAND_H = 4.8, 7.0   # standoff: r fits an M2.5 nut slot, h clears the nut
OPI_NUT_DZ   = 1.5                    # ... its floor, above the deck's top face
# The Orange Pi as an ENVELOPE rather than as a hole pattern.  100 x 62 is the board;
# 20 mm is the stack allowance over the deck's own standoffs - board, its connector row and
# a heatsink.  It is here, in the model, because it is a keep-out that two other things
# now have to respect: the mass box every exporter builds for ELECTRONICS_KG (this file's
# section 4 rule - anything the robot carries lives once, here), and gps_mount, whose arms
# arch over it.  It used to be a literal on each side and they had already drifted - 92 x
# 62 x 20 at z 29..49 in export_sim.py against 100 x 62 x 18 at z 28..46 in the ROS 2
# generator.  **verify** with the hole pattern, off a real board.
OPI_BOX      = (100.0, 62.0, 20.0)

# The IMU, as a payload: a BMI088 breakout, and WHERE it is bolted is a model constant.
# Both sim exporters emit an `imu` site at it, and rl/checks/imu_placement.py measures what
# the choice costs.  An accelerometer rigidly offset by r from the site the model calls
# `imu` reads omega x (omega x r) + alpha x r on top of gravity, and on the 0.2 m/s trot
# that already exists that term reaches 9.0 m/s2 - 42 deg of apparent tilt - for a board
# out on the deck beside the Pi.  Once the site and the board agree it is not an error at
# all; what is left is that the swing the policy observes, and every residual mounting
# error with it, scale with |r|.  So the board goes as close to the body origin as the bay
# allows.  It lives here rather than in either exporter for the same reason the servo mass
# and the MJ_* block do: generate_model.py emitted a literal (0, 0, 0) against
# export_sim.py's BODY_Z1, so the two files described robots whose IMUs were 25 mm apart -
# the same defect a fourth time, and the one rl/ was reading.
#
# The bay decides the rest, and it is tight.  On the centreline the only opening is the
# deck's own window at x +-16, |y| <= 34; under it the pack's top is at BODY_Z0+BATT_H =
# 21.4 and the deck's underside is at BODY_Z1 = 25.  That is 3.6 mm; the board and its
# components are 2.8 of it; and the deck's 4 mm above is not free space either - it is
# OPI_BOX's floor.  So the board bolts up under two tabs that bridge the window, component
# face DOWN, and the 0.8 mm left over is the whole margin.  This is the camera's slot
# again: treat every number here as load-bearing.  imu_clear() is what checks the pack,
# because the pack is a payload and interference() cannot see one.
IMU_BOARD    = (20.0, 15.0, 1.6)      # PCB: x, y, thickness          **verify** ref/imu/
IMU_STACK    = 1.2                    # components over the PCB, and HEADERLESS: a 2.54 mm
                                      # pin header is 8.5 mm and misses this bay by 3x.
                                      # Solder to the pads.           **verify** ref/imu/
IMU_HOLE_P   = 15.0                   # M2.5 mounting holes, on x     **verify** ref/imu/
IMU_X, IMU_Y = 0.0, 0.0               # the centreline - as near the body origin as the
                                      # bay goes, which is what the whole block is for
IMU_TAB      = (9.0, 10.0)            # the tabs that carry it: reach in x from the window
                                      # wall, width in y.  The reach is set by IMU_HOLE_P:
                                      # the hole at x = 7.5 has to land on solid tab.
IMU_TAB_T    = DECK_T - M25_NUT_H - NUT_CLR    # 1.75.  The tab takes the deck's bottom,
                                      # the nut channel takes exactly the rest of DECK_T
                                      # and comes out flush with the top face, so nothing
                                      # here reaches into OPI_BOX.
IMU_Z0       = BODY_Z1                # ... and so the PCB's top face IS the deck's own
                                      # underside plane.  Not a coincidence: it is what
                                      # the line above leaves.
# LiDAR pedestal.  LIDAR_X is shared: chassis_top drills the bolt circle at it and
# lidar_mount is built on it, and they used to be two independent literals.
# LIDAR_BASE_R is set by the Orange Pi standoffs, not by the pedestal: at the old 30.0 the
# base disc and the two standoffs at (OPI_X+46, +-27) shared 95 mm3 of solid.  26.0 still
# covers the bolt circle with a 1.8 mm rim and clears the standoffs by 1.6 mm.
#
# TWO bolt circles, and they are deliberately different.  LIDAR_BC is ours: pedestal down
# onto the deck, on the same 45 deg rays as the four legs so every screw is under a post.
# LIDAR_L2_* is the sensor's, measured off the L2 mechanical drawing in the Unitree
# manual (Installation Dimensions, p.10) - 4 x M3 BLIND 6 mm deep on a @51 circle at
# 22.5 deg + k*90, inside a @60 spigot on a @75 base.  Those M3s are cut in the L2's own
# base, so this is the second joint on the robot - after the servo hubs - where a screw
# threads into stock hardware instead of a nut_slot.  Nothing threads into plastic here
# either: the pedestal side is a clearance hole.
#
# The two circles cannot be merged.  @51 for the deck screws would need the base disc out
# at r=30, which is exactly the Orange Pi standoff clash noted above; and turning our legs
# to 22.5 deg would push the deck holes to x = 42 + 22.5*cos(22.5) = 62.8, through the
# body's own front face at 63.  So the L2 screws live in the top flange alone, and the
# flange grew to r=32 to carry them with a 4.8 mm rim outside the hole.
#
# WHY THE SEAT IS TILTED, AND WHY IT IS NOT A MAST.
# The L1/L2 do not scan a band around themselves - they scan a HEMISPHERE ABOVE
# themselves.  The manual is explicit: "360 * 90 deg hemispherical ultra-wide-angle scan,
# which can measure the three-dimensional space ABOVE the radar".  Vertical FOV runs from
# the sensor's own base plane up to its axis; NEGA mode (the factory default) buys 6 deg
# BELOW that plane and nothing more.  Two things follow, and they are the whole design:
#
#   1. Height is worthless.  A mast helps a sensor that scans outward; this one is blind
#      below its own base whatever the altitude.  Standing upright at LIDAR_SEAT_Z = 73 it
#      met the ground 2.5 m in front of the dog - the pedestal was 38 mm of raised centre
#      of gravity buying literally no field of view.  The only thing height must do is keep
#      the robot's own bodywork out of the cone, which is one inequality, below.
#   2. Tilt is everything.  Leaning the sensor forward by LIDAR_TILT drops the forward rim
#      of the cone by exactly that angle, and that - not altitude - is what puts ground
#      under the dog's nose.
#
# 45 deg, because that is where the sensor's DENSEST ring lands on the horizon (the manual
# notes point density is highest at the centre of the vertical FOV, i.e. 45 deg off the
# base plane).  The horizon is where the things that stop a walking robot live - table
# legs, door frames, thresholds - so they get sampled best, while the lower rim still
# reaches the ground 147 mm in front of the leading foot in NEGA, one to two strides of
# lead for a rolling elevation map.  Leaning it further keeps buying near ground and starts
# spending the rear hemisphere on bare sky; leaning it less gives that back and pushes the
# near edge out past 300 mm.
#
# LIDAR_SEAT_Z is DERIVED, not styled.  No static part of the robot may sit above the seat
# plane, i.e. for every body point  z + (x - LIDAR_X)*tan(tilt) < LIDAR_SEAT_Z.  The
# binding point is the deck's own stiffening lip - the top of the robot - at (63, z=35),
# giving 35 + 21*tan45 = 56.0.  60.0 keeps 4 mm of margin on that and leaves 31 mm under
# the seat for the L2's RJ45 to turn down into the cable core, which is the other floor on
# this number.  Change LIDAR_TILT and this has to be re-derived; mini_dog.py checks it.
#
# LIDAR_X is pinned and is NOT a free choice: the base disc has to clear the Orange Pi
# standoffs at (24, +-27) behind it (LIDAR_X >= 38.8) and its own deck bolts have to land
# on a deck that ends at 63 (LIDAR_X <= 44.1).  42.0 sits in the middle of that 5 mm
# corridor.  Do not "move the LiDAR forward" - there is nowhere to move it to.
LIDAR_X      = 42.0
LIDAR_TILT   = 45.0                   # nose-down, about +y.  See above.
LIDAR_SEAT_Z = 60.0                   # seat plane on the pedestal axis.  Derived, see above.
LIDAR_BC, LIDAR_N          = 45.0, 4
LIDAR_L2_BC, LIDAR_L2_ANG  = 51.0, 22.5   # measured: Unitree L2 base, 4 x M3 v6
LIDAR_L2_THREAD = 6.0                     # ... usable thread depth in the L2
LIDAR_L2_BOX = (75.0, 75.0, 65.0)         # ... and its envelope, same drawing
LIDAR_OPT    = 44.5                       # ... and the 44.50 on its side view: the height
                                          # its scan core sits at, up its own axis from
                                          # the seat.  Every FOV number here is measured
                                          # from that point, not from the seat.
LIDAR_BASE_R, LIDAR_BASE_T = 26.0, 6.0
# ... and the disc is FLAT-CUT in front, at LIDAR_BASE_FLAT.  A @52 disc centred on
# LIDAR_X reaches x = 68, which put 6 mm of pedestal directly over the only slot the
# camera fits in - see the camera block.  Nothing needs that material: the deck bolts are
# on a @45 circle whose front pair sits at x = 57.9, and the four pedestal legs at r = 21
# reach x = 63.35, so a flat at 63.5 leaves the legs untouched and still keeps 5.6 mm of
# rim in front of the bolt.  It costs 105 mm2 of a 2124 mm2 disc.
LIDAR_BASE_FLAT = 63.5
LIDAR_TOP_R,  LIDAR_TOP_T  = 32.0, 7.0
LIDAR_LEG_R,  LIDAR_LEG_D  = 21.0, 13.0
LIDAR_CORE_R = 11.0                   # cable core straight up the middle
LIDAR_NUT_Z  = (8.0,)                 # nut-slot floor above the deck, for the M3 that
                                      # comes up from under the deck.  There is no second
                                      # nut any more: the L2 screw threads into the L2.

# The L2 as a SENSOR rather than as a lump of mass.  Everything above describes where the
# thing is bolted; these are what it sees, and they are here for the same reason the
# masses are: lidar.py, export_sim.py and ../ros2/.../generate_model.py all model the scan
# and none of them may keep its own copy of a number that belongs to the sensor.
#
# LIDAR_FOV is the "360 x 90" of the catalogue read as what it is - a cone of half-angle
# 90 deg about the sensor's own axis, from that axis down to its base plane.  NEGA, the
# factory default, buys 6 more degrees BELOW that plane and nothing else; lidar_fov_clear()
# lidar_fov_clear() has always measured against 96; it just used to spell it inline.
#
# The other three are catalogue figures, not measurements off a part in ref/, and they are
# marked **verify** in README.md until someone reads them off a real L2: point rate,
# range window and range accuracy.  Nothing structural depends on them - they set how
# dense and how noisy the simulated cloud is - but a perception result quoted from this
# model is only as good as they are.
LIDAR_FOV, LIDAR_FOV_NEGA = 90.0, 96.0    # scan cone half-angle about the axis, deg
LIDAR_RATE   = 21600.0                # points per second                    **verify**
LIDAR_R_MIN, LIDAR_R_MAX = 50.0, 30000.0  # mm, usable range window          **verify**
LIDAR_SIGMA  = 20.0                   # mm, 1-sigma range noise (+-2 cm spec) **verify**
# WHY THERE IS NO LIDAR GUARD.
# There was one, and it is gone on purpose - `git log` has the shape if it is ever wanted
# back.  Two things killed it and the second is the honest one:
#
#   1. You cannot cage this sensor anyway.  Its FOV is a hemisphere referenced to its own
#      base plane, so EVERY bar above that plane is a permanent blind stripe, and the
#      manual is blunt: "do not block its FOV.  Even installing a transparent glass plate
#      on the optical window will affect the performance".  The only free space is the
#      wedge UNDERNEATH the cone, so the guard could only ever be a low bow ahead of the
#      nose - never a cage over it.
#   2. That bow did not protect much.  hip_bracket already reaches x = 114.3, level with
#      the L2's nose at 114.5, so the front legs take a flat wall at the same moment the
#      sensor does; and the bow could not cover a bare horizontal edge at the sensor's own
#      height - a table top, a shelf - because guarding that means a bar dead ahead, in
#      the cone.  What it did cover was the case of something reaching below ~205 mm with
#      clear air above it.
#
# What removing it bought: 11.1 g off the very nose, which on this robot is worth more
# than it sounds (see README, "Terrain feedback" - 11 g there moved the flat trot 778 ->
# 547 mm), a clear lower half of the camera frame, and two M3 x 12 back where the deck's
# front screws used to need M3 x 24.  If it comes back it has to clear the camera, which
# now occupies the slot its inboard arms used to fly through.


# =====================================================================================
# GPS - u-blox NEO-6M on the GY-NEO6MV2 carrier, with the 25 x 25 ACTIVE ceramic patch
# it ships with (the patch carries its own LNA, biased over the coax by the module; there
# is nothing mechanical to it beyond "it is 25 x 25 and it must see sky").
#
# WHERE IT CAN GO, WHICH IS ALMOST NOWHERE
# A patch antenna is the same kind of sensor as the L2: it looks at a hemisphere ABOVE
# itself and everything that stands over it is a permanent hole in its view.  So it wants
# the highest flat spot on the robot, and this robot has none free:
#
#   * the deck's top face is the Orange Pi.  OPI_BOX spans x -72..28, |y| <= 31, z 29..49
#     - which is the whole of the deck between its two stiffening lips,
#   * ahead of the Pi is the LiDAR pedestal (base disc r = 26 at x = 42) and then the L2,
#   * the strips outboard of the Pi are 12 mm wide, against a 25 mm patch,
#   * and anything that clears all of that by standing tall lands in the L2's own cone.
#
# The one thing that IS free is the volume above the Pi and behind the L2, and it is free
# for a reason worth writing down: the L2's cone is referenced to its own base plane, and
# that plane leans LIDAR_TILT forward, so "up and behind" is BELOW it.  At the mast's
# worst vertex, z + (x - LIDAR_X)*tan45 = -9 against the 60 the seat sits at - 69 mm of
# margin, where the deck lip in front has 4.  lidar_fov_clear() charges the real 96 deg
# NEGA cone against it on every build, exactly as it does the camera.
#
# So: a trestle on the deck's REAR pair of boss screws, the trick the old lidar_guard
# with the front pair - it takes no new holes anywhere, those two screws just grow from
# M3x12 to M3x24 - arching over the Pi to a platform at GPS_SEAT_Z.  The receiver sits on
# the platform and the patch sits on the receiver, which is how the module ships.
#
# THE KINK IN THE ARMS IS NOT STYLING.  Two constraints fix it and they pull opposite ways:
#   * no run of an arm may lean more than 45 deg off vertical, or the part stops printing
#     without support - the same rule that shapes lidar_mount,
#   * and the arm has to be clear of OPI_BOX, whose top corner is at (|y| = 31, z = 49),
#     before it gets inboard of it.  A straight arm from the pad cannot: at 45 deg its
#     lower edge passes that corner 8 mm too low, whatever the platform's width.
# So the arm goes straight UP out of its pad first and makes exactly one 45 deg run
# inboard, and where it turns is derived, not chosen.  A rod leaning 45 deg carries its
# lower edge GPS_ROD/sqrt(2) = 2.47 mm below and inboard of its axis, which puts the edge
# at z = knee + 2.06 as it crosses |y| = 31: the knee goes at 48 and the solid clears the
# envelope by 1 mm.  That is checked and not asserted - gps_clear() intersects the real
# part with the real envelope on every build, the way interference() does the body parts.
# From the knee, 45 deg buys exactly as much inboard reach as it buys height, so the
# platform's half-width and its height are ONE number: land the arms at |y| = 24 and the
# underside has to be at 48 + 14 = 62.  Widen the platform and the mast gets shorter and
# heavier; narrow it and it gets taller and lighter.  24 is where it stops: past that the
# platform is wider than the deck's own boss pair and the arms start leaning outward.
#
# What it costs the antenna, honestly: the L2 blocks the forward sky below ~27 deg
# elevation and the Pi's connector row blocks a little more of it, so this is a receiver
# with a mask over one azimuth sector, not a survey antenna.  Nothing about the robot can
# fix that - the other sensor is bigger than this one and it was here first.  What the
# height does buy is 28 mm of separation from the Pi, which is the part that matters at
# 1.575 GHz: a USB 3 stack under a patch antenna is a well documented way to lose a fix.
GPS_X, GPS_Y = -52.0, 38.0            # the deck's rear boss pair - the mast's two feet
GPS_PAD_R, GPS_PAD_H = 4.8, 10.0      # r keeps the pad clear of the deck's stiffening lip
GPS_ROD      = 3.5                    # arm radius
GPS_KNEE     = (38.0, 48.0)           # (|y|, z) the arms turn inboard at - derived, see above
GPS_LAND     = 24.0                   # |y| where they meet the platform
GPS_PLATE    = (40.0, 52.0, 3.0)      # platform x, y, t
GPS_SEAT_Z   = GPS_KNEE[1] + (GPS_KNEE[0]-GPS_LAND) + GPS_PLATE[2]   # derived - see above
GPS_BOARD    = (36.0, 26.0, 1.6)      # GY-NEO6MV2 PCB                        **verify**
GPS_ANT      = (25.0, 25.0, 8.0)      # its active ceramic patch              **verify**
GPS_TIE      = (3.4, 1.6)             # cable-tie slot through the platform
GPS_TIE_X    = 14.0                   # ... at +-this, i.e. outside the 25 mm patch
GPS_PHASE    = 6.0                    # patch phase centre, up from the patch's base plane
GPS_STACK    = (GPS_BOARD[0], GPS_BOARD[1], GPS_BOARD[2]+GPS_ANT[2]+2.5)   # mass envelope

# Camera.  Weinan WN-L2101.K203L: a Sony IMX415 (1/2.8", 8 MP) behind a fixed-focus M12
# lens, USB 2.0 UVC, digital mic on the board.  It is here to put a NAME on what the L2
# already puts a SHAPE on: the lidar returns geometry and no identity, and recognising a
# face is the one job on this robot that wants pixels instead of points.
#
# Everything but the pose is off the vendor's Product Dimensions drawing.  The module is a
# 90 x 15 mm STRIP with the lens near one end - not the 38 x 38 square the same seller's
# other modules are - and that one fact drives the whole mount.
CAM_BOARD  = (90.0, 15.0, 1.6)        # PCB length x width x thickness
CAM_LENS_D, CAM_LENS_H = 14.0, 16.2   # M12 holder OD, and its stand-off from the PCB face
CAM_LENS_U = 70.59                    # optical axis, from the connector end of the board
CAM_EAR_P, CAM_EAR_D = 18.0, 2.2      # the two mounting holes straddling the lens
CAM_CONN   = (12.0, 5.2)              # USB tail connector: along the board, and its height
CAM_TAIL   = -1.0                     # which way the 70.6 mm tail runs.  -1 = -y.

# WHERE IT GOES, AND WHY THERE IS EXACTLY ONE PLACE.
# The nose is boxed in on four sides and every one of them is a measured part, not taste:
#
#   floor    the hip-roll cradles.  roll_module is solid over |y| <= 14 from x = 63 out to
#            x = 106.5 and its root gusset is solid over |y| <= 49 from x = 63 to 67.5,
#            both topping out at CAM_LEDGE.  Nothing sits below that in front of the body.
#   ceiling  the LiDAR pedestal's base disc, z = 29..35, reaching x = 68.  It had to be
#            flat-cut - see LIDAR_BASE_FLAT - because the slot under it was 13.6 mm and
#            the module's board is 15.0.  Above the cut the ceiling is the L2's own case,
#            at z = 36 over the lens.
#   back     the chassis front face at x = 63.
#   front    the hip-roll fork's REAR ARM at x = ROLL_X + FORK_Y0 = 68.1.  That arm is a
#            disc about the roll axis; over the roll ROM it sweeps everything within
#            r = 34 of it, so 68.1 is a wall for anything near (|y| = 36, z = 0).
#
# So the board lives in a 5 x 15 mm slot and only the LENS goes past x = 68.1 - it may,
# because on the centreline its closest approach to either roll axis is 36.3 mm, outside
# the 34 the fork sweeps.  That is not a guess; the alternatives were built and measured
# against the real solids and the real ROM:
#
#   landscape at z = 23, x = 68..72  hip roll clashes from +5 deg (9.5 mm3) and reaches
#                                    224 mm3 by +90 - the 70.6 mm tail crosses the fork's
#                                    23 < r < 34 annulus over y = -64 .. -31.
#   under the belly, z = -20/-32/-45 the front legs sweep it at every height: thigh, shin
#                                    and foot all clash from +30 deg of hip pitch.
#
# CAM_TILT is small for the same reason the board is where it is: the slot is 5 mm deep
# fore-aft and a tilted board eats CAM_BOARD[1]*sin(t) of that.  6 deg costs 1.6 mm and
# leaves ~1 mm of wall front and back; 15 deg does not fit at all.  What that buys, and it
# is worth being honest about it: the lens ends up 209 mm off the ground with a 52 deg
# vertical FOV, so a 1.6 m face is in frame from 2.2 m out and is still >= 90 px across -
# enough to recognise - to about 4 m.  Closer than 2.2 m the head leaves the top of the
# frame.  More tilt would fix that and does not fit; a shorter board would.
CAM_X, CAM_Z = 66.8, 24.5             # the PCB's FRONT face, on the optical axis
CAM_TILT     = 6.0                    # nose-up, about +y
CAM_OPT      = 12.0                   # entrance pupil, up the axis from the PCB  **verify**
CAM_FOV_D    = 90.0                   # the fitted lens, DIAGONAL                 **verify**
CAM_PIX      = (3840, 2160)           # the mode the pipeline runs; H and V FOV follow
CAM_RATE     = 15.0                   # frames/s at CAM_PIX over USB 2.0 MJPEG    **verify**
CAM_LEDGE    = S_W/2 + SLEEVE_W       # 15.36 - roll_module's own top face, and the only
                                      # flat surface anywhere near the camera.  It is a
                                      # derived number there too, spelled as a literal.
# The mount is a C-section standing on that ledge.  It cannot grip the FRONT of the board's
# lower half (0.5 mm to the fork arm) and it cannot put a nut behind the board (2.9 mm of
# depth, an M3 nut is 5.85 across), so the board slides in endwise and is trapped: a slot
# in the shelf below, a wall in front of its upper half, a skirt behind.  The two M3 both
# live at the +y end, past the board's short end, where the boss can be full size; the far
# end is keyed against sliding by a printed tongue in a pocket, which fastens nothing and
# so needs no nut.  Both screws come down into nuts in slots in the gusset, opening
# forward - open air under the chin, and the only face still reachable with the board in.
CAM_BACK, CAM_BACK_HI = 63.0, 63.6    # mount back plane, below / above the deck's top
CAM_FRONT    = 68.0                   # ... and its front, 0.1 inside the fork arm plane
CAM_END      = (31.0, -73.0)          # the channel's two ends, in y
CAM_FOOT_X   = 65.5                   # the two M3, outboard of the void in the gusset
CAM_FOOT_Y   = (23.0, 29.0)
CAM_NUT_DZ   = 8.0                    # nut-slot floor, below the ledge: 5 mm of gusset
                                      # over the nut, which is what takes the preload
CAM_KEY      = (-45.0, 6.0, 3.0)      # locating tongue: y centre, length, depth

STAND_PITCH, STAND_KNEE = -22.0, 46.0

# =====================================================================================
# 4. mass / drive - the single source for fea.py, export_sim.py, the BOM AND the ROS 2
#    description generator in ../ros2/smalldog_description/scripts/generate_model.py.
#    Nothing downstream may keep its own copy of these; that is how the servo mass ended
#    up as 55 g on this side and 60 g in the ROS 2 model.
# =====================================================================================
PETG_RHO          = 1.27          # g/cm3 - filament, from the spool
TPU_MAT_RHO       = 1.22          # g/cm3 - TPU 95A filament
# Fill factor = printed mass / solid-volume mass.  MEASURED, not assumed: every part was
# sliced in OrcaSlicer 2.4.2 (0.2 mm layer, 0.8 mm line, gyroid, walls/infill per the note
# in PARTS, no brim, no support) and the factor is that plate's filament mass over
# volume x filament density.  These parts are thin-walled, so the walls - not the infill -
# set the mass and the factor sits near 1; the single 0.55 that used to stand here was
# wrong by 330 g of robot.  Re-measure after any wall-count, line-width or reprofiling
# change - out/gcode/summary.json is the slicer side of this number.
PRINT_FILL        = {"chassis_bottom": 0.93, "chassis_top": 0.80, "lidar_mount": 0.85,
                     "hip_bracket_A":  0.95, "hip_bracket_B": 0.95,
                     "thigh_A":        0.97, "thigh_B":       0.97,
                     "shin_A":         0.92, "shin_B":        0.92,
                     "servo_gauge":    0.94, "foot":          0.65}
# gps_mount is deliberately absent: it has not been sliced yet, so part_rho() gives it
# PRINT_FILL_MEAN.  Slice it and put the measured factor in the table.
PRINT_FILL_MEAN   = 0.92          # PETG parts, mass-weighted - the fallback for an
TPU_FILL_MEAN     = 0.65          # unmeasured part and for consumers that carry a union
PRINT_RHO         = PETG_RHO * PRINT_FILL_MEAN   # g/cm3 - solid volume x this
TPU_RHO           = TPU_MAT_RHO * TPU_FILL_MEAN  # the feet
SERVO_KG          = 0.060         # ST3215 incl. both hubs and the bolts - **verify**: not in
                                  # ref/, vendor figure only.  Weigh one before trusting it;
                                  # 12 of them are a third of the robot.
N_SERVO           = 12
BATTERY_KG        = 0.42          # 3S2P, 6x21700
ELECTRONICS_KG    = 0.25          # Orange Pi 5 Pro / BMS / wiring
LIDAR_KG          = 0.230         # Unitree L2 on the pedestal - confirmed, L2 manual
                                  # Parameter Specifications: 230 g, 75x75x65 mm, 12 V 10 W
CAMERA_KG         = 0.012         # IMX415 module: PCB, M12 holder, lens, connector.
                                  # **verify** - vendor gives no mass; weigh one.  It
                                  # sits at the very nose, which is the worst place on
                                  # this robot for mass - see README, "Terrain feedback".
IMU_KG            = 0.003         # BMI088 breakout, headerless, with its six wires.
                                  # **verify** - ref/imu/README.md is a transcription and
                                  # this is the least certain line in it.  It is also the
                                  # smallest mass here, and 3 g is enough to move the flat
                                  # trot: see CLAUDE.md step 6 before reading a distance.
GPS_KG            = 0.025         # GY-NEO6MV2 + its 25x25 active patch + the lead.
                                  # 22 g is the vendor figure for the pair - **verify**,
                                  # like every other number on this module: it is a bazaar
                                  # part, not a documented one.
TPU_PARTS         = ("foot",)     # printed in TPU_RHO, everything else in PRINT_RHO
SERVO_STALL_NM    = 2.94          # 30 kg*cm at 12 V
SERVO_NOLOAD_RADS = 4.71          # 0.222 s / 60 deg at 12 V

# MuJoCo joint feel.  NOT measured - the ST3215 gearbox is a black box and these
# are plausible values that keep the model stable at 2 ms.  They live here and
# nowhere else: both sim exporters (export_sim.py and the ROS 2
# generate_model.py) read them from this block, because they used to each carry
# their own literals and silently diverged - 0.5/0.01/0.05/20 against
# 0.12/0.008/0.02/25 - which is the servo-mass failure this file's mass block
# already records, repeated one section down.  rl/checks/check_model.py is what
# caught it.
#
# The surviving values are the ROS 2 set, deliberately: every gait baseline in
# CLAUDE.md was measured against those, so adopting them leaves the ROS 2 model
# byte-identical and re-baselines nothing.  When rl/ fits the real actuator these
# become its initial guess, not a second opinion - see rl/actuator.py, whose
# Params.J_m is already this same 0.008.
MJ_DAMPING        = 0.12          # N*m*s/rad at the joint
MJ_ARMATURE       = 0.008         # kg*m2, reflected rotor+gearbox inertia
MJ_FRICTIONLOSS   = 0.02          # N*m
MJ_KP             = 25.0          # position-actuator gain
MJ_DAMPRATIO      = 1.0           # critically damped.  export_sim.py carried no
                                  # dampratio at all, which is what made its kp
                                  # incomparable with the ROS 2 one rather than
                                  # merely different.

# =====================================================================================
# helpers
# =====================================================================================
def W(sh): return cq.Workplane(obj=sh)
def part_rho(name):
    """g/cm3 for a named part - measured fill factor x filament density."""
    rho = TPU_MAT_RHO if name in TPU_PARTS else PETG_RHO
    mean = TPU_FILL_MEAN if name in TPU_PARTS else PRINT_FILL_MEAN
    return rho * PRINT_FILL.get(name, mean)
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

def nut_slot(at, out, up=(0,0,1), af=M3_NUT_AF, h=M3_NUT_H, back=3.6, run=20.0):
    """Side-loaded nut pocket: a channel af+NUT_CLR wide - so its two walls hold the nut's
    flats and it cannot turn - running from `back` behind the bolt axis out to `run` in
    front of it, along `out`.  `at` sits on the bolt axis on the channel's floor and `up`
    is the bolt axis, the way the channel is dug from there; which of the channel's two
    faces carries the load depends on which end the screw comes in from, so leave enough
    material on both (~3 mm here).  Nothing on this robot threads into plastic: every M3
    and M2.5 that is not going into the stock aluminium hubs lands in one of these, and
    `out` is always chosen so the open end of the channel is reachable at the point in the
    assembly order when that nut goes in - which is what fixes the order in README.md."""
    s = bxc(-back, run, -(af+NUT_CLR)/2, (af+NUT_CLR)/2, 0.0, h+NUT_CLR)
    return mv(s, frame(at, xdir=out, zdir=up))
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

def sleeve(length=SLEEVE_LEN, wall=SLEEVE_W, window=True, lighten=True, clamp=True):
    x0, x1 = -S_AX-wall, S_L-S_AX+wall
    y0, y1 = -S_W/2-wall, S_W/2+wall
    s = bxc(x0, x1, y0, y1, -length/2, length/2)
    if clamp:                                         # thrust-clamp lug (see THRUST_* above)
        s = s.union(bxc(x0-THRUST_L, x0+2.0, -THRUST_YL, THRUST_YL,
                        -min(THRUST_Z, length/2), min(THRUST_Z, length/2)))
    s = s.cut(servo_case(CLR))
    if window:                                        # cable / connector escape
        s = s.cut(bxc(S_L-S_AX-1, x1+1, -CONN_W/2, CONN_W/2, -length/2-1, length/2+1))
    if clamp:
        for sg in (1, -1):
            yb = sg*THRUST_Y
            # clearance the whole way: the bolt is a jack screw, it must reach the case
            s = s.cut(cyl(M3_CLR, THRUST_L+wall+3.0, (x0-THRUST_L-1, yb, 0), axis=(1,0,0)))
            # nut channel, opening on the lug's own +-y face.  Its floor is THRUST_SEAT of
            # lug: the bolt pushes the case, so the nut is driven back onto that face.
            s = s.cut(nut_slot((x0-THRUST_L+THRUST_SEAT, yb, 0), (0, sg, 0), up=(1,0,0),
                               run=THRUST_YL-THRUST_Y+2.0))
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
    # No nut pockets here.  The screw is driven from the outside and threads into the
    # stock aluminium hub - the @2.5 holes in both plates are tapped, and there is nowhere
    # for a nut anyway: behind the driven hub sit 0.30 mm to the case top, behind the
    # passive one the 0.55 mm base recess.  A hex pocket would also take 2.2 of the 4.0 mm
    # arm exactly under the screw head, where the bolt load enters the part.
    # Screws: M2.5x6 driven (4.0 arm + <=2.5 hub), M2.5x7 passive (4.0 + 0.95 pedestal
    # + <=2.2 hub).  Longer bottoms out on the case - the vendor FAQ warns it burns servos.
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
    # 6x 21700 cradle: two layers of three, cells along X, low and central.  Four fins on
    # BATT_PITCH, so each of the three channels comes out exactly CELL_D+CELL_FIT wide.
    for i in range(4):
        y = -BATT_W/2 + BATT_FIN/2 + i*BATT_PITCH
        s = s.union(bxc(-BATT_L/2-1, BATT_L/2+1, y-BATT_FIN/2, y+BATT_FIN/2,
                        BODY_Z0+3.0, BODY_Z0+BATT_H))
    for x in (-BATT_L/2-3.0, BATT_L/2):                       # end stops
        s = s.union(bxc(x, x+3.0, -BATT_W/2, BATT_W/2, BODY_Z0+3.0, BODY_Z0+BATT_H))
    for x in (-24.0, 24.0):                                   # battery strap slots
        for y in (-BODY_W/2-1, BODY_W/2-WALL-1):
            s = s.cut(bxc(x-1.7, x+1.7, y, y+WALL+2, BODY_Z0+3.0, BODY_Z0+6.5))
    # BMS bay (front) and ESP32 + URT-1 bay (rear)
    for xc in (46.0, -46.0):
        s = s.union(bxc(xc-1.5, xc+1.5, -BMS_W/2-1.5, BMS_W/2+1.5, BODY_Z0+3, BODY_Z0+16))
    # deck bosses.  The deck screw lands in a nut, not in a printed thread: the boss is
    # drilled M3 clearance and carries a nut slot near its top, opening toward the middle
    # of the tray - the one direction that is open air with the deck off, which is when
    # the nuts go in.  The screw pulls the nut up against the slot's roof and the deck
    # down onto the boss top, so the 3 mm of boss above the slot is in compression.
    # The mid pair is the exception on both counts - see DECK_SCREWS: it is clipped to the
    # body's own side face, and its channel opens along +x, into the free strip beside the
    # pack, because inboard of it is the battery.
    for x, ay in DECK_SCREWS:
        for y in (-ay, ay):
            b = cyl(DECK_BOSS_R, BODY_Z1-(BODY_Z0+3.0), (x, y, BODY_Z0+3.0))
            s = s.union(b.intersect(bxc(-BODY_L/2, BODY_L/2, -BODY_W/2, BODY_W/2,
                                        BODY_Z0, BODY_Z1)))
            s = s.cut(cyl(M3_CLR, 15.0, (x, y, BODY_Z1-14.0)))
            out = ((1.0, 0.0, 0.0) if abs(y) > 40.0 else
                   (0.0, -1.0 if y > 0 else 1.0, 0.0))
            s = s.cut(nut_slot((x, y, BODY_Z1-DECK_NUT_DZ), out, run=DECK_BOSS_R+6.0))
    for y in (-BODY_W/2-1, BODY_W/2-WALL-1):                  # vents / side cable ports
        for x in (-34.0, 0.0, 34.0):
            s = s.cut(bxc(x-11, x+11, y, y+WALL+2, -4.0, 14.0))
    s = s.cut(bxc(-BODY_L/2-1, -BODY_L/2+WALL+1, -16, 16, -10, 10))
    # Rear connector panel - see the PANEL_* block.  Pad, then the pocket out of it, then
    # the lip's smaller opening through the wall's outer skin.
    xw = -BODY_L/2                                            # the wall's outer face
    for cy, cz, (w, h) in PANEL_AT:
        s = s.union(bxc(xw+WALL, xw+PANEL_T, cy-w/2-3.0, cy+w/2+3.0,
                        max(cz-h/2-3.0, BODY_Z0+3.0), cz+h/2+3.0))
        s = s.cut(bxc(xw+PANEL_LIP_T, xw+PANEL_T+1, cy-w/2, cy+w/2, cz-h/2, cz+h/2))
        s = s.cut(bxc(xw-1, xw+PANEL_LIP_T, cy-w/2+PANEL_LIP, cy+w/2-PANEL_LIP,
                      cz-h/2+PANEL_LIP, cz+h/2-PANEL_LIP))
    by, bz = PANEL_BAL_AT
    s = s.cut(bxc(xw-1, xw+WALL+1, by-PANEL_BAL[0]/2, by+PANEL_BAL[0]/2,
                  bz-PANEL_BAL[1]/2, bz+PANEL_BAL[1]/2))
    # Camera mount: two M3 come down through the gusset into nuts in slots that open
    # forward, on the gusset's own front face - open air under the chin, and the only face
    # still reachable once the module is in its channel.  CAM_FOOT_X is outboard of the
    # gusset's lightening void, so the nut's two walls are solid PETG.
    for fy in CAM_FOOT_Y:
        s = s.cut(cyl(M3_CLR, 20.0, (CAM_FOOT_X, fy, CAM_LEDGE-CAM_NUT_DZ-4.0)))
        s = s.cut(nut_slot((CAM_FOOT_X, fy, CAM_LEDGE-CAM_NUT_DZ), (1.0, 0.0, 0.0),
                           run=8.0))
    ky, kl, kd = CAM_KEY                      # ... and the pocket its far end keys into
    s = s.cut(bxc(CAM_FOOT_X-3.0-CLR, CAM_FOOT_X+3.0+CLR, ky-kl/2-CLR, ky+kl/2+CLR,
                  CAM_LEDGE-kd-CLR, CAM_LEDGE+1.0))
    return s.cut(env_all())

# =====================================================================================
# PART: chassis_top / lidar_mount
# =====================================================================================
def chassis_top():
    z0, z1 = BODY_Z1, BODY_Z1+DECK_T
    s = bxc(-BODY_L/2, BODY_L/2, -BODY_W/2, BODY_W/2, z0, z1)
    for x, ay in DECK_SCREWS:
        for y in (-ay, ay):
            s = s.cut(cyl(M3_CLR, 20, (x, y, z0-1))).cut(cyl(3.2, 2.2, (x, y, z1-2.2)))
    # Orange Pi 5 Pro standoffs.  M2.5 through the board, through the standoff, into a nut
    # in a slot opening outboard in y - fitted before the board goes on, and still the only
    # face you can reach once the deck is on the tray.  The M2.5 nut, not M3: the board's
    # own holes are 2.5, and its 5.0 across-flats leaves 2.2 mm of standoff wall.
    for sx in (-1, 1):
        for sy in (-1, 1):
            p = (OPI_X+sx*OPI_HOLES[0]/2, sy*OPI_HOLES[1]/2, z1)
            s = s.union(cyl(OPI_STAND_R, OPI_STAND_H, p)).cut(cyl(M25_CLR, OPI_STAND_H+2, p))
            s = s.cut(nut_slot((p[0], p[1], z1+OPI_NUT_DZ), (0.0, float(sy), 0.0),
                               af=M25_NUT_AF, h=M25_NUT_H, back=3.2,
                               run=OPI_STAND_R+6.0))
    for i in range(LIDAR_N):
        a = math.radians(360.0*i/LIDAR_N+45.0)
        s = s.cut(cyl(M3_CLR, 20, (LIDAR_X+LIDAR_BC/2*math.cos(a), LIDAR_BC/2*math.sin(a), z0-1)))
    s = s.cut(cyl(LIDAR_CORE_R, 20, (LIDAR_X, 0, z0-1)))   # the LiDAR cable, into the tray
    s = s.cut(bxc(-16, 16, -34, 34, z0-1, z1+1)).cut(bxc(58, 60, -26, 26, z0-1, z1+1))
    # IMU tabs, bridging the window just cut, on the centreline - see the IMU_* block.  The
    # board bolts UP against their underside, so its component face looks down at the pack
    # and nothing it carries reaches into OPI_BOX.  The M2.5 nut sits in the deck's own top
    # 2.25 mm in a channel opening toward x = 0 - open air inside the window, and the only
    # face still reachable at the moment it goes in, which is with the deck off the tray
    # and before the board.  That fixes its place in the assembly order in README.md.
    for sx in (-1.0, 1.0):
        xa, xb = sx*(16.0-IMU_TAB[0]), sx*16.0
        s = s.union(bxc(min(xa, xb), max(xa, xb), -IMU_TAB[1]/2, IMU_TAB[1]/2,
                        IMU_Z0, IMU_Z0+IMU_TAB_T))
        hx = sx*IMU_HOLE_P/2
        s = s.cut(cyl(M25_CLR, DECK_T+4.0, (hx, IMU_Y, IMU_Z0-1.0)))
        s = s.cut(nut_slot((hx, IMU_Y, IMU_Z0+IMU_TAB_T), (-sx, 0.0, 0.0),
                           af=M25_NUT_AF, h=M25_NUT_H, back=3.2, run=12.0))
    for y in (-BODY_W/2, BODY_W/2-3.0):                       # stiffening lips
        s = s.union(bxc(-BODY_L/2, BODY_L/2, y, y+3.0, z1, z1+6.0))
    for x, ay in DECK_SCREWS:                                 # ... notched at the mid pair:
        if ay > 40.0:                                         # a socket head at |y| = 41
            for sy in (-1.0, 1.0):                            # stands 0.8 mm proud of its
                y0 = sy*(BODY_W/2-3.0)                        # counterbore and reaches into
                s = s.cut(bxc(x-6.0, x+6.0, min(y0, y0+sy*3.0), max(y0, y0+sy*3.0),
                              z1, z1+2.0))
    return s

def lidar_pose():
    """The L2's own frame in robot coordinates: (seat point, unit axis).  The axis is the
    centre of the sensor's up-hemisphere, so it is also the extrinsic every consumer wants.
    export_sim.py and ../ros2/.../generate_model.py both read the sensor's position from
    here; each of them used to carry its own 42.0 literal and its own guess at the height."""
    t = math.radians(LIDAR_TILT)
    return (LIDAR_X, 0.0, LIDAR_SEAT_Z), (math.sin(t), 0.0, math.cos(t))

def lidar_com():
    """Centroid of the L2 itself - half its own height up its own axis from the seat."""
    (px, py, pz), (nx, ny, nz) = lidar_pose()
    h = LIDAR_L2_BOX[2]/2.0
    return (px+nx*h, py+ny*h, pz+nz*h)

def lidar_seat_min():
    """The lowest LIDAR_SEAT_Z that keeps the robot's own bodywork out of the L2's cone.

    The cone's floor is the sensor's base plane, so a body point occludes exactly when it
    is above that plane: z + (x - LIDAR_X)*tan(tilt) >= LIDAR_SEAT_Z.  The maximum of that
    expression over the static body is what this returns.  Only the static body counts -
    the legs sweep through the forward-down cone at every stride and no mount geometry can
    change that, which is why every quadruped masks its own legs in software.
    """
    t = math.radians(LIDAR_TILT)
    worst = 0.0
    for x, z in ((BODY_L/2, BODY_Z1+DECK_T+6.0),        # deck stiffening lip - the top
                 (BODY_L/2, BODY_Z1+DECK_T),            # deck itself
                 (OPI_X+OPI_HOLES[0]/2, BODY_Z1+DECK_T+OPI_STAND_H),   # Pi standoffs
                 (GPS_X+GPS_PLATE[0]/2, GPS_SEAT_Z)):   # GPS mast - tall, but
                                                     # far enough back to be under the
                                                     # tilted base plane by 68 mm
        worst = max(worst, z + (x-LIDAR_X)*math.tan(t))
    return worst

def rod(a, b, r):
    """A cylinder from a to b — the GPS mast is all rods and there was no helper."""
    d = tuple(q-p for p, q in zip(a, b))
    L = math.sqrt(sum(v*v for v in d))
    return cyl(r, L, a, axis=tuple(v/L for v in d))

def lidar_fov_clear(wp, nega=True):
    """Worst-case angle from the L2's axis over a solid's vertices, in degrees.

    The sensor sees a cone of HALF-angle 90 deg about its axis - the "360 x 90" in the
    catalogue is that hemisphere, axis to base plane - and 96 with NEGA, measured
    from its optical centre.  A point occludes exactly when it falls inside that cone, so
    a part is clear when its WORST vertex is still outside - and the returned number is
    the margin readers actually want: how many degrees of slack a part has before it
    starts eating the view the whole pedestal exists to buy.
    """
    (sx, sy, sz), n = lidar_pose()
    o = (sx+n[0]*LIDAR_OPT, sy+n[1]*LIDAR_OPT, sz+n[2]*LIDAR_OPT)
    worst = 180.0
    # tessellate, do not use Vertices(): a cylinder has vertices only on its end circles,
    # so a rod laid across the rim would sail through this check on its seam points while
    # its barrel sat inside the cone.
    for v in wp.val().tessellate(0.3)[0]:
        d = (v.x-o[0], v.y-o[1], v.z-o[2])
        L = math.sqrt(sum(q*q for q in d))
        if L < 1e-9:
            return 0.0
        c = sum(q*m for q, m in zip(d, n))/L
        worst = min(worst, math.degrees(math.acos(max(-1.0, min(1.0, c)))))
    return worst - (LIDAR_FOV_NEGA if nega else LIDAR_FOV)

def lidar_mount():
    """Pedestal.  Base disc + four @13 legs + a seat, and the seat is TILTED - it is the
    L2's mounting face, leaning LIDAR_TILT forward.  Why that is the whole point of the
    part is argued in the parameter block; here is what it does to the geometry.

    The legs are still vertical and still land on the same @45 circle at 45 deg, so the
    deck interface - four M3 up from underneath into nuts in slots that open radially
    outward - is untouched.  They are simply grown past the seat and then cut back to it,
    which leaves the rear pair tall and the front pair short and puts a 45 deg face on top
    of each.  Nothing on this part overhangs downward at less than the tilt angle, so it
    still prints on its base disc without support.

    Two bolt circles, and they are deliberately different.  Ours is @45 at 45 deg, every
    screw under a leg.  The sensor's is @51 at 22.5 deg, normal to the SEAT, 4 x M3 into
    the L2's own tapped holes - no nut, which is why the screw is M3x12: LIDAR_TOP_T of
    seat plus 5 mm of thread against the LIDAR_L2_THREAD it actually has.  A longer one
    bottoms in the blind hole and jacks the sensor off its seat without ever feeling loose.
    The two circles cannot be merged: @51 for the deck screws needs the base disc out at
    r = 30, which is the Orange Pi standoff clash the disc was shrunk to 26 to avoid.

    The cable core stays vertical and straight through the middle.  Cut through a 45 deg
    seat it opens as a @22 x 31 ellipse, still 6 mm clear of the nearest L2 bolt, and it
    drops the L2's three tails straight down into the tray instead of round a corner."""
    t  = math.radians(LIDAR_TILT)
    n  = (math.sin(t), 0.0, math.cos(t))
    dn = tuple(-v for v in n)
    seat = (LIDAR_X, 0.0, LIDAR_SEAT_Z)
    cx, z0 = LIDAR_X, BODY_Z1+DECK_T
    ztop = LIDAR_SEAT_Z + LIDAR_TOP_R*math.sin(t) + 6.0
    s = cyl(LIDAR_BASE_R, LIDAR_BASE_T, (cx, 0, z0))
    s = s.cut(bxc(LIDAR_BASE_FLAT, cx+LIDAR_BASE_R+2.0, -LIDAR_BASE_R-2.0,
                  LIDAR_BASE_R+2.0, z0-1.0, z0+LIDAR_BASE_T+0.5))   # room for the camera
    for i in range(LIDAR_N):
        a = math.radians(360.0*i/LIDAR_N+45.0)
        px, py = cx+LIDAR_LEG_R*math.cos(a), LIDAR_LEG_R*math.sin(a)
        s = s.union(cyl(LIDAR_LEG_D/2, ztop-z0, (px, py, z0)))
    s = s.union(cyl(LIDAR_TOP_R, LIDAR_TOP_T, seat, axis=dn))
    s = s.cut(cyl(400.0, 400.0, seat, axis=n))            # everything above the seat plane
    s = s.cut(cyl(LIDAR_CORE_R, 200, (cx, 0, z0-10)))
    for i in range(LIDAR_N):                              # pedestal -> deck, 45 deg
        a = math.radians(360.0*i/LIDAR_N+45.0)
        px, py = cx+LIDAR_BC/2*math.cos(a), LIDAR_BC/2*math.sin(a)
        s = s.cut(cyl(M3_CLR, 20.0, (px, py, z0-1)))
        for zn in LIDAR_NUT_Z:
            s = s.cut(nut_slot((px, py, z0+zn), (math.cos(a), math.sin(a), 0.0),
                               run=LIDAR_LEG_D/2+4.0))
    ex = (math.cos(t), 0.0, -math.sin(t))                 # the seat plane's own x
    for i in range(LIDAR_N):                              # L2 -> seat, 22.5 deg, normal to it
        a = math.radians(360.0*i/LIDAR_N+LIDAR_L2_ANG)
        r = LIDAR_L2_BC/2
        p = tuple(seat[k] + r*math.cos(a)*ex[k] + r*math.sin(a)*(0.0, 1.0, 0.0)[k] + n[k]
                  for k in range(3))
        s = s.cut(cyl(M3_CLR, LIDAR_TOP_T+2.0, p, axis=dn))
    return s

def opi_com():
    """Centroid of the Orange Pi stack's envelope - the keep-out gps_mount arches over and
    the point every exporter hangs ELECTRONICS_KG on."""
    return (OPI_X, 0.0, BODY_Z1+DECK_T+OPI_BOX[2]/2.0)

def imu_xyz():
    """The `imu` site in robot coordinates (mm): the BMI088's own package, at the centre of
    the board's component face - which looks down, so it is one PCB thickness below the
    tabs the board hangs from.

    Same contract as lidar_pose() and gps_pose(): export_sim.py and
    ../ros2/.../generate_model.py both read the site from here.  They did not, once - the
    ROS 2 generator wrote pos="0 0 0" while export_sim.py wrote BODY_Z1 - and rl/ loads the
    ROS 2 model, so the 25 mm went straight into the observation the policy trains on."""
    return (IMU_X, IMU_Y, IMU_Z0 - IMU_BOARD[2])

def imu_module():
    """The breakout itself, not a printed part - here for the same reason camera_module()
    is: interference() and the assembly have to see the thing that is actually bolted on,
    and neither can see a number in a table.  Component face down, per ref/imu/."""
    L, W_, T = IMU_BOARD
    return bxc(IMU_X-L/2, IMU_X+L/2, IMU_Y-W_/2, IMU_Y+W_/2,
               IMU_Z0-T-IMU_STACK, IMU_Z0)

def imu_clear():
    """mm3 of the IMU board inside the battery pack's envelope, plus the mm of air under
    it - the board's own binding constraint and the one thing interference() cannot see,
    because the pack is a payload and not a part (gps_clear() exists for the same reason
    against OPI_BOX).  Returns (overlap_mm3, gap_mm); the gap is what is actually thin."""
    top = BODY_Z0 + BATT_H
    pack = bxc(-BATT_L/2, BATT_L/2, -BATT_W/2, BATT_W/2, BODY_Z0+3.0, top)
    try:    v = imu_module().val().intersect(pack.val()).Volume()
    except Exception: v = 0.0
    return v, (IMU_Z0 - IMU_BOARD[2] - IMU_STACK) - top

def gps_pose():
    """The patch antenna's phase centre in robot coordinates.

    Same contract as lidar_pose(): the sim exporters and the ROS 2 generator put the GPS
    frame here rather than each guessing where the antenna ended up.  A patch radiates
    about its own normal, which is +Z - the platform is deliberately level, not raked -
    so the frame needs no axis, only a point."""
    return (GPS_X, 0.0, GPS_SEAT_Z + GPS_BOARD[2] + GPS_PHASE)

def gps_com():
    """Centroid of the receiver + patch stack sitting on the platform."""
    return (GPS_X, 0.0, GPS_SEAT_Z + GPS_STACK[2]/2.0)

def gps_mount():
    """Trestle over the Orange Pi carrying the NEO-6M and its active patch.

    Two feet on the deck's REAR pair of boss screws, so this part drills nothing: those
    two M3 x 12 just become M3 x 24 and the deck keeps its eight fixings.  From each foot an arm rises straight up past the Orange
    Pi and then makes one 45 deg run inboard to the platform; why it turns where it turns,
    and why the platform's width and its height are the same number, is argued in the
    parameter block.

    The receiver is strapped down by two ties at +-GPS_TIE_X - outside the 25 mm patch, so
    the tie bears on bare PCB and not on ceramic - and it is located by them: the four slots
    sit hard against the board's long edges, so each tie's rising leg is a post at the edge
    and the board cannot walk sideways out from under them.  Nothing on the platform stands
    proud of it, and that is a PRINTING constraint, not a styling one - see below.  Nothing
    holds the board's ends either, on purpose: the 4-pin header leaves one and the u.FL lead
    the other, and which is which depends on the board variant, of which there are several
    wearing the same silkscreen.  The patch sits on the board on its own tape, as the module
    ships.

    IT PRINTS UPSIDE DOWN, on the platform's top face.  That face is the part's one big
    flat, and the right way up the platform is a 40 x 52 ceiling 33 mm above the bed: the
    whole thing would print on two little discs and then want support under a table top.
    Inverted there is nothing to support - the arms grow out of the platform at 45 deg, the
    vertical run is vertical, the pads end up on top and their @9.6 tops are the only thing
    left facing the bed.  tools/orient_scan.py: 2018 mm2 of bed against 126 the right way
    up, 339 mm2 of overhang against 2238.  That is what the capture rails this part used to
    have cost - 1.2 mm of rail turned the bed face into two 40 x 1.6 strips."""
    z0 = BODY_Z1 + DECK_T
    ky, kz = GPS_KNEE
    px, py, pt = GPS_PLATE
    pz = GPS_SEAT_Z - pt                              # platform underside
    s = None
    for sy in (-1.0, 1.0):
        a = cyl(GPS_PAD_R, GPS_PAD_H, (GPS_X, sy*GPS_Y, z0))
        a = a.union(rod((GPS_X, sy*GPS_Y, z0+GPS_PAD_H), (GPS_X, sy*ky, kz), GPS_ROD))
        a = a.union(rod((GPS_X, sy*ky, kz), (GPS_X, sy*GPS_LAND, pz+1.0), GPS_ROD))
        a = a.cut(cyl(M3_CLR, GPS_PAD_H+2.0, (GPS_X, sy*GPS_Y, z0-1.0)))
        s = a if s is None else s.union(a)
    s = s.union(bxc(GPS_X-px/2, GPS_X+px/2, -py/2, py/2, pz, GPS_SEAT_Z))
    for sx in (-1.0, 1.0):                            # tie slots, hard against the board's
        for sy in (-1.0, 1.0):                        # long edges - they ARE the location
            y0 = sy*(GPS_BOARD[1]/2 + CLR)
            s = s.cut(bxc(GPS_X+sx*GPS_TIE_X-GPS_TIE[0]/2, GPS_X+sx*GPS_TIE_X+GPS_TIE[0]/2,
                          y0, y0+sy*GPS_TIE[1], pz-1.0, GPS_SEAT_Z+1.0))
    # cable notch in the rear edge: the harness turns down here for the Pi's UART, instead
    # of running over the platform's corner and being chafed by it.
    s = s.cut(bxc(GPS_X-px/2-1.0, GPS_X-px/2+4.0, -5.0, 5.0, pz-1.0, GPS_SEAT_Z+1.0))
    return s

def camera_frame():
    """(origin, axis, up) of the module in robot coordinates.

    `origin` is the PCB's FRONT face on the optical axis - the one face the drawing
    dimensions everything from - `axis` is where the lens looks and `up` is the image's
    own +v.  Every other camera function is written in these three vectors so that moving
    the module is CAM_X / CAM_Z / CAM_TILT and nothing else."""
    t = math.radians(CAM_TILT)
    return ((CAM_X, 0.0, CAM_Z), (math.cos(t), 0.0, math.sin(t)),
            (-math.sin(t), 0.0, math.cos(t)))

def cam_loc():
    o, n, u = camera_frame()
    return cq.Location(cq.Plane(origin=o, xDir=n, normal=u))

def cam_box(d0, d1, y0, y1, w0, w1):
    """A box in BOARD coordinates: `d` along the optical axis (0 = the PCB's front face),
    `w` across the board (0 = the axis, + = image up), `y` straight in robot y - the board
    is level in y whatever the tilt, so there is nothing to transform there."""
    return W(bxc(d0, d1, y0, y1, w0, w1).val().moved(cam_loc()))

def cam_span():
    """(y0, y1) of the board: the connector end and the short end, in robot coordinates."""
    a = CAM_TAIL*CAM_LENS_U
    b = -CAM_TAIL*(CAM_BOARD[0]-CAM_LENS_U)
    return (min(a, b), max(a, b))

def camera_pose():
    """Entrance pupil and optical axis, robot coordinates.

    Same contract as lidar_pose() and gps_pose(): every consumer reads the extrinsic from
    here instead of keeping its own copy.  CAM_OPT is where the pupil sits up the axis
    from the PCB face and it is marked **verify** - it is the one number on this module
    the drawing does not give."""
    o, n, _ = camera_frame()
    return tuple(c + d*CAM_OPT for c, d in zip(o, n)), n

def camera_fov():
    """(horizontal, vertical) FOV in degrees.

    The catalogue states one number and it is the DIAGONAL; H and V are what a pipeline
    actually needs, and they follow from the sensor's aspect.  Derived here so there is
    one figure to correct if CAM_FOV_D turns out to be the horizontal instead."""
    w, h = CAM_PIX
    d = math.hypot(w, h)
    k = math.tan(math.radians(CAM_FOV_D)/2.0)
    return (2*math.degrees(math.atan(k*w/d)), 2*math.degrees(math.atan(k*h/d)))

def camera_com():
    """Centroid of the module as bought - board, holder and lens, on the optical axis."""
    o, n, _ = camera_frame()
    return tuple(c + d*(CAM_LENS_H/4.0 - CAM_BOARD[2]/2.0) for c, d in zip(o, n))

def camera_module():
    """The module itself, not a printed part.

    Here for the reason servo_dummy() is: interference(), the ROM scan and the assembly
    all have to see the thing that is actually bolted on, and none of them can see a
    number in a table."""
    L, Wd, T = CAM_BOARD
    o, n, _ = camera_frame()
    y0, y1 = cam_span()
    s = cam_box(-T, 0.0, y0, y1, -Wd/2, Wd/2)
    s = s.union(cyl(CAM_LENS_D/2, CAM_LENS_H, o, axis=n))
    cy = CAM_TAIL*CAM_LENS_U                              # the connector end
    s = s.union(cam_box(-T-CAM_CONN[1], -T, min(cy, cy-CAM_TAIL*CAM_CONN[0]),
                        max(cy, cy-CAM_TAIL*CAM_CONN[0]), -Wd/2, Wd/2))
    return s

def camera_mount():
    """The channel that holds it, standing on the hip-roll gusset's own top face.

    Why it is shaped like this rather than like a bracket is argued in the parameter
    block; what the geometry does is this.  The shelf spans the ledge and carries the
    board's lower edge in a slot.  The skirt behind the board steps back CAM_BACK ->
    CAM_BACK_HI where it passes the deck's top, because above that the only thing keeping
    it out of the LiDAR pedestal is LIDAR_BASE_FLAT.  The wall in front exists only over
    the board's UPPER half - lower down there is 0.5 mm to the fork arm and no wall fits -
    and it is cut away over the lens, which is what stops the mount vignetting its own
    camera.  The board goes in from the +y end and the same two screws that hold the mount
    down close that end.

    IT PRINTS ON ITS BACK, on the skirt: that face is the part's one big flat, and stood
    up the right way the whole 90 mm channel is a 20 mm wall on a 4 mm foot."""
    L, Wd, T = CAM_BOARD
    o, n, _ = camera_frame()
    y0, y1 = cam_span()
    e0, e1 = min(CAM_END), max(CAM_END)
    slot = T + 2*CLR
    top  = CAM_Z + Wd/2*math.cos(math.radians(CAM_TILT))          # the board's upper edge
    bot  = CAM_Z - Wd/2*math.cos(math.radians(CAM_TILT))          # ... and its lower one
    deck = BODY_Z1 + DECK_T
    s = bxc(CAM_BACK, CAM_FRONT, e0, e1, CAM_LEDGE, bot)          # shelf
    s = s.union(bxc(CAM_BACK, CAM_FRONT, e0, e1, bot, deck))      # ... and the skirt, in
    s = s.union(bxc(CAM_BACK_HI, CAM_FRONT, e0, e1, deck, top))   # two steps past the deck
    # everything in front of the board below its upper half has to go: 0.5 mm to the fork
    s = s.cut(cam_box(-CLR, 40.0, e0-1, e1+1, -Wd, 1.0))
    # ... and so does everything in front of the lens, all the way across the holder
    s = s.cut(cam_box(-CLR, 40.0, -CAM_LENS_D/2-2.0, CAM_LENS_D/2+2.0, -Wd, Wd))
    # the board's own pocket, open at both ends so it slides in
    s = s.cut(cam_box(-T-CLR, CLR, e0-1, e1+1, -Wd/2-CLR, Wd/2+CLR))
    s = s.cut(cam_box(-T-CLR-CAM_CONN[1], -T-CLR, e0-1, y0+CAM_CONN[0]+CLR,
                      -Wd/2-CLR, Wd/2+CLR))                       # ... and its connector
    # two M3 down into nuts in the gusset, both past the board's short end
    for fy in CAM_FOOT_Y:
        s = s.cut(cyl(M3_CLR, 40.0, (CAM_FOOT_X, fy, CAM_LEDGE-1.0)))
        s = s.cut(cyl(3.2, 6.0, (CAM_FOOT_X, fy, top-6.0)))       # head counterbore
    ky, kl, kd = CAM_KEY                                          # locating tongue
    s = s.union(bxc(CAM_FOOT_X-3.0, CAM_FOOT_X+3.0, ky-kl/2, ky+kl/2, CAM_LEDGE-kd,
                    CAM_LEDGE))
    return s

def camera_clear(wp):
    """Highest elevation, in degrees off the optical axis, at which a solid still shows up
    inside the frame - and -180 if it is out of frame altogether.

    The camera's analogue of lidar_fov_clear(), and it answers the only question that
    matters for the job: a 1.6 m face at 2.2 .. 4 m sits at +19 .. +32 deg from this lens,
    so anything of the robot's own bodywork that reaches into that band is in the way of
    the thing the camera is for.  Below it the robot may appear - it is a quadruped, it
    sees its own legs - and the number says how far below."""
    (ox, oy, oz), n = camera_pose()
    _, _, u = camera_frame()
    r = (n[1]*u[2]-n[2]*u[1], n[2]*u[0]-n[0]*u[2], n[0]*u[1]-n[1]*u[0])   # image +h
    hf, vf = (math.radians(a/2.0) for a in camera_fov())
    worst = -180.0
    for v in wp.val().tessellate(0.4)[0]:
        d = (v.x-ox, v.y-oy, v.z-oz)
        f = sum(q*m for q, m in zip(d, n))
        if f <= 1e-6:
            continue
        a = math.atan2(sum(q*m for q, m in zip(d, r)), f)
        e = math.atan2(sum(q*m for q, m in zip(d, u)), f)
        if abs(a) <= hf and abs(e) <= vf:
            worst = max(worst, math.degrees(e))
    return worst

# =====================================================================================
# PART: hip_bracket / thigh / shin / foot
# =====================================================================================
def hip_bracket():
    x0, x1 = PITCH_X-SPINE_W/2, PITCH_X+SPINE_W/2
    ys, ye = ROLL_Y+SPINE_R0-1.0, LEG_Y+SLEEVE_LEN/2
    # shelf: roll spine -> pitch sleeve.  CLOSED box section - as an open 3 mm U-channel
    # this was the softest path in the leg and its root never converged in FEA.
    b = (bxc(x0, x1, ys, ye, ROLL_Z-17.5, ROLL_Z-9.5)
         .cut(bxc(x0+3, x1-3, ys+6.0, ye-3, ROLL_Z-14.5, ROLL_Z-12.5)))
    b = b.union(bxc(x0, x1, ys, LEG_Y-S_W/2-SLEEVE_W,                  # web down the inboard face
                    ROLL_Z-17.5, ROLL_Z+8.0))
    # ramp away the step where the shelf hangs below the fork spine (ROLL_Z-14): a sharp
    # re-entrant corner there has no converged stress, only a mesh-dependent one.
    #
    # The ramp shapes the SHELF, and only the shelf.  Applied to the whole part it also
    # notched the pitch sleeve's two outer flats - and, extruded to the shelf's own width
    # (x0-1, 30 long), it stopped 0.36 mm short of those flats and left a fin exactly that
    # thin hanging in the void it had just cut: under one extrusion width, and the site of
    # this part's peak von Mises (12.4 MPa at x = 74.91 against a p99 of 2.8).  So the tool
    # is swept clear past the shelf on both sides - no sliver can survive - and the sleeve
    # is unioned in AFTER it, which fills the flats back.  Do not instead cut the sleeve out
    # of the tool: that leaves the tool's faces exactly on the sleeve's, and gmsh will not
    # mesh the result ("PLC Error: a segment and a facet intersect").
    b = b.cut(cq.Workplane("YZ")
              .polyline([(ys-0.1, ROLL_Z-13.9), (ys+9.0, ROLL_Z-13.9), (ys-0.1, ROLL_Z-18.5)])
              .close().extrude(x1-x0+8).translate((x0-4, 0, 0)))
    s = mv(fork(), ROLL_LOC).union(mv(sleeve(), PITCH_LOC)).union(b)
    return s.cut(env_all("roll"))          # its fork bolts to the roll hubs

def thigh():
    s = mv(fork(), PITCH_LOC).union(mv(sleeve(), KNEE_LOC))
    x0, x1 = PITCH_X-S_W/2-SLEEVE_W, PITCH_X+S_W/2+SLEEVE_W
    y0, y1 = LEG_Y-SLEEVE_LEN/2, LEG_Y+SLEEVE_LEN/2
    s = s.union(bxc(x0, x1, y0, y1, KNEE_Z+S_L-S_AX+SLEEVE_W, PITCH_Z-SPINE_R0-3.0)  # box beam
                .cut(bxc(x0+3, x1-3, y0+3, y1-3, KNEE_Z+S_L-S_AX, PITCH_Z-SPINE_R0)))
    # Cable-tie slots, left as sharp rectangles on purpose - the obvious "round the corners
    # for strength" move was tried here and measured, and it buys nothing.  thigh_A stall at
    # 1.2 mm over four variants: sharp 23.19 MPa, obround 23.15, obround stopping short of
    # the pocket wall 22.95, no slot at all 22.98.  The slot is worth 0.2 MPa of 23.  At
    # 2.0 mm the same four spread 15.7 .. 18.5, which is one mesh's luck on a corner that
    # has not converged and not a difference between the parts - do not read that spread as
    # a result in either direction.  What the peak IS: the knee sleeve's cable window leaves
    # this beam's two +-y walls unsupported over the 15 mm it spans, and both hot cells sit
    # 0.3 mm inside its edges (23.1 MPa against a p99 of 9.0, at 82.81 / 97.17 in x against
    # window edges at 82.5 / 97.5).  Bounding that window in local z - it runs the sleeve's
    # whole length today and the connector only needs +-7 - is the change that would move
    # this number.  It touches all three sleeves, so it is its own job.
    for z in (PITCH_Z-34.0, PITCH_Z-44.0):
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
    s = s.union(cyl(SPIGOT_R, SPIGOT_H, (PITCH_X, LEG_Y, zf+SPIGOT_Z0)))   # foot spigot
    # Foot bolt.  This one used to be an M3 self-tapped straight up the spigot - the worst
    # thread-into-plastic on the robot: impact-loaded, and in pull-out on every step.  Now
    # it is a clearance hole into a nut sitting in a slot in the ankle boss, FOOT_NUT_Z up,
    # which is above the TPU foot's top face - so the nut is reachable with the foot on and
    # the foot stays a press-and-bolt part.  Slot opens +y, outboard on the A legs.
    s = s.cut(cyl(M3_CLR, FOOT_NUT_Z+M3_NUT_H+5.0, (PITCH_X, LEG_Y, zf)))
    s = s.cut(nut_slot((PITCH_X, LEG_Y, zf+FOOT_NUT_Z), (0.0, 1.0, 0.0), run=SPIGOT_R+6.0))
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
    s = s.cut(cyl(SPIGOT_R+FOOT_FIT, SPIGOT_H, (PITCH_X, LEG_Y, zf+SPIGOT_Z0)))
    # The foot bolt, M3 x FOOT_BOLT_L, driven up from the sole into the shin's nut slot.
    # Both cuts start BELOW the sole (-1) so they break the dome's surface cleanly instead
    # of leaving a skin over the entry - the bug this replaces was exactly a hole that
    # started inside the solid.  The clearance hole stops at the 2 mm pad the spigot lands
    # on; above that it is the shin's own bore that carries the shank.
    sole = zf - FOOT_D/2
    s = s.cut(cyl(M3_CLR, (zf+SPIGOT_Z0) - (sole-1.0), (PITCH_X, LEG_Y, sole-1.0)))
    # Head pocket.  Its ceiling at zf-FOOT_CB_Z is the only downward-facing face in the
    # part, and it is what the head pulls against when the nut above takes up - the old
    # pocket's one annulus faced UP, so even a bolt that could reach would have pulled
    # straight through.  Bearing on TPU is soft by nature: this is a retention bolt, snug,
    # not a preloaded joint.
    s = s.cut(cyl(FOOT_CB_R, (zf-FOOT_CB_Z) - (sole-1.0), (PITCH_X, LEG_Y, sole-1.0)))
    return s

def servo_gauge():
    g = sleeve(length=18.0, window=False)
    # half sleeve: quick print.  The cut starts behind the thrust lug and takes it with it
    # - the gauge is here to check the bore against a real ST3215, and half a lug with half
    # a nut channel in it would only be something to misread.
    g = g.cut(bxc(-S_AX-SLEEVE_W-THRUST_L-1, S_L-S_AX+4, 0.0, S_W, -20, 20))
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

BODY_PARTS = ("chassis_bottom", "chassis_top", "lidar_mount",
              "gps_mount", "camera_mount")
INTERF_TOL = 1.0        # mm3 - under this it is two faces meeting, not two solids sharing

def interference(names=BODY_PARTS):
    """Pairwise boolean overlap between the parts that are all bolted into one rigid body,
    in robot coordinates.  Nothing else checks this: rom_scan covers leg-vs-body because
    those move, and isValid() is perfectly happy with two parts occupying the same 95 mm3
    - which is exactly what chassis_top's Orange Pi standoffs and the LiDAR base disc did.
    Cheap on purpose: three solids, three intersections, no sweep."""
    bad = []
    for i, a in enumerate(names):
        for b in names[i+1:]:
            try:
                v = PARTS[a][0].val().intersect(PARTS[b][0].val()).Volume()
            except Exception:
                v = 0.0
            if v > INTERF_TOL:
                bad.append((a, b, v))
    return bad

def gps_clear():
    """mm3 of gps_mount inside the Orange Pi's envelope - the mast's own binding
    constraint, and the one thing about it isValid() and interference() cannot see: the
    Pi is a payload, not a part, so it exists in this model only as OPI_BOX."""
    cx, _, cz = opi_com()
    L, W, H = OPI_BOX
    box = bxc(cx-L/2, cx+L/2, -W/2, W/2, cz-H/2, cz+H/2)
    try:
        return PARTS["gps_mount"][0].val().intersect(box.val()).Volume()
    except Exception:
        return 0.0

def foot_bolt_check():
    """The foot bolt's path, checked against the real solid instead of against the numbers
    that were supposed to produce it.  Returns (blocked, reach, spare), all mm.

    blocked  how much of the on-axis run from the sole up to the pad the spigot lands on
             is still solid TPU.  It has to be 0.  A bolt hole that starts INSIDE the dome
             is what isValid() cannot see and what interference() does not cover - the
             foot shipped that way, with 7 mm of material under the entry and no way in.
    reach    how far the tip passes the nut's far face, from the head's bearing shoulder.
    spare    how much of the shin's own clearance hole is left beyond the tip.
    Both of the last two have to stay positive, and they are what fixes FOOT_BOLT_L."""
    from OCP.BRepClass3d import BRepClass3d_SolidClassifier
    from OCP.gp import gp_Pnt
    from OCP.TopAbs import TopAbs_IN
    sol = PARTS["foot"][0].val().wrapped
    z0, z1 = FOOT_Z - FOOT_D/2, FOOT_Z + SPIGOT_Z0
    n = max(2, int((z1 - z0) / 0.25)); dz = (z1 - z0) / n
    blocked = sum(dz for i in range(n)
                  if BRepClass3d_SolidClassifier(
                      sol, gp_Pnt(PITCH_X, LEG_Y, z0 + (i+0.5)*dz), 1e-7).State() == TopAbs_IN)
    tip = FOOT_BOLT_L - FOOT_CB_Z                     # above FOOT_Z, from the shoulder
    return blocked, tip - (FOOT_NUT_Z + M3_NUT_H), (FOOT_NUT_Z + M3_NUT_H + 5.0) - tip

PARTS, REPORT = {}, {}
def build():
    hb, th, sh, ft = hip_bracket(), thigh(), shin(), foot()
    PARTS["chassis_bottom"] = (chassis_bottom(), 1, "PETG/ASA, 4 walls, 30% gyroid")
    PARTS["chassis_top"]    = (chassis_top(),    1, "PETG/ASA, 4 walls, 25%")
    PARTS["lidar_mount"]    = (lidar_mount(),    1, "PETG/ASA, 4 walls, 30%")
    PARTS["gps_mount"]      = (gps_mount(),      1, "PETG/ASA, 4 walls, 30% - platform down")
    PARTS["camera_mount"]   = (camera_mount(),   1, "PETG/ASA, 4 walls, 40% - skirt down")
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
    a.add(PARTS["gps_mount"][0],      name="gps_mount",      color=grey)
    a.add(PARTS["camera_mount"][0],   name="camera_mount",   color=grey)
    a.add(camera_module(),            name="camera",         color=dark)
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
            a.add(f(posed(srv[i], kind)), name=f"servo_{tag}_{JOINTS[i][0]}", color=dark)
            a.add(f(posed(hub[i], kind)), name=f"hub_{tag}_{JOINTS[i][0]}",
                  color=cq.Color(0.66,0.70,0.76))          # stock aluminium hubs
    return a

# Build direction, not taste: measured with tools/orient_scan.py (bed contact and
# unsupported area per direction) and confirmed by slicing all six axis-aligned directions
# in Orca.  For hip_bracket / thigh / shin this is also the strongest direction - see
# `fea.py --all --orient`, which scores the traction on the layer plane - so it is not
# free to flip them for less support.
PRINT_ORIENT = {"chassis_bottom": ((1,0,0),0), "chassis_top": ((1,0,0),0),
                "lidar_mount": ((1,0,0),180),
                "hip_bracket_A": ((0,1,0),90),
                "hip_bracket_B": ((0,1,0),90), "thigh_A": ((1,0,0),90),
                "thigh_B": ((1,0,0),90), "shin_A": ((0,1,0),90), "shin_B": ((0,1,0),90),
                "foot": ((1,0,0),180), "servo_gauge": ((1,0,0),0),
                # upside down, on the platform's top face: it is the only flat on
                # the part (2080 mm2 against 72 on the two pads) and it turns a 40 x 52
                # unsupported ceiling into the bed itself.  See gps_mount's docstring.
                "gps_mount": ((1,0,0),180),
                # on its back skirt: the one big flat.  Stood up the right way this is a
                # 90 x 20 mm wall on a 4 mm foot.  90 - CAM_TILT lays that face on the bed.
                "camera_mount": ((0,1,0), 90.0+CAM_TILT)}

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
    bad = interference()
    # The camera module is not a printed part and so is not in PARTS, but it is bolted to
    # the same rigid body and it is the thing with 1 mm of clearance on four sides - it
    # has to be in this check, not just its channel.
    # The camera module and the IMU board are not printed parts and so are not in PARTS,
    # but both are bolted to the same rigid body and both are the thing with ~1 mm of
    # clearance rather than the bracket that holds it - they have to be in this check.
    for pname, psolid in (("camera", camera_module().val()), ("imu", imu_module().val())):
        for nm in BODY_PARTS:
            try:    v = psolid.intersect(PARTS[nm][0].val()).Volume()
            except Exception: v = 0.0
            if v > INTERF_TOL:
                bad.append((pname, nm, v))
    for na, nb, v in bad:
        print(f"  !! INTERFERENCE  {na} x {nb}  {v:.1f} mm3")
    if not bad:
        print(f"  body clear: {' / '.join(BODY_PARTS)} + camera + imu share no solid")
    # ... and the IMU against the battery pack, which interference() cannot see at all.
    iv, igap = imu_clear()
    if iv > INTERF_TOL:
        print(f"  !! INTERFERENCE  imu x battery pack  {iv:.1f} mm3")
    else:
        print(f"  imu clear: {igap:+.2f} mm of air between the board and the pack")
    # The LiDAR's own field of view is a geometric invariant like the interference check:
    # the L2 sees nothing below its base plane, so any static bodywork above that plane is
    # a permanent blind wedge in the direction that matters.  isValid() cannot see this and
    # neither can rom_scan.
    need = lidar_seat_min()
    if LIDAR_SEAT_Z < need:
        print(f"  !! LIDAR FOV  seat at {LIDAR_SEAT_Z:.1f} is below the {need:.1f} the body"
              f" needs at {LIDAR_TILT:.0f} deg tilt - the deck is in the cone")
    else:
        print(f"  lidar clear: seat {LIDAR_SEAT_Z:.1f} vs {need:.1f} needed at"
              f" {LIDAR_TILT:.0f} deg tilt ({LIDAR_SEAT_Z-need:+.1f} mm margin)")
    # ... and the same invariant per part, against the real 96 deg cone.  camera_mount is
    # the one that can go wrong quietly: it is the closest thing to the rim now.
    blocked, reach, spare = foot_bolt_check()
    if blocked > 0.05 or reach < 0 or spare < 0:
        print(f"  !! FOOT BOLT  {blocked:.1f} mm of the sole-to-pad run is solid,"
              f" tip {reach:+.1f} mm past the nut, {spare:+.1f} mm of shin bore left")
    else:
        print(f"  foot bolt:   M3 x {FOOT_BOLT_L:.0f} clears the sole, {reach:+.1f} mm past"
              f" the nut, {spare:+.1f} mm of shin bore to spare")
    v = gps_clear()
    if v > INTERF_TOL:
        print(f"  !! GPS MAST  {v:.1f} mm3 of gps_mount is inside the Orange Pi envelope")
    else:
        print(f"  gps clear:   mast over the {OPI_BOX[0]:.0f}x{OPI_BOX[1]:.0f}x{OPI_BOX[2]:.0f}"
              f" Orange Pi envelope, seat {GPS_SEAT_Z:.0f}")
    for nm in ("chassis_top", "gps_mount", "camera_mount"):
        mg = lidar_fov_clear(PARTS[nm][0])
        if mg < 0:
            print(f"  !! LIDAR FOV  {nm} is {-mg:.1f} deg INSIDE the cone")
        else:
            print(f"  lidar fov:   {nm:14s} {mg:+.1f} deg outside the 96 deg cone")
    # ... and the camera's own view.  A 1.6 m face at 2.2 .. 4 m is at +19 .. +32 deg
    # from this lens, so anything of the robot reaching into that band is in the way of
    # the one job the camera has.  Below it the dog may see itself - it is a quadruped.
    hf, vf = camera_fov()
    print(f"  camera:      {hf:.0f} x {vf:.0f} deg at "
          f"{CAM_PIX[0]}x{CAM_PIX[1]}, {CAM_TILT:+.0f} deg nose-up")
    for nm in ("chassis_bottom", "chassis_top", "camera_mount"):
        e = camera_clear(PARTS[nm][0])
        if e < -180.0 + 1e-6:
            print(f"  camera view: {nm:14s} out of frame")
        elif e > 19.0:
            print(f"  !! CAMERA VIEW  {nm} reaches {e:+.1f} deg - into the face band")
        else:
            print(f"  camera view: {nm:14s} up to {e:+.1f} deg, below the face band")
    a = assembly(hb, th, sh, ft)
    a.save(os.path.join(OUT, "mini_dog_assembly.step"))
    tm = sum(r["est_mass_g"]*r["qty"] for r in rows)
    carried = (N_SERVO*SERVO_KG + BATTERY_KG + ELECTRONICS_KG + LIDAR_KG + GPS_KG
               + CAMERA_KG + IMU_KG)*1000.0
    print(f"\n  printed mass  ~{tm:.0f} g   + {N_SERVO} servos {N_SERVO*SERVO_KG*1000:.0f} g"
          f" + 3S2P pack ~{BATTERY_KG*1000:.0f} g"
          f" + Orange Pi/BMS/wiring ~{ELECTRONICS_KG*1000:.0f} g"
          f" + LiDAR ~{LIDAR_KG*1000:.0f} g + GPS ~{GPS_KG*1000:.0f} g"
          f" + camera ~{CAMERA_KG*1000:.0f} g + IMU ~{IMU_KG*1000:.0f} g"
          f"  ->  ~{(tm+carried)/1000:.2f} kg")
    print("  ROM scan (coarse, 10 deg steps, real solids):")
    rom = {}
    # gps_mount goes in MIRRORED: it stands over the rear pair of deck bosses and this
    # scan swings the FRONT-left leg, so mirroring it forward is exactly the rear-leg
    # scan against the real one - the legs are mirror images and the roll axis is x.
    rom["hip_roll"] = rom_scan(hip_bracket(),
                               chassis_bottom().union(mirX(gps_mount()))
                                               .union(camera_mount())
                                               .union(camera_module())
                                               .union(mv(servo_dummy(), ROLL_LOC)),
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
