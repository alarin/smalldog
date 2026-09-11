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
HUB_BC        = 14.00                 # 4x HUB_BOLT_D clearance holes, at 0/90/180/270
HUB_BOLT_D    = 3.00                  # M3, and this one is measured on a real hub rather
                                      # than read off the vendor STEP, which says @2.5 and
                                      # is wrong.  The BOLT CIRCLE is right - the printed
                                      # gauge dropped onto the real hub before this was
                                      # found - so only the hole size and the screw move.
HUB_N         = 4
HUB_TOP_Z     = +20.30                # driven-hub outer face  (case top  +17.50)
HUB_BOT_Z     = -16.95                # passive-hub outer face (case base -17.50, recessed)
HUB_REC_D     = 25.00                 # recess in the case base around the passive hub
# hub plates as they actually are in ref/ST3215-3D/ST3215.step (axis = STEP +Y at
# x=-25.5, z=0; STEP y + 10.70 = local z).  Both plates carry 4 holes on the @14 circle at
# 0/90/180/270.  The STEP draws them @2.5 and through; the hubs that ARRIVED are TAPPED M3
# (HUB_BOLT_D, and see fork()) - a vendor number a real part contradicts loses, so the
# screw threads INTO the aluminium here and this is the one place in a torque path with no
# nut behind it.  Driven plate: y 7.10..9.60 -> z 17.80..20.30, central @6.2 pocket stepping to
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
M25_CLR, M3_CLR = 1.45, 1.70          # clearance-hole RADII: @2.9 / @3.4
M25_NUT_AF, M25_NUT_H = 5.00, 2.00
M3_NUT_AF, M3_NUT_H = 5.60, 2.70
NUT_CLR    = 0.25                     # slide fit on a nut pocket's flats
# Thread-FORMING radii, for the one case a nut is not required: a cover or a bracket that
# carries no torque path (see README, "Screws into plastic").  The screw cuts its own
# thread, so the hole is between the thread's minor and pitch diameter - M3 is 2.459 and
# 2.675, M2.5 is 2.013 and 2.208 - and these sit just under the middle of that band
# because an FDM hole prints ~0.1 mm undersize on top of it.  Two rules travel with them:
# the boss wall is >= the screw diameter all round (@3 M3 hole -> >= 9 mm of boss), and
# the engaged length is 2 x D, because a formed thread in plastic strips at ~1 x D.
# Never use these on a joint, a fork, a hub or anything the leg loads.  UNVERIFIED until
# a printed coupon says otherwise - print one before the first part relies on it.
M25_TAP, M3_TAP = 1.05, 1.30          # thread-forming RADII: @2.1 / @2.6
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
# inboard web already comes to r = 22.0, so the lug corner is held at 21.6.
THRUST_Y    = 5.50                    # the two bolts, either side of the axis
THRUST_YL   = 10.00                   # lug half-width; the nut channels open on its faces
THRUST_Z    = 7.00                    # lug half-height
THRUST_L    = 6.00                    # how far the lug stands off the -x end wall
THRUST_SEAT = 3.00                    # lug behind each nut - this is what takes the preload
# The SCREW, and it is a set screw rather than a cap screw for a measured reason.  The lug
# clears the spine with 1.4 mm to spare; its screw does not, and the screw is the part that
# reaches furthest out.  Found on the first assembled leg, which bound on its own clamp,
# then reproduced in the arithmetic below: an M3x10 DIN 912 cap head, standing 0.65 proud
# with the case jacked forward, puts its corner at r = 24.2 against a spine that sweeps
# from 23.0 - so the joint fouls between about 2 and 38 deg of travel, a band that contains
# STAND_PITCH and that every stride crosses.  An ISO 7380 button head comes to 22.98, which
# is 0.02 mm and not a clearance.  Headless comes to 20.96 and clears by 2.0.
#
# Nothing is lost by removing the head.  The bolt is a JACK: the nut is captive in the lug,
# the tip presses the case, the reaction goes into THRUST_SEAT - the head bears on nothing
# at any point, and is only ever a driving feature.  The clearance hole is @3.4 and runs a
# millimetre past the lug's outer face, so a hex key still reaches the screw with the fork
# in place.  thrust_clear() is the check that was missing; it prints on every run.
# A screw you cannot reach is a screw that is not fitted.  The fork's two arms are bolted
# to the hubs LAST, with the servo already in its sleeve - the arms straddle the sleeve, so
# there is no order in which the fork goes on first - and each arm's four screws are driven
# from OUTSIDE it, along the joint axis.  Whether a driver fits there is a property of the
# neighbouring part, not of the fork, so fork_access() probes it against the real solid.
#
# At the pitch and knee that is a plain key with open air behind it, and SINCE 2026-09-09
# so is the roll joint - because the cradle is a bolted part now (see the CRADLE_* block).
# The fork goes on with the cradle IN HAND, before its four screws hold it to the tray, and
# at that moment all four hub screws are in open air: they sit on the @14 hub circle, i.e.
# y 29..43 and z +-7, and the flange is a FRAME whose opening is y +-43.11 by z +-9.36.
# Measured on the solid: a DRIVER_D key on all four axes is blocked by 0.7 mm3 against the
# cradle alone, and by 569 against the cradle bolted to the tray.  So the order is what
# makes it reachable, and the order is free - the cradle has to come off to change a
# servo anyway.
#
# WHAT THAT DELETED, because it is worth knowing why the geometry looks simpler than the
# history: the roll joint's inboard arm used to face the chassis with FORK_GAP of air and
# the fork could only go on after the servo was in its bore, so those four screws were
# always last and always blind.  Two answers were built for that.  The first was four @6
# bores coaxial with the screws, driven from inside the tray; it reached ONE screw of four
# - a deck boss stood in front of two, the outboard bore's axis ran 0.2 mm inside the
# tray's own side wall, and a coaxial bore only lets a key tilt 12 deg (git history, 2026-09-03).
# The second was two @6 channels leaning 20 deg outboard through the tray's front corner,
# with the leg turned a quarter turn by hand between screws so two channels served all
# four.  That one worked, and it cost the chassis a @6 hole through its front corner post
# and another through the lower half of each corner deck boss, out into the side vent.
# Splitting the cradle off gives the same access for no holes at all.
DRIVER_D      = 5.00                  # hex driver shank, and the room to turn it
KEY_D         = 2.00                  # the key: ISO 7380 M3 button heads take 2 mm, not
                                      # the 2.5 a DIN 912 cap head would
KEY_REACH     = 40.0                  # straight run a plain key wants behind a screw.
                                      # All six arms now, not five.
FORK_DRIVER_R = HUB_BC/2 + DRIVER_D/2 + CLR   # 9.85 - the circle a driver sweeps reaching
                                      # the four hub screws, and what the cradle's flange
                                      # is relieved to.  Derived from the hub circle on
                                      # purpose: the outermost screw sits at y = 42.9 and
                                      # the frame's rib starts at 43.11, so without this
                                      # the flange fouls the driver by 38 mm3 - which is
                                      # what fork_access() read the moment the two tilted
                                      # channels came out and stopped relieving it by
                                      # accident.
# The hub screws as bought: ISO 7380 button heads, M3 x 6, one length for all 96.  The
# head is the part that matters and it was never in the model: 1.65 tall on an arm whose
# outer face has FORK_GAP of air to the chassis gusset.  A cap head is 3.0 and would not
# fit either.  So the PASSIVE arm - the one that faces the gusset at the hip, and the same
# arm at every joint, because fork() is one part - carries a counterbore FORK_CB deep, and
# the two numbers are set against each other: deeper hides more head but leaves less arm
# for the screw to pass before it runs out of thread into the servo case.  With FORK_CB =
# 1.2 the head stands 0.45 proud in 1.1 of air, and the M3 x 6 reaches 2.25 of the hub's
# 2.2 with 0.55 - the case's own base recess - still to spare.  fork_screws() is the
# hardware, in every rom_scan on the moving side; head_clear() prints the margin.
HUB_SCREW_L   = 6.00
HUB_HEAD_D    = 5.70                  # ISO 7380 M3: dk max 5.7
HUB_HEAD_H    = 1.65                  # ... k max 1.65
FORK_CB       = 1.20                  # counterbore in the passive arm's outer face
FORK_GAP      = 1.10                  # air between the hip fork's inboard arm and the
                                      # chassis gusset; was 0.6 before the heads existed
THRUST_BOLT_L = 10.00                 # M3 x 10: 9.35 of it is lug + nut + reach to the case
THRUST_HEAD_D = 3.00                  # set screw, so the head IS the thread OD ...
THRUST_HEAD_H = 0.00                  # ... and so it stands nothing proud of the shank

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
# THE BATTERY IS A MODULE, NOT SIX CELLS IN THE CHASSIS.
#
# It used to be a cradle: four printed fins on the cell pitch, two end stops, a strap, and
# six loose 21700 sitting in the tray with their welded nickel a millimetre under the
# deck.  That is the arrangement every check in this repo is blind to - the cells are a
# payload, so interference() cannot see them, and nothing at all can see a tab that has
# chafed through its insulation against a printed edge.  The pack is now a self-contained
# unit: six cells welded into a 3 x 2 brick, HEATSHRUNK over the whole brick, its BMS in
# the same box, and a printed case with a screwed lid around both.  Two leads leave it and
# nothing else.  The robot carries one payload, not seven loose ones, and a pack that
# comes out for charging or storage is still enclosed when it is out.
#
# CELL_D and CELL_L are the datasheet MAXIMA of a Molicel INR21700-P42A over its wrap; a
# cell that measures bigger is a cell to re-measure, not a number to shave here.
CELL_D, CELL_L = 21.3, 70.2           # 21700, at its maximum over the wrap
BATT_TAB       = 4.0                  # welded nickel + insulation, at each cell end
BATT_WRAP      = 0.30                 # heatshrink, per side.  It goes over the finished
                                      # assembly - cells AND holder - so the holder is
                                      # INSIDE the wrap, not around it.
#
# THE CELLS NO LONGER TOUCH: TWO PRINTED END CAPS HOLD THEM ON A PITCH.
#
# They used to.  Welding six loose cells into a 3 x 2 brick with nothing holding them is
# the assembly step this design had no answer for - you are aligning six cylinders by hand
# while a spot welder is in the other one - so `cell_holder` is two combs, one at each end
# of the cells, that turn the six into one riggable object before any nickel goes on.
#
# WHAT IT COSTS IS THE PITCH AND NOTHING ELSE, and that is a geometric decision, not a
# happy accident.  The obvious holder is a frame round the outside of the array, and it
# does not fit: the module is already 0.60 mm under the deck and 0.5 mm off the deck
# screws' nut bosses, whose inner faces stand at |y| = 35.2 over the module's whole
# height.  So the caps are CLIPPED FLUSH with the outermost cells' own tangent planes in
# both y and z - there is no material outboard of a cell anywhere - and the module grows
# by exactly the two gaps the separator webs open up, one in z and two in y.  The cells
# are still captured: clipping a CH_WALL-thick annulus at the cell's tangent leaves a
# ~7.9 mm flat, and a 21.3 mm cell does not come out through 7.9 mm.  The heatshrink then
# lands on the CELLS at those flats and on the holder everywhere else, which is why
# BRICK_W/BRICK_H below add BATT_WRAP to a cell envelope and not to a holder envelope.
#
# CELL_GAP is the one number that costs height, so it is the knob to turn if batt_clear()
# ever has to come back: at 0.8 the deck gap is 0.60 mm, at 0.6 it is 0.80.  Do not read
# it as a print wall - the printed web between two bores is CELL_GAP - CH_FIT = 0.6 mm.
CELL_GAP       = 0.8                  # air between neighbouring cells, and so the pitch
CH_WALL        = 0.8                  # the comb's wall around a cell.  It lives BETWEEN
                                      # cells and inside the array's envelope; outboard
                                      # of the outer cells it is clipped away entirely.
CH_LEN         = 10.0                 # how far each cap grips down the cell from its
                                      # terminal.  The cap's outer face is FLUSH with the
                                      # terminal - not proud - so a nickel strip lies flat
                                      # across the cells instead of bridging plastic.
CH_FIT         = 0.2                  # bore over the cell, on diameter: a push fit
CELL_P         = CELL_D + CELL_GAP    # 22.1, the pitch, the same in y and in z
BATT_FIT       = 0.4                  # slip fit of the wrapped brick into the case, total
BATT_CASE_T    = 1.6                  # case floor and side walls - 4 perimeters at 0.4
BATT_LID_T     = 1.2                  # the lid: a cover, and the deck is above it
BATT_REB       = 1.0                  # rebate the lid drops into, all round
BATT_SEAT      = 0.8                  # register recess in the tray floor: what locates the
                                      # module in x and y now that there are no fins
BATT_FRONT_T   = 7.5                  # the FRONT wall, thickened, and the module's only
                                      # screwed fixing: the lid's front edge lands on it
                                      # and two M2.5 form their own thread straight down
                                      # into it.  2.7 mm of wall each side of an M25_TAP
                                      # hole, against the >= 1 x D the rule asks for.
                                      # It is a wall and not a pair of corner posts, and
                                      # the reason is length: posts wanted a 9 mm zone in
                                      # front of the brick, and the 9 mm came out of the
                                      # rear strip the ESP32/URT-1 bay lives in.
BATT_POST_L    = 5.0                  # ... engaged thread, 2 x D
BMS_L, BMS_W, BMS_H = 60.13, 37.2, 5.9        # MEASURED 2026-09-09 on the real board,
                                      # heatsink plate included in BMS_H.  Was 64/27/13
                                      # **verify**; the orientation is forced, since a
                                      # 60.13 mm edge cannot stand up in a BATT_IN_H =
                                      # 43.6 interior.  The board is SMALLER than the
                                      # guess in every direction that costs anything: it
                                      # is 3.9 shorter than the brick's width so the
                                      # max() below now picks the brick outright, and
                                      # 7.1 THINNER, which takes 7.1 mm straight off the
                                      # module's length in x.
BMS_GAP        = 2.0                  # air between the BMS and the brick's -x nickel, and
                                      # it has to swallow the retaining rib as well as the
                                      # gap: rib at BMS_X1+CLR, 0.8 wide, leaves 0.85 mm.
# The brick, wrapped.  This is what the case has to swallow and what the exporters hang
# BATTERY_KG on - see batt_com().
BRICK_L = CELL_L + 2*BATT_TAB         # 78.2, along x
BRICK_W = 2*CELL_P + CELL_D + 2*BATT_WRAP     # 66.1, three cells across y on the pitch
BRICK_H = 1*CELL_P + CELL_D + 2*BATT_WRAP     # 44.0, two layers on the pitch.  Both are
                                      # a CELL envelope plus the wrap, not a holder
                                      # envelope: see the clipping note above.
# The case interior.  Width is the BRICK or the BMS standing on edge, whichever is wider.
# Before the board was measured those two were 64.0 and 64.5, half a millimetre apart and
# the max() decided nothing; the real board is 60.13, so the brick wins by 4.4 mm and the
# case's width is the cells' alone.  Length is BMS + gap + brick + the front zone.
BATT_IN_W = max(BRICK_W + BATT_FIT, BMS_L + 1.0)
BATT_IN_H = BRICK_H + BATT_FIT
BATT_IN_L = BMS_H + BMS_GAP + BRICK_L
BATT_L = BATT_IN_L + BATT_CASE_T + BATT_FRONT_T    # 95.2 - the module's own outside
BATT_W = BATT_IN_W + 2*BATT_CASE_T                 # 68.2
BATT_H = BATT_CASE_T + BATT_IN_H + BATT_LID_T      # 46.4
BATT_X  = 5.0                         # module centre in x.  Not the body origin and not
                                      # the middle of the tray either: it was pushed
                                      # forward until the rear strip was the 8.9 mm the
                                      # ESP32 + URT-1 bay needs, with 4.3 mm at the front.
                                      # The measured BMS then took 7.1 mm off BATT_L and
                                      # the module shrank about this same centre, so both
                                      # ends now have 3.55 mm more than they need.  It is
                                      # deliberately NOT re-centred: holding x fixed walks
                                      # the heavy brick 3.55 mm AFT, which is the CoM
                                      # direction the module owed the robot in the first
                                      # place (the pack sits ahead of its own BMS).
BATT_Z0 = BODY_Z0 + 3.0 - BATT_SEAT   # the case's underside, sitting in its seat recess
BATT_Z1 = BATT_Z0 + BATT_H            # ... and its top: 24.4, with 0.60 mm to the deck
                                      # (batt_clear(); it was 1.4 before the cell holder
                                      # added its two separator gaps)
# The interior, and the three zones along it.  Derived once here because battery_case(),
# battery_lid(), the payload envelopes and both sim exporters all have to agree on them.
BATT_XI0 = BATT_X - BATT_L/2 + BATT_CASE_T    # -41.00, the rear wall's inner face
BATT_XI1 = BATT_X + BATT_L/2 - BATT_FRONT_T   # the front wall's - thicker than the rest
BATT_YI  = BATT_IN_W/2                        #  32.50
BATT_ZI0 = BATT_Z0 + BATT_CASE_T              # -21.20, the floor's top face
BATT_ZI1 = BATT_Z1 - BATT_LID_T               #  22.40, where the lid sits down
BMS_X0   = BATT_XI0                           # BMS on edge against the rear wall: BMS_H
BMS_X1   = BATT_XI0 + BMS_H                   # thick in x, BMS_L across y, BMS_W tall
BRICK_X0 = BMS_X1 + BMS_GAP                   # -33.10
BRICK_X1 = BRICK_X0 + BRICK_L                 # ... and so the brick is NOT centred on
                                              # the module: batt/brick_com() differ
BATT_SCREW_Y = 22.0                           # the lid's two screws, in the front wall
BATT_FRAME_W = 3.0                            # the lid's BMS retaining bar: width in x
BATT_FRAME_D = 2.0                            # ... and how far it hangs below the lid.
                                              # It exists ONLY over the BMS zone.  A lip
                                              # round the whole opening was drawn first
                                              # and is impossible: the wrapped brick fills
                                              # the interior to BATT_FIT/2 = 0.2 mm on
                                              # every side, so anything hanging 2 mm into
                                              # it crushes the pack - and the pack is a
                                              # payload, so no boolean in this file would
                                              # ever have said so.  module_clear() is the
                                              # check that does.
BMS_RIB_Y    = 22.0                           # the BMS's two rib pairs, on y
BMS_Z0       = BATT_ZI1 - BATT_FRAME_D - BMS_W    # ... and the ledge they sit on: the
                                              # board's top edge lands exactly on the
                                              # underside of the lid's retaining bar
BMS_SIDE_Y   = BMS_L/2 + CLR                  # and the y stops.  The guessed 64.0 board
                                              # was a press fit in a 64.9 interior and
                                              # needed none of this; the MEASURED 60.13
                                              # leaves 2.4 mm a side, which is a loose
                                              # PCB with wires on it inside a battery
                                              # case.  Two ribs off the rear wall take
                                              # that back, and they are the reason the
                                              # docstring's "retained in all six
                                              # directions" is still true.  They stop at
                                              # the board's top edge, not the lid: the
                                              # lid's bar is already there.
BATT_GROOVE = 1.2                     # a groove in the rear wall's inner face, at lid
                                      # height: the lid's rear edge slides INTO it and
                                      # only the front end is screwed.  A groove and not
                                      # an overhanging tongue - a tongue cannot work here,
                                      # because the lid is flush with the module's top and
                                      # so a tongue at the lid's own height has nowhere to
                                      # be.  There is nowhere for a rear post either: the
                                      # measured BMS is 60.13 across a 64.9 interior, so
                                      # each rear corner has 2.4 mm - not a post, and not
                                      # worth reworking the lid's retention for.
BATT_VENT_D = 4.0                     # two vents high in the rear wall.  A sealed case
                                      # around six cells is the wrong kind of safe - a
                                      # cell that vents has to have somewhere to go, and
                                      # it goes out the back, away from the electronics.
ESP_RIB_Y = 13.5                      # ... and its half length: clear of the driver run
                                      # to the lower pair of rear CRADLE_BOLT screws
ESP_X     = -51.0                     # the ESP32 + URT-1 divider rib, behind the
                                      # module and in front of the connector panel's pads
BATT_WIRE = (14.0, 8.0)               # grommet slot in the rear wall: pack leads and the
                                      # 3S balance lead, out to the rear connector panel
#
# WHAT THE MODULE COST, AND WHERE IT CAME FROM.  A case is walls, and the bay had none to
# give: from the tray floor at BODY_Z0+3 = -22 to the old pack ceiling at 21.4 there were
# 43.4 mm for 42.6 mm of cell, i.e. 0.8 mm for a floor, a lid and a fit.  What was above
# the pack was the IMU, in a 3.6 mm slot.  The IMU moved out - onto the deck's TOP face,
# inside the Orange Pi's standoff gap, see the IMU_* block - and the module now runs from
# -22.8 to 24.4 with 0.60 mm of air under the deck.  Nothing was thinned and no cell format
# changed; the slot was simply the wrong home for the board.
DECK_BOSS_R  = 5.8                    # tray boss: fat enough to swallow an M3 nut slot
DECK_NUT_DZ  = 6.0                    # ... its floor, below the boss top
# Deck screws as (x, |y|, nut-channel direction), once: chassis_bottom grows a boss under
# each and chassis_top drills each.
#
# ALL FOUR pairs sit at |y| = 41 now, and that is the battery module's doing.  The corner
# pair used to be at 38, which was already 2.7 mm inside the outer cell and only got away
# with it because the old cradle ended at x = +-42 and the bosses stand at +-52.  The
# module is 104.9 mm long - it reaches x = +-52 - so those bosses are no longer beside it
# in x, and the only way past them is to be outboard of it in y.  Measured on the solid:
# the module at |y| <= 34.1 against a DECK_BOSS_R boss whose inboard face is at 35.2, so
# 1.1 mm, and at |y| = 38 it was a 1.9 mm bite out of the module's corners.
#
# The channel direction is now explicit rather than derived from |y|, because |y| no
# longer distinguishes the two pairs.  It is given for the +y instance and mirrored in y
# for the -y one, the same way the boss itself is.  The mid pair opens along +x into the
# clear strip beside the module; the corner pair opens INBOARD in y, leaving its boss at
# y = 35.2 into the tray's own air.  Both are open air at the moment those nuts go in,
# which is with the deck off and BEFORE the module - and after the module the corner
# channels are closed by it, so a deck nut is not replaceable with the pack in.  The
# corner pair must not open along x at all: at x = +-52 a DECK_BOSS_R+6 channel runs out
# through the tray's own inner face at +-60.2.
# The mid pair still blocks the side cable channel at x = +-18; cables go around the
# module's ends or out through the side ports, not over it - there is 0.60 mm over it.
DECK_SCREWS  = ((-52.0, 41.0, (0.0, -1.0, 0.0)),
                (-18.0, 41.0, (1.0,  0.0, 0.0)),
                ( 18.0, 41.0, (1.0,  0.0, 0.0)),
                ( 52.0, 41.0, (0.0, -1.0, 0.0)))
# The deck's two stiffening lips, and the relief every deck screw needs through them.  An
# ISO 7380 head at |y| = 41 stands 0.8 mm proud of its counterbore and reaches into the
# lip, so chassis_top notches DECK_LIP_NOTCH out of it at all eight screws.
# AT THE REAR PAIR THAT NOTCH GOES FULL DEPTH and runs out to the deck's own rear end,
# and the reason is gps_mount: its two feet ARE that pair of screws, and a GPS_PAD_R pad
# centred at |y| = 41 reaches 45.8 into a lip whose inner face is at 43.  Shrinking the
# pad instead is not an option - clipped at 42.8 it would leave 0.05 mm of wall outboard
# of an M3 clearance hole, i.e. an open slot - so the lip is what gives way.  It costs
# 17 x 3 x 6 mm a side, at the very end of a 126 mm run, and it takes the useless 5 mm
# stub the old x +-6 notch left behind x = -58 with it.  Added 2026-09-11.
DECK_LIP_W, DECK_LIP_H = 3.0, 6.0     # the lip: width inboard of the deck edge, height
DECK_LIP_NOTCH = 2.0                  # ... head relief at a deck screw, down from the top
DECK_LIP_CUT   = 6.0                  # ... and its half length along x
# The hip-roll cradles are BOLTED parts, not part of the tray.
# ---------------------------------------------------------------------------------
# There are TWO of them - cradle_front and cradle_rear - and each carries both of that
# end's hip-roll servos.  They used to be four quarters unioned into chassis_bottom, and
# that made the tray the biggest print on the robot: 213 x 110 x 50 and 208 g, of which
# 110 g was servo cradle.  So every change to the tray - a deck screw, the battery seat,
# this connector panel - meant reprinting all four cradles and taking all four hip servos
# out of their sleeves, which is the whole reason this joint exists.
#
# ONE PART PER END, not four quarters, and that is forced rather than tidy: the two rails
# OVERLAP on the centreline (roll_module reaches y = -14 and its mirror reaches +14), so
# the four quarters are not four separable bodies.  Merging them is free and then some -
# one duct instead of two, one flange instead of two, and a bolt pattern that can span the
# full 98 mm instead of 63.
#
# THE JOINT IS FOUR SCREWS AND A REGISTER, and the register is not decoration.  The screws
# are nowhere near their limit: worst case, land3g and stall adding, is 22 N of axial and
# 28 N of shear on an M3, against ~3 MPa of bearing in a 4 mm flange.  What four fasteners
# do NOT replace is 795 mm2 of welded face for STIFFNESS, and stiffness is the whole point
# of the joint being here - the roll axis is 27 mm outboard of it and the foot 187 mm
# below, so a tenth of a millimetre of slop at the flange is 0.7 mm at the ground.  Each
# screw therefore runs through a spigot: the cradle's flange carries a CRADLE_REG_D boss
# standing CRADLE_REG proud of its face, into a pocket in a locally thickened tray wall.
# The four spigots take the shear and the torsion; the screws only clamp.
#
# WHERE THE SCREWS MAY GO IS NOT FREE, and what decides it is the fork, not the flange.
# The flange face is 98 x 30.7 mm but only CRADLE_T deep, because its outer face sits
# FORK_GAP behind the roll fork's rear arm - and an M3 nut does not fit in 4 mm with any
# wall left.  Past that face, over the flange's own z band, the ONLY thing in the way is
# that arm: a disc of ARM_R about each roll axis.  The spine's 23 < r < 34 annulus is
# swept only on the OUTBOARD side over a +-90 deg ROM, and at |z| <= CRADLE_Z1 it never
# gets inboard of the axis at all - which is also why the rails have always been allowed
# to run the full length at |y| <= 14.  So a nut boss may run out to CRADLE_BOSS_X (the
# servo case starts at 72.5) anywhere that clears ARM_R, and the four screws sit just
# inboard of the two arms with ~1.5 mm to spare.  Do not take that on trust: `rom_scan`
# is what checks the arm, and `cradle_clear()` is what checks a driver can reach the
# heads - a screw nobody can turn is the failure mode this repo has now shipped twice.
CRADLE_X      = BODY_L/2              # 63.0 - the joint plane, the tray's end wall face
CRADLE_T      = 4.0                   # flange depth: 63..67 = (ROLL_X-21.9) - FORK_GAP
CRADLE_Y1     = ROLL_Y + 13.11        # 49.11 - flange half width (the old root gusset's)
CRADLE_Z1     = S_W/2 + SLEEVE_W      # 15.36 - ... and half height.  CAM_LEDGE is this.
CRADLE_RIB    = 6.0                   # the flange is a FRAME: rib width round the opening.
                                      # The opening is what lets the connector panel out -
                                      # see the panel block below, which is laid out
                                      # against it and against the four nut bosses.
CRADLE_BOLT   = ((17.0, 10.5), (17.0, -10.5), (-17.0, 10.5), (-17.0, -10.5))
CRADLE_BOSS   = 10.0                  # nut boss, square in y and z.  y 12..22 keeps it
                                      # 1 mm off the camera's foot at y = 23 and 1 mm off
                                      # the bus window at y = 11; its nearest corner is
                                      # r = 15.0 from the roll axis against ARM_R = 13.5.
CRADLE_BOSS_X = 71.5                  # ... and how far out it runs (servo case at 72.5)
CRADLE_NUT_X  = 67.0                  # nut-slot floor: the full CRADLE_T under the nut,
                                      # which is what takes the clamp.  A pocket in the
                                      # 4 mm flange instead would leave 1.3 mm and creep.
CRADLE_NUT_RUN = 5.5                  # nut channel: opens toward the centreline, in air
                                      # between the rail box at |y| = 14 and the boss
CRADLE_REG_D  = 7.0                   # register spigot on the cradle's flange face ...
CRADLE_REG    = 2.0                   # ... standing this proud of CRADLE_X
CRADLE_SEAT   = 2.8                   # tray boss inboard of the wall: it is what makes
                                      # room for a 2.2 mm register pocket AND leaves a
                                      # flat head seat, in a wall that is only WALL thick
CRADLE_CB_D   = 6.5                   # the head is COUNTERBORED into the tray's seat, and
CRADLE_CB     = 2.0                   # that is not tidiness: the front seat's face is at
                                      # x = 57.4 and the battery module's front face is at
                                      # BATT_X + BATT_L/2 = 52.6, so the head has 4.8 mm
                                      # and the counterbore buys it 2.0 more.  It is kept
                                      # because cradle_head_clear() reads the real solid
                                      # and the pack has moved twice already.
                                      # Same class of defect
                                      # as the thrust clamp's cap head and the hub screws'
                                      # - the part in the way is HARDWARE, so isValid(),
                                      # interference() and rom_scan are all blind to it.
                                      # cradle_head_clear() is what prints the margin.
CRADLE_BOLT_L = 12.0                  # M3 x 12: counterbore floor 59.4 to the nut's far
                                      # face at 69.7 is 10.3, and 12 is the next length up
STRAP_Y       = 21.5                  # the strap's outboard edge where it passes the fork
                                      # arm.  A hard ROM limit, not a guess - see the
                                      # straps in roll_module().
STRAP_TAPER   = 8.0                   # ... and how far past the sleeve it takes to open
                                      # out to CRADLE_Y1, instead of stepping there
CRADLE_REACH  = 25.0                  # straight run a socket wants behind each head, and
                                      # what cradle_clear() probes against the tray
PANEL_REACH   = 20.0                  # ... and how far panel_clear() looks out from the
                                      # rear wall.  The plate that blocked every opening
                                      # started 1.2 mm behind it, so this is generous on
                                      # purpose: the question is "is anything there", not
                                      # "does the plug's own length fit".
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
# EVERY ONE OF THESE OPENED INTO SOLID PLASTIC UNTIL 2026-09-09, and nothing in this
# repository could see it.  The two rear hip-roll cradles met across the centreline and
# formed one continuous 2.8 mm plate at x = -64.2 .. -67.0 over |y| <= 49.11 and
# |z| <= 15.36, with 1.2 mm of air behind the wall and no way through.  Measured on the
# solid: the bus window 100 % blocked, the XT30 100 %, the balance lead 100 %, the XT60's
# top 3.61 mm - and the XT mating halves stand PANEL_LIP_T proud of x = -63, i.e. to
# -64.5, which is already 0.3 mm INSIDE the plate.  Not one of them could ever have been
# plugged in.  `interference()` pairs the static body parts and the cradles WERE
# chassis_bottom, and a part cannot interfere with itself; `isValid()` and `rom_scan` see
# nothing either.  The block below reasoned about the deck bosses at |y| = 32.2 and never
# about the cradle, which was 30 mm closer.
#
# What fixed it is the cradle becoming a bolted part with a FRAME flange (see the CRADLE_*
# block): the frame's opening is y +-43.11, z +-9.36, and everything here is laid out
# inside it and around the four CRADLE_BOLT bosses, whose tray-side seats reach y = 23 and
# z = 16.5.  `panel_clear()` probes each opening along -x against the assembled body on
# every run, and it is a failure line like `!! INTERFERENCE`, not a note.
# THE XT60 IS THE ONE THAT DOES NOT FIT INSIDE THE FRAME.  16.5 mm of width does not
# survive the bus window and the two bosses, so it goes ABOVE the cradle instead: the
# rear wall is clear of everything for z = 15.36 .. 25, the interior there is the
# ESP32/URT-1 strip, and the deck closes it at 25.  That band is 9.64 mm for an 8.5 mm
# body - 0.6 mm of margin top and bottom, so PANEL_AT's z is not a round number and moving
# BODY_Z1, CRADLE_Z1 or PANEL_XT60 moves it.  The old note here - "the clear strip between
# the window's edge and the rear deck boss's inboard face at |y| = 32.2 is 16.2 mm" - was
# stale twice over: the deck screws went to |y| = 41 with the battery module, so that face
# is at 35.2, and the strip was never clear in the first place.
PANEL_XT60   = (16.5, 8.5)            # body over the moulding, + fit         **verify**
PANEL_XT30   = (12.0, 6.6)            #                                       **verify**
PANEL_BAL    = (13.0, 6.0)            # JST-XH 3S plug, passing through       **verify**
PANEL_T      = 8.0                    # pocket depth, from the wall's outer face
PANEL_LIP    = 0.8                    # ... the lip left at the outer face, all round
PANEL_LIP_T  = 1.5                    # ... and how thick that lip is
PANEL_WIN    = (22.0, 16.0)           # the bus window: w x h, on the centreline.  It was
                                      # 32 x 20 and blocked; 22 keeps 1 mm off the bolt
                                      # bosses at y = 12 and 16 fits the frame's opening.
PANEL_AT     = ((0.0, 20.0, PANEL_XT60),      # (y, z, size) - XT60 ABOVE the cradle,
                (32.0, 0.0, PANEL_XT30))      # XT30 outboard of the bolt boss
PANEL_BAL_AT = (-32.0, 0.0)           # ... balance lead, mirrored on the other side
OPI_X        = -22.0                  # Orange Pi 5 Pro board centre on the deck
OPI_HOLES    = (92.0, 54.0)
OPI_STAND_R, OPI_STAND_H = 4.8, 7.0   # standoff: r fits an M2.5 nut slot, h clears the nut
OPI_NUT_DZ   = 1.5                    # ... its floor, above the deck's top face
# THE REAR PAIR HAS NO DECK UNDER IT, and the deck grows a tab rather than the hole
# pattern moving.  OPI_HOLES at OPI_X puts that pair at x = -68 on a deck that ends at
# -63: at OPI_STAND_R the standoff's nearest material is at -63.20, so it floated 0.2 mm
# clear of the edge and chassis_top came back as THREE solids - the deck, and two 383.8
# mm3 bosses attached to nothing.  isValid() and Volume() > 0 both pass on that, and
# PART_SOLIDS said 3, so it shipped.  OPI_HOLES is still **verify** and has never met a
# real board (README, "Orange Pi 5 Pro"), so it is deliberately NOT adjusted to make the
# geometry close - that would bake a guess into printed material.  Moving OPI_X forward
# is not available either: the Pi would have to come 9.8 mm forward and OPI_BOX would
# then run into the LiDAR pedestal, which is the same clash LIDAR_BASE_R was shrunk to
# 26.0 to avoid.  So the deck is extended locally under each rear standoff, and the tab
# DISAPPEARS on its own the moment a measured hole pattern puts the standoff back on the
# deck (chassis_top only unions it while OPI_TAB_X is outboard of the deck's end).
# Nothing is behind the rear wall at deck height to object: measured 0.0 mm3 against
# chassis_bottom, cradle_rear (z <= 15.36) and battery_case over x -80..-63, |y| <= 36,
# z 25..29, and the XT60 above the cradle tops out at the deck's underside.
OPI_TAB_RIM  = 1.2                    # ... rim of deck left round the standoff's foot
OPI_TAB_X    = OPI_X - OPI_HOLES[0]/2 - OPI_STAND_R - OPI_TAB_RIM
OPI_TAB_W    = OPI_STAND_R + OPI_TAB_RIM      # ... half width of the tab, in y
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
# WHERE THE BAY IS, AND WHY IT IS NO LONGER UNDER THE DECK.  The board used to hang from
# two tabs bridging the deck's own window, component face down, in the 3.6 mm between the
# pack's top at 21.4 and the deck's underside at 25 - 2.8 mm of board and 0.8 mm of
# margin.  That slot was the only reason the battery could not be a module: a cased pack
# needs about 3.5 mm the bay did not have, and this was the only 3.5 mm anywhere near it.
# So the board moved UP, out of the tray entirely, onto the deck's TOP face - inside the
# Orange Pi's own standoff gap, which is 7 mm of air nothing else uses.  Two M2.5
# standoffs on the centreline, component face DOWN into the gap under the board, the same
# nut-in-a-slot as the Pi's own four.  The battery module's lid now runs to 23.6 and the
# deck's underside above it is solid.
#
# WHAT THAT COSTS, STATED PLAINLY: the site rises from z = 23.4 to 31.0, so |r| from the
# body origin grows by 7.6 mm and the omega x (omega x r) term the paragraph above is
# about grows with it - about a third, on the centreline, where the whole term is small.
# It is not a guess that has to be lived with: rl/checks/imu_placement.py measures exactly
# this by adding real accelerometers at candidate mounts through MjSpec, and it is the
# check to re-run whenever this block or the gait moves.  The board is still ON the
# centreline, which is the part of the argument above that mattered - 42 deg of apparent
# tilt was a board out beside the Pi in x and y, not one 7.6 mm higher.
#
# imu_clear() still exists but now looks UP: the wall that is thin is the Pi's own board
# over it (OPI_BOX's floor), not a battery under it.
IMU_BOARD    = (20.0, 15.0, 1.6)      # PCB: x, y, thickness          **verify** ref/imu/
IMU_STACK    = 1.2                    # components under the PCB, and HEADERLESS: a
                                      # 2.54 mm pin header is 8.5 mm and would fill the
                                      # standoff gap.  Solder to the pads.  **verify**
IMU_HOLE_P   = 15.0                   # M2.5 mounting holes, on x     **verify** ref/imu/
IMU_X, IMU_Y = 0.0, 0.0               # the centreline - as near the body origin as the
                                      # deck allows, which is what the whole block is for
IMU_STAND_R  = 4.4                    # standoff: 3.35 mm of wall round an M25_TAP hole,
                                      # against the >= 1 x D the rule wants; two of them
                                      # IMU_HOLE_P apart still leave 6.2 mm between their
                                      # walls for the board's own components
IMU_STAND_H  = 2.0                    # ... and its height: IMU_STACK plus 0.8 mm, so the
                                      # component face clears the deck's top face
IMU_TAP_L    = 5.0                    # M2.5 FORMED thread, 2 x D, down through the 2 mm
                                      # standoff and 3 mm into the deck's own 4 - it does
                                      # not break through the underside.  This is the
                                      # second of the two places on the robot that thread
                                      # into plastic (battery_lid() is the other): a 3 g
                                      # breakout that holds nothing but itself, exactly
                                      # the case CLAUDE.md's "off the torque path" rule
                                      # was written for.  A nut cannot go here - a 2 mm
                                      # standoff cannot swallow a 2.25 mm pocket, and
                                      # under the deck is the battery module.  It is a
                                      # ONE-ASSEMBLY thread; a board reseated often wants
                                      # the hole drilled out and a nut under the deck.
IMU_WINDOW   = (-34.0, -12.0)         # the deck's cable window, x.  It used to be at
                                      # x +-16 and the IMU hung in it; the board needs
                                      # solid deck under it now, so the window moved off
                                      # the centreline to the Pi's side.  It clears the
                                      # board's -x hole at -7.5 by 4.5 mm, and |y| <= 34
                                      # keeps it clear of the deck screws at |y| = 41.
IMU_Z0       = BODY_Z1 + DECK_T + IMU_STAND_H + IMU_BOARD[2]
                                      # the PCB's TOP face, at 32.6.  imu_xyz() takes one
                                      # board thickness off it to get the component face,
                                      # which is where the BMI088's package actually is,
                                      # exactly as before - only the plane moved.
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
# POINT RATE AND FRAME RATE ARE MEASURED.  They were catalogue figures - 21600 pts/s at
# 10 Hz - and the real L2 does neither: 62341 points per second in 240 frames over 20.0 s,
# i.e. 12.0 Hz and 5196 points per frame, on firmware 2.8.11.1.  That is 2.9x the point
# rate the sim had been casting, and it was wrong in the direction that flatters the
# model, so anything that read "the cloud is sparse" off this repo before 2026-09-05 read
# it off a number nobody had checked.  Two independent captures 5 s and 20 s apart agree
# to 2 parts in 62000, so this is the sensor's clock and not a sample.  Method:
# ref/lidar/README.md; the capture is ref/lidar/l2_room.pcd.
#
# LIDAR_FRAME_HZ lives here rather than in lidar.py for the reason every other number in
# this block does: it is the sensor's, both sim exporters write it into the model, and
# lidar.py's own header is explicit that its file-local constants are the ones that are
# *guesses*.  It stopped being a guess when the sensor arrived.
#
# The remaining two are still catalogue, and still **verify** in README.md: the range
# window (the room the capture was taken in is 10 m across, which tests neither end) and
# the range noise (which needs a flat wall at a known standoff, not a room).
#
# LIDAR_FOV is the "360 x 90" of the catalogue read as what it is - a cone of half-angle
# 90 deg about the sensor's own axis, from that axis down to its base plane.  NEGA, the
# factory default, buys 6 more degrees BELOW that plane and nothing else.  The capture
# reaches 96.4 deg off-axis, so NEGA is confirmed as the envelope; FOV is left at 90 as
# the base-plane figure it has always been.  lidar_fov_clear() has always measured against
# 96; it just used to spell it inline.
LIDAR_FOV, LIDAR_FOV_NEGA = 90.0, 96.0    # scan cone half-angle about the axis, deg
LIDAR_RATE     = 62340.0              # points per second                     MEASURED
LIDAR_FRAME_HZ = 12.0                 # clouds per second out of the sensor   MEASURED
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
# So the arm rises out of its pad first and makes exactly one 45 deg run inboard, and
# where it turns is derived, not chosen.  (That first run was vertical while the feet sat
# at |y| = 38; on the deck's real screws at 41 it leans 18.4 deg - 3 mm inboard over 9 mm
# of rise, which is well inside the same 45 deg print rule and changes nothing else.
# GPS_KNEE stays at 38: taking it out to 41 would buy 3 mm of inboard reach at 3 mm of
# extra mast height, for nothing.)  A rod leaning 45 deg carries its
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
# THE FEET ARE READ FROM DECK_SCREWS AND NOT TYPED BESIDE IT, and that is the whole fix
# for a defect that shipped: this line was `GPS_X, GPS_Y = -52.0, 38.0`, and when the
# battery module pushed all four deck pairs from |y| = 38 to 41 the mast stayed behind.
# gps_mount then drilled its feet at (-52, +-38) and chassis_top drilled the deck at
# (-52, +-41) - 3 mm apart, so the one M3 x 24 that is supposed to pass through both
# parts could pass neither.  NO CHECK IN THIS FILE COULD SEE IT: the two parts share no
# solid, so interference() reads clear, and a fastener that lines up is not a volume.
# Reading the pair means the mast now follows the deck wherever it goes.  Fixed
# 2026-09-11.
GPS_X, GPS_Y = DECK_SCREWS[0][0], DECK_SCREWS[0][1]   # the deck's rear boss pair
GPS_PAD_R, GPS_PAD_H = 4.8, 10.0      # r is 1 x D of wall round an M3 clearance hole,
                                      # which is the whole job of the pad.  At |y| = 41
                                      # it reaches 45.8 into the deck's stiffening lip
                                      # (43 .. 46, z 29 .. 35) and it may not be shrunk
                                      # to fit: clipped at 42.8 the wall outboard of the
                                      # hole is 0.05 mm.  The lip gives way instead - see
                                      # DECK_LIP_* - and the pad is unchanged.
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
CAM_LEDGE    = CRADLE_Z1              # 15.36 - the front cradle's own top face, and the
                                      # only flat surface anywhere near the camera.  It
                                      # used to be spelled S_W/2 + SLEEVE_W a second time
                                      # here; it is the cradle's half height, so it is
                                      # read from there.  Note what that means now the
                                      # cradle is a bolted part: the camera's floor and
                                      # its two nuts are on cradle_front, not on the tray.
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
CAM_LENS_REL = 2.0                    # relief round the @14 holder, on radius: the mount
                                      # must not vignette its own camera
# ... AND THE WALL IN FRONT OF THE BOARD HAS TO BE ATTACHED TO SOMETHING.  It was not.
# The board pocket ran the full length of the channel, which severed that wall from the
# skirt behind it, and camera_mount came back as three bodies: the channel, plus the wall
# in two loose pieces of 529.3 and 104.9 mm3 (x 66.37..68.00, z 25.53..31.96), split by
# the lens relief.  PART_SOLIDS said 3 - "the channel plus its two rails", which they
# never were - so the count check saw the number it had been told to expect and the one
# feature that stops the board falling out forward printed as two chips of plastic.
# TWO TIES PUT IT BACK, and they are the same idea in two places: the pocket stops where
# the board is not.
#   * at the FAR end the pocket now ends CLR short of the board's own end instead of 1 mm
#     past the channel's, which leaves 2.06 mm of end wall.  That ties the long piece and
#     it also gives the board a positive stop in y, which it did not have;
#   * at the NEAR end the board is inserted through, so nothing may close it - the tie
#     goes OVER the board instead.  The channel's top rises CAM_CAP above the pocket's own
#     ceiling from the lens relief to the board's near end, which is 10.4 mm of roof clear
#     of both CAM_FOOT_Y screws (23 and 29), so neither counterbore moves and the M3 x 20
#     in the BOM is unchanged.
# There is room above: the ceiling here is LIDAR_BASE_FLAT = 63.5, i.e. the pedestal is
# cut away in front of the mount's own back plane, and measured, a roof to z 33.9 shares
# 0.0 mm3 with lidar_mount and leaves lidar_fov_clear at +22.6 deg outside the 96 deg
# cone (it was +23.1).  Fixed 2026-09-11.
CAM_CAP      = 1.2                    # roof over the board pocket at the near end: three
                                      # 0.4 mm perimeters, and it carries nothing but a
                                      # 4 g board that must not lift out of its slot

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
# gps_mount, cradle_front, cradle_rear, battery_case, battery_lid, camera_mount and
# cell_holder are deliberately absent: none has been sliced
# yet, so part_rho() gives them PRINT_FILL_MEAN.  Slice them and put the measured factor
# in the table.  chassis_bottom's own 0.93 was measured on the part WITH the four cradles
# in it and is now the tray's alone - it is a thin-walled box either way, so the number is
# still the right order, but it is one of the things to re-slice.
PRINT_FILL_MEAN   = 0.92          # PETG parts, mass-weighted - the fallback for an
TPU_FILL_MEAN     = 0.65          # unmeasured part and for consumers that carry a union
PRINT_RHO         = PETG_RHO * PRINT_FILL_MEAN   # g/cm3 - solid volume x this
TPU_RHO           = TPU_MAT_RHO * TPU_FILL_MEAN  # the feet
SERVO_KG          = 0.060         # ST3215 incl. both hubs and the bolts - **verify**: not in
                                  # ref/, vendor figure only.  Weigh one before trusting it;
                                  # 12 of them are a third of the robot.
N_SERVO           = 12
BATTERY_KG        = 0.42          # the six wrapped cells alone - 3S2P, 6 x 21700.  NOT
                                  # the module: battery_case and battery_lid are printed
                                  # parts with PARTS entries, so their mass comes off
                                  # their own solids like every other part's.  It hangs
                                  # at brick_com(), which is not the module's centre.
BMS_KG            = 0.055         # **verify** - split out of ELECTRONICS_KG when the BMS
                                  # moved inside the battery module.  It is no longer at
                                  # opi_com() and never was; it used to be averaged into
                                  # the Pi's box 46 mm away and 40 mm up.  Hangs at
                                  # bms_com().
ELECTRONICS_KG    = 0.195         # Orange Pi 5 Pro / wiring - was 0.25 with the BMS in it
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
SERVO_STALL_NM    = 4.50          # MEASURED 2026-09-10 on torque_rig, NOT the vendor's
                                  # 2.94 (30 kg*cm).  A scale in newtons at a 170 mm arm,
                                  # three duty rungs at 12 V: k_u = 0.400 N*m/V, friction
                                  # intercept -0.30 N*m, extrapolated to full duty.  See
                                  # the block below - this is what settled PLAN.md 2c.
SERVO_NOLOAD_RADS = 3.86          # MEASURED 2026-09-11 on the bench stand, free hub,
                                  # 12.1 V, robot/bench/noload_speed.py: 3.864 rad/s from
                                  # the position, 3.835 from PRESENT_SPEED.  The vendor's
                                  # 4.71 (0.222 s / 60 deg) was 22 % high.  It is a
                                  # PLATEAU, not a back-EMF ceiling: TORQUE_LIMIT 800 and
                                  # 1000 read the same speed and the register sits at a
                                  # flat 2500 counts/s, while the rungs below are linear
                                  # at k_e = 2.32 V*s/rad.  Whether that plateau is the
                                  # position loop's profile or its PWM ceiling is open
                                  # (`noload_speed.py --pwm` decides it), and the stall
                                  # above depends on the answer - see PLAN.md 3c.

# MuJoCo joint feel.  MEASURED, 2026-09-09, on the bench rig - this block used to
# say "NOT measured ... plausible values" and every number in it has now been
# replaced by one off a real ST3215.  They live here and nowhere else: both sim
# exporters (export_sim.py and the ROS 2 generate_model.py) read them from this
# block, because they used to each carry their own literals and silently diverged
# - 0.5/0.01/0.05/20 against 0.12/0.008/0.02/25 - which is the servo-mass failure
# this file's mass block already records, repeated one section down.
# rl/checks/check_model.py is what caught it.
#
# Where each comes from, and why it is the right quantity for a POSITION actuator
# with no electrical model behind it - which is what both exporters emit, and the
# reason three of these four differ from what rl/ uses.  rl/model.py zeroes
# damping and frictionloss because rl/actuator.py supplies them as physics; here
# there is no back-EMF, no winding and no duty, so everything the real servo does
# through those has to arrive as joint damping, joint friction and a gain:
#
#   MJ_KP           the friction-cancelled position-loop stiffness at 12 V.
#                   40.9 N*m/rad, NOT the 28.8 measured before: a hold ladder
#                   walked in one direction bills the friction band to elasticity,
#                   and only approaching every angle from both sides separates
#                   them.  It is 25 -> 40.9 here and that is a correction, not a
#                   tuning choice.  The old 25 was the value that reproduced the
#                   observed droop WITHOUT a friction term; with MJ_FRICTIONLOSS
#                   below finally set to something real, 25 would be soft twice.
#   MJ_FRICTIONLOSS the Coulomb friction, 0.184 N*m, from two independent routes
#                   that had disagreed by 2x and now agree (0.184 bidirectional
#                   ladder, 0.186 free swing).  The old 0.02 was a tenth of it.
#                   Note this is the one place the STATIC friction can be modelled
#                   at all: MuJoCo's frictionloss is a proper stick-slip
#                   constraint, whereas rl/actuator.py's tanh(w/v_eps) is exactly
#                   zero at rest and cannot hold a joint (PLAN.md step 2b).
#   MJ_DAMPING      the TOTAL speed-proportional torque, 1.37 N*m*s/rad, measured
#                   to 5 % over three supply voltages.  Deliberately the total and
#                   not the viscous part: that total is b_v + k_u*k_e, this bench
#                   cannot split them (both cost a motor voltage proportional to
#                   omega and neither depends on supply), and a position actuator
#                   has nowhere to put back-EMF anyway.  It is 11x the old 0.12,
#                   and it is what makes the model run out of speed near 1.8 rad/s
#                   the way the real servo does.
#   MJ_ARMATURE     the reflected rotor inertia, 0.0165 kg*m2, from the free swing
#                   at the corrected friction.  Do NOT use the 0.024-0.042 range
#                   this project quoted before: that was computed against a tau_c
#                   half the real size, and J = (m*g*r*sin q0 - tau_c)/alpha makes
#                   J_m inherit every error in tau_c amplified by the ratio of the
#                   two.  rl/actuator.py's Params.J_m default is still 0.008 and
#                   should stay there - it is flagged as an unfitted vendor prior,
#                   and rl/params/st3215.json carries the fit.
#
# Provenance for all four: robot/bench/{sweep,hysteresis,fit_bam}.py over
# robot/bench/data/, three voltages, 1.066 kg on a 90 mm arm.  The public
# write-up is ST3215_STS3215_measured_parameters.md.
#
# THE REGISTER-SCALING QUESTION IS SETTLED, AND BOTH SIDES OF IT WERE WRONG.
# The hysteresis ladder put the torque constant at 2.39 N*m/A against a vendor
# 1.09 - two register channels agreeing to 0.3 % and implying a 7.4 N*m stall -
# while the datasheet said 2.94.  A scale settles it at 4.50, BETWEEN the two:
# the register channel is high by 1.64x, not the 2.2x this note used to assume,
# and the vendor figure is low by 1.53x.  So there was never one mis-scaling to
# find; the registers are off AND the datasheet is optimistic in the other
# direction.  The four constants above are unaffected either way - each is a
# ratio in the same register units converted through the known m*g*r, or an
# inertia off a timed fall - which is why they did not have to be re-measured
# when SERVO_STALL_NM moved.
#
# Method, because it is the part that was hard: robot/bench/torque_hold.py holds
# a blocked push for 8 s while robot/bench/torque_limit.py caps TORQUE_LIMIT.
# Do NOT read a scale off sweep.py --traj stall: its 0.8 s bursts are right for
# the electrical fit and wrong for a scale, inflating the reading by a roughly
# CONSTANT ~130 g of arm inertia and filter ringing, which distorts the slope as
# well as the level.  Full write-up in robot/README.md, "The stall torque, in
# newton-metres".
MJ_DAMPING        = 1.37          # N*m*s/rad at the joint (b_v + back-EMF)
MJ_ARMATURE       = 0.0165        # kg*m2, reflected rotor+gearbox inertia
MJ_FRICTIONLOSS   = 0.184         # N*m, Coulomb
MJ_KP             = 40.9          # position-actuator gain, N*m/rad at 12 V
MJ_DAMPRATIO      = 1.0           # critically damped.  export_sim.py carried no
                                  # dampratio at all, which is what made its kp
                                  # incomparable with the ROS 2 one rather than
                                  # merely different.

# The foot's contact.  Same rule as the block above and for the same reason: it was the
# duplicated-constant defect a fifth time, and this one had already diverged - export_sim.py
# wrote friction="1.2 0.05 0.001" on the foot geom while generate_model.py's `foot` default
# class wrote "1.2 0.02 0.001", so the two files described robots whose feet gripped
# differently in torsion by 2.5x, and which sim you loaded decided which foot you had.
#
# CONDIM is the substance of the fix.  Both exporters shipped the foot at MuJoCo's default
# condim 3, which uses only the FIRST column of the friction it declares - so the torsion
# and roll numbers above were in both files and read by neither, and a planted foot resisted
# no twist at all.  The foot is a sphere, so it makes exactly one contact point at any
# attitude; with condim 3 that point was a frictionless pivot.  4 adds torsion.
#
# It is 4 and not 6, and that is measured, not taste.  6 adds the ROLLING dimension too,
# and over terrain seeds 7..12 it takes the trot from 657 +-35 mm to 615 +-124: the mean
# moves less than the spread, but the spread itself more than triples, which is the walker
# wandering rather than the walker working harder.  The column is not even the cause - at
# condim 6 with the roll coefficient set to zero it is still 612 +-72, because condim 6
# adds the rolling constraint to the solver whatever its coefficient is.  condim 4 reads
# 653 +-28, i.e. the same distribution as the old behaviour and the tightest of the five.
# It is also the better physics: a foot is not a wheel, and the leg already resists the
# dome rolling.  Do not "upgrade" this to 6.
#
# The torsion figure is derived, not chosen.  MuJoCo's torsional coefficient has units of
# LENGTH: it caps the twist torque at ~(2/3)*a*mu*fn for a patch of radius a.  Measured off
# the flat trot the foot carries ~25 N mean and 57 N peak; 95A TPU works near 4 MPa, so the
# dome's patch is r = 1.4..2.1 mm and the coefficient is 0.667 * 2.0e-3 * 1.2 = 1.6e-3 m.
# Both shipped numbers were an order of magnitude generous (0.02) or 25x (0.05) - nobody
# had picked them, they had simply never been used.  The roll column is kept at a plausible
# 1e-3 m so the number is there and honest, but condim 4 does not read it.
#
# Do not expect any of this in a distance, and that is the point rather than a
# disappointment: ros2/tools/foot_contact.py measures the whole question - four ways of
# growing the contact patch, two controls - and NONE of them moves the walker outside the
# seed noise.  Whether the patch is one point or nine millimetres across, the compliance
# that decides how far the body gives under a push is MJ_KP, not the foot.  What this fix
# buys is that the model stops discarding friction it declares.  See ros2/README.md,
# "The foot's contact patch", for the full table and for why the heightfield cannot be
# used to compare feet at all.
# One more level of the same trap, found by reading the COMPILED contacts rather than the
# files: MuJoCo does not use a geom's friction, it uses the elementwise MAX of the pair
# unless one geom has the higher `priority`.  Both exporters' feet declared the same
# numbers after the fix above and still met the ground with different ones, because their
# floors differ - export_sim.py's global default is friction="0.9 0.02 0.001" and the ROS 2
# floor is "1.0 0.005 0.0001", so the effective torsion came out 0.02 against 0.005, a
# factor of four, in two files that now agreed.  The foot is the part whose friction is
# derived from something, so the foot wins the pair: both exporters give it priority 1.
# Check this the same way if it is ever touched - d.contact[i].friction, not the XML.
#
# `priority` settles solref and solimp as well, and that is deliberate rather than a side
# effect worth hiding: without it the ROS 2 foot met the ground at solref 0.014 and solimp
# 0.925/0.97 - an average of the foot's declared 0.008 / 0.95 0.99 with whatever the floor
# happened to say - so the foot's contact was as soft as the scenery it landed on, and a
# different softness in each exporter.  Now both read the foot's own numbers.  It is a real
# change and it re-baselined the gait figures in CLAUDE.md step 6, every one of them
# upward, with the condim-3 control re-run beside it on the same tree.
MJ_FOOT_CONDIM    = 4             # slide + torsion.  3 discards torsion; 6 costs, see above
MJ_FOOT_FRICTION  = (1.2, 0.002, 0.001)   # slide, torsion (m), roll (m; unread at condim 4)
MJ_FOOT_PRIORITY  = 1             # the foot's friction wins the pair, not the floor's

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
def tri(pts, z0, z1):
    """triangular prism from three (x, y) points, z0..z1 - for tapering a step out of a
    box.  A re-entrant corner between two boxes is a stress riser and fea.py sees it."""
    w = cq.Workplane("XY", origin=(0, 0, z0)).moveTo(*pts[0])
    for q in pts[1:]:
        w = w.lineTo(*q)
    return W(w.close().extrude(z1-z0).val())
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

def overlap(a, b):
    """mm3 shared by two solids, and inf if the boolean FAILED.

    Every clearance probe in this file asks the same question and they used to answer a
    failed intersect three different ways: 0.0 (a pass), -1.0 (a sentinel nobody tested
    for) and, in rom_scan, a value that counted the angle as FREE.  OCC drops an
    intersection on a degenerate contact without raising anything downstream notices - the
    same silent failure the chassis_bottom -257 mm3 note is about - so the only safe
    reading is the PESSIMISTIC one: a boolean that did not run has not cleared anything.
    Callers compare against INTERF_TOL, and inf fails every one of those comparisons."""
    try:
        return a.intersect(b).Volume()
    except Exception:
        return float("inf")

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
        z0 = HUB_TOP_Z-HUB_T_TOP
        h = cyl(HUB_D/2, HUB_T_TOP, (0,0,z0))
        h = h.cut(cyl(HUB_CTR_D/2, HUB_T_TOP-HUB_SCR_T, (0,0,z0)))
        h = h.cut(cyl(HUB_SCR_D/2, HUB_T_TOP+2, (0,0,z0-1)))
    else:
        z0 = HUB_BOT_Z
        h = cyl(HUB_D/2, HUB_T_BOT, (0,0,z0))
        h = h.cut(cyl(HUB_CTR_D/2, HUB_T_BOT+2, (0,0,z0-1)))
    for i in range(HUB_N):
        th = math.radians(90*i)
        h = h.cut(cyl(HUB_BOLT_D/2, 12, (HUB_BC/2*math.cos(th), HUB_BC/2*math.sin(th), z0-1)))
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

def thrust_bolts():
    """The two clamp screws, in sleeve-local coordinates.

    Hardware, with the same standing as hubs() and servo_dummy(): no printed part is cut
    against it.  It exists because rom_scan sweeps solids rather than intentions, and the
    screw reaches further out of the lug than the lug does - which is exactly how a leg
    that passes every check in this file can still bind on its own clamp.

    The tip sits on the case's -x face with the case jacked fully forward, which is where
    a tightened jack screw leaves it and the shortest the screw can look."""
    x_tip  = -S_AX + CLR                          # case -x face, pushed onto the front legs
    x_head = x_tip - THRUST_BOLT_L
    s = None
    for sg in (1, -1):
        b = cyl(3.0/2, THRUST_BOLT_L, (x_head, sg*THRUST_Y, 0), axis=(1,0,0))
        if THRUST_HEAD_H > 0:
            b = b.union(cyl(THRUST_HEAD_D/2, THRUST_HEAD_H,
                            (x_head-THRUST_HEAD_H, sg*THRUST_Y, 0), axis=(1,0,0)))
        s = b if s is None else s.union(b)
    return s

def fork_screws():
    """The eight hub screw HEADS of one fork, in servo-local coordinates - hardware, with
    the same standing as thrust_bolts(): no printed part is cut against it, it exists so
    that rom_scan sweeps what is actually there.  The driven arm's four stand HUB_HEAD_H
    proud of its outer face; the passive arm's four sit FORK_CB into it.  The shanks are
    inside the arm and the hub and cannot touch anything, so they are not modelled - the
    length is checked in closed form in head_clear()."""
    s = None
    for i in range(HUB_N):
        th = math.radians(90*i)
        px, py = HUB_BC/2*math.cos(th), HUB_BC/2*math.sin(th)
        for z0 in (HUB_TOP_Z + ARM_T, FORK_Y0 + FORK_CB - HUB_HEAD_H):
            h = cyl(HUB_HEAD_D/2, HUB_HEAD_H, (px, py, z0))
            s = h if s is None else s.union(h)
    return s

def head_clear():
    """The passive-arm screw head against the chassis gusset at the hip, and the M3 x 6
    against the servo case on both arms.  Returns (gap, spare_passive, spare_driven), all
    of which must stay positive.  Closed form, like thrust_clear(), for the same reason:
    a sweep steps over sub-millimetre fouls and this cannot."""
    gap   = FORK_GAP - (HUB_HEAD_H - FORK_CB)
    reach = HUB_SCREW_L - (ARM_T - FORK_CB) - (HUB_BOT_Z - ARM_BOT_TOP) - HUB_T_BOT
    spare_p = (HUB_BOT_Z + S_H/2) - reach                 # the base recess, minus the tip
    spare_d = (HUB_TOP_Z - HUB_T_TOP - S_H/2) - (HUB_SCREW_L - ARM_T - HUB_T_TOP)
    return gap, spare_p, spare_d

def fork(spine_r0=SPINE_R0, spine_r1=SPINE_R1, spine_w=SPINE_W):
    """distal-link end; link direction is servo -X."""
    def arm(z0, z1):
        a = cyl(ARM_R, z1-z0, (0,0,z0))
        a = a.union(bxc(-spine_r1, 0.0, -spine_w/2, spine_w/2, z0, z1))
        a = a.cut(cyl(3.2, 40, (0,0,z0-10)))
        for i in range(HUB_N):
            th = math.radians(90*i)
            px, py = HUB_BC/2*math.cos(th), HUB_BC/2*math.sin(th)
            a = a.cut(cyl(M3_CLR, 40, (px,py,z0-10)))
        return a
    top = arm(HUB_TOP_Z, HUB_TOP_Z+ARM_T)
    bot = arm(ARM_BOT_TOP-ARM_T, ARM_BOT_TOP)
    bot = bot.union(cyl(HUB_REC_D/2-1.0, HUB_BOT_Z-ARM_BOT_TOP, (0,0,ARM_BOT_TOP)))
    z = ARM_BOT_TOP-ARM_T-1                       # re-drill through arm + pedestal, or the
    bot = bot.cut(cyl(3.2, 40, (0,0,z)))          # pedestal plugs the four M2.5 holes
    for i in range(HUB_N):
        th = math.radians(90*i)
        px, py = HUB_BC/2*math.cos(th), HUB_BC/2*math.sin(th)
        bot = bot.cut(cyl(M3_CLR, 40, (px, py, z)))
        # counterbore for the button head, from the outer face - see FORK_CB.  The top
        # arm gets none: there the M3 x 6 has only 0.3 mm behind the hub before the case.
        bot = bot.cut(cyl((HUB_HEAD_D+0.5)/2, FORK_CB+1.0, (px, py, z)))
    # No nut pockets here.  The screw is driven from the outside and threads into the
    # stock aluminium hub - the @2.5 holes in both plates are tapped, and there is nowhere
    # for a nut anyway: behind the driven hub sit 0.30 mm to the case top, behind the
    # passive one the 0.55 mm base recess.  A hex pocket would also take 2.2 of the 4.0 mm
    # arm exactly under the screw head, where the bolt load enters the part.
    # Screws: ISO 7380 M3 x 6 everywhere.  Driven: 4.0 arm + <=2.5 hub, 6.0 engages 2.0 of
    # the hub with 0.3 to the case top.  Passive: (4.0 - FORK_CB) arm + 0.95 pedestal
    # + 2.2 hub = 5.95, so 6.0 is through the hub with the 0.55 base recess to spare.
    # Longer bottoms out on the case - the vendor FAQ warns it burns servos.  (It was
    # M3 x 7 on the passive side, a length nobody stocks, before the counterbore.)
    # M3 and not M2.5: HUB_BOLT_D is measured on the hub, the STEP's @2.5 is not what
    # arrived.  Only the hole grew - @2.9 to @3.4 - so a leg already printed can be
    # drilled out rather than reprinted; the counterbore can be spot-faced the same way.
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
# PART: cradle_front / cradle_rear, and chassis_bottom
# =====================================================================================
def roll_module():
    """ONE HALF of a cradle - the front-left: sleeve + boxed neck back to the tray's front
    wall.  Not a part on its own: cradle() unions this with its own mirror, because the
    two halves overlap on the centreline (this reaches y = -14, the mirror reaches +14).
    Nothing may sit in the swept sector of the rotating fork (outboard, r<31)."""
    s = mv(sleeve(), ROLL_LOC)
    xa, xb = CRADLE_X, ROLL_X + SLEEVE_LEN/2                 # 63 .. 106.5
    xm = ROLL_X - 21.9                                       # rear fork-arm plane
    s = s.union(bxc(xa, xb, -14.0, 2.0, -CRADLE_Z1, CRADLE_Z1)       # inboard rail
                .cut(bxc(xa-1, xb-6.0, -11.0, -1.0, -11.0, 11.0)))
    # The top and bottom straps.  Narrow to STRAP_Y past the fork, then full width - and
    # the two things that are NOT obvious here are both stress, found the first time this
    # part went into fea.py's set (it was chassis_bottom before, and fea.py has never
    # covered that).  land3g read 53.7 MPa peak against a p99 of 7.3 - interlayer SF 0.4.
    #
    #   * the narrow strap used to stop at xm+SLEEVE_W+1 = 72.1 while the wide one starts
    #     at 73.5, leaving a 1.4 mm NOTCH in the strap that carries the whole leg.  It
    #     runs to the sleeve now.  Peak 53.7 -> 26.4.
    #   * the width then STEPPED 21.5 -> 49.11 in one plane, and that re-entrant corner
    #     was the next hottest thing in the part.  The wide strap is tapered back to the
    #     narrow one over STRAP_TAPER instead.
    #
    # STRAP_Y itself is not free and was not over-conservative: rom_scan puts the hip
    # bracket into everything from y = 22 outward at strap height over the fork arm's own
    # x band at +-90 deg of roll.  A version of this that ran full width and cut only the
    # arm's disc read `hip_roll +0 .. +0`.  Do not widen it without re-running the scan.
    for z0, z1 in ((12.36, CRADLE_Z1), (-CRADLE_Z1, -12.36)):
        s = s.union(bxc(xa, ROLL_X-SLEEVE_LEN/2, -14.0, STRAP_Y, z0, z1))
        s = s.union(bxc(ROLL_X-SLEEVE_LEN/2, xb, -14.0, CRADLE_Y1, z0, z1)
                    .cut(tri(((ROLL_X-SLEEVE_LEN/2, STRAP_Y),
                              (ROLL_X-SLEEVE_LEN/2, CRADLE_Y1+1),
                              (ROLL_X-SLEEVE_LEN/2+STRAP_TAPER, CRADLE_Y1+1)),
                             z0-1, z1+1)))
    s = s.union(bxc(xa, xm-FORK_GAP, -14.0, CRADLE_Y1, -CRADLE_Z1, CRADLE_Z1))  # flange
    return s

def cradle_frame():
    """The flange's opening: everything inside the CRADLE_RIB frame is air, and that is
    what the rear connector panel and the bus window come out through.  It used to be a
    smaller pocket that left a continuous 2.8 mm skin across the whole face - see the
    PANEL_* block for what that cost."""
    return bxc(CRADLE_X-1, CRADLE_X+CRADLE_T+1,
               -(CRADLE_Y1-CRADLE_RIB), CRADLE_Y1-CRADLE_RIB,
               -(CRADLE_Z1-CRADLE_RIB), CRADLE_Z1-CRADLE_RIB)

def cradle_bolt_axes():
    """(y, z) of the four screws that hold one cradle on, in robot coordinates.  The same
    tuple positions the cradle's nut bosses and the tray's seats, so the two can never
    drift apart the way a hand-copied pattern would."""
    return CRADLE_BOLT

def cradle_bolts(cradle=True):
    """The four screw features for one end of the robot.

    cradle=True  -> (bosses to union on, cuts to take off the cradle)
    cradle=False -> (seats to union on, cuts to take off the tray)

    Kept in one function on purpose: a bolted joint is the one place in this model where
    two parts have to agree to a tenth of a millimetre, and the way that goes wrong is
    someone editing the hole and not the boss.  Here there is one number for each."""
    boss = cut = seat = tcut = None
    b2 = CRADLE_BOSS/2
    for y, z in cradle_bolt_axes():
        # --- cradle: nut boss out to CRADLE_BOSS_X, register spigot proud of the face
        # clipped to the flange's own footprint: at z = 10.5 a CRADLE_BOSS square stands
        # 0.14 mm proud of CRADLE_Z1, and CRADLE_Z1 is the camera mount's shelf.
        bs = bxc(CRADLE_X, CRADLE_BOSS_X, y-b2, y+b2, z-b2, z+b2).intersect(
             bxc(CRADLE_X-1, CRADLE_BOSS_X+1, -CRADLE_Y1, CRADLE_Y1,
                 -CRADLE_Z1, CRADLE_Z1))
        bs = bs.union(cyl(CRADLE_REG_D/2, CRADLE_REG, (CRADLE_X-CRADLE_REG, y, z),
                          axis=(1, 0, 0)))
        boss = bs if boss is None else boss.union(bs)
        c = cyl(M3_CLR, CRADLE_BOSS_X-CRADLE_X+CRADLE_REG+2,
                (CRADLE_X-CRADLE_REG-1, y, z), axis=(1, 0, 0))
        # the nut channel opens toward the centreline: at x >= CRADLE_NUT_X the only solid
        # is the boss itself and the rail box at |y| <= 14, so it ends in air between them
        c = c.union(nut_slot((CRADLE_NUT_X, y, z), (0.0, -1.0 if y > 0 else 1.0, 0.0),
                             up=(1, 0, 0), run=CRADLE_NUT_RUN))
        cut = c if cut is None else cut.union(c)
        # --- tray: a boss inboard of the wall, deep enough for the register pocket AND a
        # flat head seat.  WALL alone is 2.8 mm and holds neither.
        x0 = CRADLE_X - WALL - CRADLE_SEAT
        st = bxc(x0, CRADLE_X, y-b2-1.0, y+b2+1.0, z-b2-1.0, z+b2+1.0)
        seat = st if seat is None else seat.union(st)
        t = cyl(CRADLE_REG_D/2+CLR, CRADLE_REG+CLR+1,
                (CRADLE_X-CRADLE_REG-CLR, y, z), axis=(1, 0, 0))
        t = t.union(cyl(M3_CLR, CRADLE_X-x0+2, (x0-1, y, z), axis=(1, 0, 0)))
        t = t.union(cyl(CRADLE_CB_D/2, CRADLE_CB+1, (x0-1, y, z), axis=(1, 0, 0)))
        tcut = t if tcut is None else tcut.union(t)
    return (boss, cut) if cradle else (seat, tcut)

def cradle_screws():
    """The eight M3 heads that hold the cradles on, as solids in robot coordinates.

    Here for the reason fork_screws() and thrust_bolts() are: the head is hardware, so no
    check in this file can see it, and this robot has now been bitten by exactly that three
    times - the thrust clamp's cap head inside the fork spine, the hub screws' heads
    against the gusset, and this one, 0.4 mm inside the battery module.  Modelled as the
    ISO 7380 button the rest of the machine uses.
    """
    x0 = CRADLE_X - WALL - CRADLE_SEAT
    h = None
    for sx in (1.0, -1.0):
        for y, z in cradle_bolt_axes():
            c = cyl(HUB_HEAD_D/2, HUB_HEAD_H,
                    (sx*(x0+CRADLE_CB), y, z), axis=(-sx, 0.0, 0.0))
            h = c if h is None else h.union(c)
    return h

def cradle(front=True):
    """One end of the robot's hip-roll cradle, as ONE printed part: two sleeves, the rail
    box back to the tray's end wall, the top and bottom straps, and the flange frame that
    bolts to it.  The rear is not this mirrored in y - that is the same part - it is this
    mirrored in x, so the two ends are two prints, one off each.

    Which is why only the FRONT carries the camera's two nuts: CAM_LEDGE is this part's
    top face, and with front and rear already distinct there is nothing to be gained by
    drilling the rear for a camera it does not have."""
    s = roll_module()
    s = s.union(mirY(s))                       # the two halves, overlapping at |y| <= 2
    # ONE duct, not the two the halves bring.  Their ducts are y -11..-1 and 1..11, so
    # unioning the halves leaves a 4 mm web straight down the centreline - which is
    # exactly where the bus window comes out.  panel_clear() found it; interference()
    # never could, because it is one part standing in front of its own hole.
    s = s.cut(bxc(CRADLE_X-1, ROLL_X + SLEEVE_LEN/2 - 6.0, -11.0, 11.0, -11.0, 11.0))
    s = s.cut(cradle_frame())                  # ... and the frame's opening through it
    # the driver's swept circle through the flange, at each roll axis.  Most of it is
    # already inside the frame's opening; what it actually takes out is the crescent past
    # the opening's edge, and that is the difference between the four hub screws being
    # reachable with the cradle in hand and not.
    rel = cyl(FORK_DRIVER_R, CRADLE_T+2, (CRADLE_X-1, ROLL_Y, ROLL_Z), axis=(1, 0, 0))
    s = s.cut(rel).cut(mirY(rel))
    boss, cut = cradle_bolts(cradle=True)
    s = s.union(boss)
    if front:
        # the camera's two M3, into nuts in slots opening forward - open air under the
        # chin, and the only face still reachable with the board in its channel
        for fy in CAM_FOOT_Y:
            s = s.cut(cyl(M3_CLR, 20.0, (CAM_FOOT_X, fy, CAM_LEDGE-CAM_NUT_DZ-4.0)))
            s = s.cut(nut_slot((CAM_FOOT_X, fy, CAM_LEDGE-CAM_NUT_DZ), (1.0, 0.0, 0.0),
                               run=8.0))
        ky, kl, kd = CAM_KEY                   # ... and the pocket its far end keys into
        s = s.cut(bxc(CAM_FOOT_X-3.0-CLR, CAM_FOOT_X+3.0+CLR, ky-kl/2-CLR, ky+kl/2+CLR,
                      CAM_LEDGE-kd-CLR, CAM_LEDGE+1.0))
    s = s.cut(cut)                             # bolt holes and nut channels
    s = s.cut(env_all())
    return s if front else mirX(s)

def cradles():
    """both cradles, placed in robot coordinates - what interference(), the ROM scan, the
    assembly and both sim exporters need.  The base link is the tray plus these."""
    return (cradle(True), cradle(False))

def chassis_bottom():
    s = (bxc(-BODY_L/2, BODY_L/2, -BODY_W/2, BODY_W/2, BODY_Z0, BODY_Z1)
         .cut(bxc(-BODY_L/2+WALL, BODY_L/2-WALL, -BODY_W/2+WALL, BODY_W/2-WALL,
                  BODY_Z0+3.0, BODY_Z1+1)))
    # The four hip-roll cradles used to be unioned on here, all 110 g of them.  They are
    # two bolted parts now (see the CRADLE_* block and cradle()); what the tray owes them
    # is a seat per screw - a boss inboard of the end wall carrying the register pocket
    # and a flat head seat, because WALL alone is 2.8 mm and holds neither.
    seat, tcut = cradle_bolts(cradle=False)
    s = s.union(seat).union(mirX(seat))
    # Battery module seat.  There is no cradle any more - no fins, no end stops, no strap:
    # the pack is a cased module (see the battery block, and battery_case()) and what the
    # tray owes it is a register, not a nest.  A BATT_SEAT-deep recess in the floor,
    # CLR proud of the module all round, locates it in x and y; the deck closes the 0.60 mm
    # above it in z, over a strip of foam that is a BOM line and not a printed feature.
    #
    # The recess is a POCKET in the floor, not a hole through it: the floor is the box's
    # bottom skin and it keeps 2.2 of its 3.0 mm here.  Cut, not unioned - which is why it
    # is safe this early, before anything is added near it.
    s = s.cut(bxc(BATT_X-BATT_L/2-CLR, BATT_X+BATT_L/2+CLR,
                  -BATT_W/2-CLR, BATT_W/2+CLR, BATT_Z0, BODY_Z0+3.0+1))
    # ESP32 + URT-1 bay, in the strip between the module's rear face and the connector
    # panel's pads.  It moved back from x = -46 because the module reaches -46.15 - the
    # battery took the tray, which is the honest consequence of making it a module - and
    # BATT_X was pushed forward until this strip was the 8.9 mm the bay had before.  Its
    # opposite number at x = +46 was the BMS bay and is simply gone: the BMS is inside the
    # module now, which is the point of putting it there - a pack that leaves the robot
    # leaves protected.
    # ESP_RIB_Y and not BMS_W/2+1.5: that half width was the BMS bay's and the BMS went
    # inside the battery module, so it was already stale - and at 15.0 it took 0.5 mm out
    # of the driver's run to the two lower REAR cradle screws.  cradle_clear() found it.
    s = s.union(bxc(ESP_X-1.5, ESP_X+1.5, -ESP_RIB_Y, ESP_RIB_Y,
                    BODY_Z0+3, BODY_Z0+16))
    # deck bosses.  The deck screw lands in a nut, not in a printed thread: the boss is
    # drilled M3 clearance and carries a nut slot near its top, opening toward the middle
    # of the tray - the one direction that is open air with the deck off, which is when
    # the nuts go in.  The screw pulls the nut up against the slot's roof and the deck
    # down onto the boss top, so the 3 mm of boss above the slot is in compression.
    # The mid pair is the exception on both counts - see DECK_SCREWS: it is clipped to the
    # body's own side face, and its channel opens along +x, into the free strip beside the
    # pack, because inboard of it is the battery.
    for x, ay, out in DECK_SCREWS:
        for sy in (-1.0, 1.0):
            y = sy*ay
            b = cyl(DECK_BOSS_R, BODY_Z1-(BODY_Z0+3.0), (x, y, BODY_Z0+3.0))
            s = s.union(b.intersect(bxc(-BODY_L/2, BODY_L/2, -BODY_W/2, BODY_W/2,
                                        BODY_Z0, BODY_Z1)))
            s = s.cut(cyl(M3_CLR, 15.0, (x, y, BODY_Z1-14.0)))
            s = s.cut(nut_slot((x, y, BODY_Z1-DECK_NUT_DZ),
                               (out[0], sy*out[1], out[2]), run=DECK_BOSS_R+6.0))
    for y in (-BODY_W/2-1, BODY_W/2-WALL-1):                  # vents / side cable ports
        for x in (-34.0, 0.0, 34.0):
            s = s.cut(bxc(x-11, x+11, y, y+WALL+2, -4.0, 14.0))
    pw, ph = PANEL_WIN                                        # the bus window
    s = s.cut(bxc(-BODY_L/2-1, -BODY_L/2+WALL+1, -pw/2, pw/2, -ph/2, ph/2))
    # Rear connector panel - see the PANEL_* block.  Pad, then the pocket out of it, then
    # the lip's smaller opening through the wall's outer skin.  The pad is clamped at BOTH
    # ends now: the XT60 sits high enough that an unclamped pad would stand proud of the
    # tray's top edge and foul the deck.
    xw = -BODY_L/2                                            # the wall's outer face
    for cy, cz, (w, h) in PANEL_AT:
        s = s.union(bxc(xw+WALL, xw+PANEL_T, cy-w/2-3.0, cy+w/2+3.0,
                        max(cz-h/2-3.0, BODY_Z0+3.0), min(cz+h/2+3.0, BODY_Z1)))
        s = s.cut(bxc(xw+PANEL_LIP_T, xw+PANEL_T+1, cy-w/2, cy+w/2, cz-h/2, cz+h/2))
        s = s.cut(bxc(xw-1, xw+PANEL_LIP_T, cy-w/2+PANEL_LIP, cy+w/2-PANEL_LIP,
                      cz-h/2+PANEL_LIP, cz+h/2-PANEL_LIP))
    by, bz = PANEL_BAL_AT
    s = s.cut(bxc(xw-1, xw+WALL+1, by-PANEL_BAL[0]/2, by+PANEL_BAL[0]/2,
                  bz-PANEL_BAL[1]/2, bz+PANEL_BAL[1]/2))
    # The camera's two M3 are NOT here any more: CAM_FOOT_X is 65.5, outboard of this
    # part's front face at 63, so both nuts live in cradle_front() - the honest consequence
    # of the ledge the camera stands on being a bolted part.  Its KEY POCKET does still
    # cross the joint, by 0.5 mm, so both sides cut their share of it the way both sides
    # cut the fork channels.
    ky, kl, kd = CAM_KEY
    s = s.cut(bxc(CAM_FOOT_X-3.0-CLR, CAM_FOOT_X+3.0+CLR, ky-kl/2-CLR, ky+kl/2+CLR,
                  CAM_LEDGE-kd-CLR, CAM_LEDGE+1.0))
    #
    # There are no fork-access channels here any more.  They used to cut a @6 hole
    # through this part's front corner post and through the lower half of each corner deck
    # boss, out into the side vent, so a key could reach the roll joint's inboard fork
    # screws - which were blind because the fork could only go on with the cradle already
    # welded to the tray.  The cradle bolts on now, so the fork goes on with it in hand and
    # those screws are in open air; see the block above DRIVER_D.  A chassis printed
    # against the old geometry still has the holes and is not wrong, just drilled.
    #
    # The ORDERING rule they were an example of still stands, and the bolt holes below are
    # the next thing it applies to: cut a through-path AFTER everything that adds material
    # near it.  The bores these replaced ended exactly on the tray's inner face at
    # x = +-60.2, and the rear connector pads are unioned on starting from that same plane;
    # they shared no volume at all, and OCC's fuse on that degenerate contact returned a
    # solid of volume -257 mm3.  isValid() said True, so nothing downstream noticed - 240
    # cm3 of chassis became a 2.5 x 14.9 x 6.9 mm sliver, the ROM scan read hip_roll as
    # +0..+0, and every interference pair fired at once.  build() also fails a part whose
    # volume is not positive, which is the check that names it in one line.
    s = s.cut(tcut).cut(mirX(tcut))           # bolt holes and register pockets, also last
    return s.cut(env_all())

# =====================================================================================
# PART: battery_case / battery_lid
# =====================================================================================
def battery_case():
    """The battery module: six 21700 welded into a 3 x 2 brick, heatshrunk, its BMS beside
    them, and this box around both.  It is a robot part - it has a PARTS entry, a mass in
    the budget and a place in interference() - but it is not structure: nothing on the
    robot loads it, which is why its two lid screws are one of only two places on this
    machine where a screw threads straight into the print - the IMU's standoffs are the
    other (M25_TAP, see CLAUDE.md, "Off the torque path").

    Two zones along x, and the order is not arbitrary.  The BMS stands on edge against the
    REAR wall, so its leads and the pack's leave through the same grommet slot and reach
    the rear connector panel without crossing the cells; the brick fills the rest, right up
    to the front wall.  That FRONT wall is thickened to BATT_FRONT_T and is the module's
    only screwed fixing - the lid's rear edge lives in a groove in the rear wall and its
    front edge takes two M2.5 down into this one.

    The BMS is retained in all six directions with no fasteners: the rear wall and a pair
    of rib pairs take it in x, two ribs off the rear wall take it in y, the ribs' ledge
    takes it down, and the one bar under the lid takes it up.  The y ribs are there
    because the board was MEASURED: the guessed 64.0 outline was a press fit against the
    case's own side walls and wanted nothing, the real 60.13 leaves 2.4 mm a side.  That bar is the only thing hanging below the lid: over the
    brick there is 0.2 mm, not 2.

    Two vents high in the rear wall: a sealed box around six cells is the wrong kind of
    safe.  A cell that vents goes out of the back of the module, away from the deck and
    the electronics above it."""
    s = bxc(BATT_X-BATT_L/2, BATT_X+BATT_L/2, -BATT_W/2, BATT_W/2, BATT_Z0, BATT_ZI1)
    s = s.cut(bxc(BATT_XI0, BATT_XI1, -BATT_YI, BATT_YI, BATT_ZI0, BATT_ZI1+1))
    # the rear wall carries on up to full height, and takes a groove in its inner face at
    # the lid's own height: that groove is what holds the lid's rear edge, and the two
    # screws in the front wall are what hold the other end.
    s = s.union(bxc(BATT_X-BATT_L/2, BATT_XI0, -BATT_W/2, BATT_W/2, BATT_ZI1, BATT_Z1))
    s = s.cut(bxc(BATT_XI0-BATT_GROOVE, BATT_XI0, -BATT_YI, BATT_YI,
                  BATT_ZI1-CLR, BATT_Z1+1))
    # the lid's two fixings, straight down into the thickened front wall
    for sy in (-1.0, 1.0):
        s = s.cut(cyl(M25_TAP, BATT_POST_L,
                      (BATT_XI1+BATT_FRONT_T/2, sy*BATT_SCREW_Y, BATT_ZI1-BATT_POST_L)))
    # BMS ribs.  The board stands with its TOP edge at the lid, not with its bottom on the
    # floor: the rear zone is BMS_W = 27 of a 43.6 mm interior either way, so one end of it
    # is dead space whichever way up it goes - and put at the top, the lid's own frame
    # lands on the board's top edge and traps it.  Each pair of ribs is a 1.8 mm slot for
    # its +x edge and a ledge for it to sit on; the rear wall takes the other edge.
    for py in (-BMS_RIB_Y, BMS_RIB_Y):
        s = s.union(bxc(BMS_X1+CLR, BMS_X1+CLR+0.8, py-5.0, py+5.0, BATT_ZI0, BMS_Z0))
        s = s.union(bxc(BMS_X0, BMS_X1, py-5.0, py+5.0, BMS_Z0-2.0, BMS_Z0))
    # ... and the y stops, one each side, floor to the board's top edge.  The measured
    # board does not reach the side walls the guessed one did - see BMS_SIDE_Y.
    for sy in (-1.0, 1.0):
        s = s.union(bxc(BMS_X0, BMS_X1,
                        sy*BMS_SIDE_Y, sy*BATT_YI, BATT_ZI0, BMS_Z0 + BMS_W))
    # grommet slot and vents, through the rear wall
    w, h = BATT_WIRE
    s = s.cut(bxc(BATT_X-BATT_L/2-1, BATT_XI0+1, -w/2, w/2, -h/2, h/2))
    for sy in (-1.0, 1.0):
        s = s.cut(cyl(BATT_VENT_D/2, BATT_CASE_T+2, (BATT_X-BATT_L/2-1, sy*24.0,
                                                     BATT_ZI1-6.0), axis=(1,0,0)))
    return s

def battery_lid():
    """The module's lid: a plate that sits on the case's walls, its rear edge in a groove
    in the rear wall and its front edge screwed down into the thickened front wall.  It
    goes in at an angle - rear tab into the groove first, then the front down.

    The only thing hanging below it is one bar across the BMS zone, which lands on the
    board's top edge and is what stops it lifting.  There is no lip anywhere else: over the
    brick the interior has BATT_FIT/2 = 0.2 mm, and a lip there would press on the cells.

    Off the torque path in every sense - it holds a cover on - so its two M2.5 form their
    own thread in the front wall.  That is a ONE-ASSEMBLY thread: a pack opened and closed
    repeatedly wants the holes drilled out and nutted, and the README says so."""
    # The rear edge runs BATT_GROOVE into the wall's groove; the front edge overhangs the
    # thickened front wall and is screwed to it.
    x0, x1 = BATT_XI0-BATT_GROOVE+CLR, BATT_X+BATT_L/2
    s = bxc(BATT_XI0, x1, -BATT_W/2, BATT_W/2, BATT_ZI1, BATT_Z1)
    # ... and the rear tab is only as wide as the groove, which is the interior: full
    # width here would put the lid's two rear corners inside the side walls.
    s = s.union(bxc(x0, BATT_XI0, -BATT_YI, BATT_YI, BATT_ZI1, BATT_Z1))
    # Locating lip: a FRAME round the interior's edge, CLR proud of it, not a slab over
    # the whole opening - a slab was 12 cm3 of plastic doing nothing but weigh 14 g, on a
    # robot whose flat trot moves on 11.  It starts clear of the groove; the plate itself
    # is what lands on the BMS's top edge and holds the board down.
    s = s.union(bxc(BMS_X0+CLR, BMS_X0+CLR+BATT_FRAME_W, -BATT_YI+CLR, BATT_YI-CLR,
                    BATT_ZI1-BATT_FRAME_D, BATT_ZI1))
    for sy in (-1.0, 1.0):                                    # ... into the front wall
        s = s.cut(cyl(M25_CLR, BATT_LID_T+2,
                      (BATT_XI1+BATT_FRONT_T/2, sy*BATT_SCREW_Y, BATT_ZI1-1.0)))
    return s

def cell_axes():
    """The six cell axes: (x0, x1, [(y, z), ...]) in robot coordinates.

    One place, because `cell_holder`, `cells_solid` and the wrapped brick's envelope all
    have to agree on where a cell actually is.  x0/x1 are the cell BODY - the terminals -
    not the brick: BATT_TAB of nickel and insulation stands off each end."""
    z0 = BATT_ZI0 + BATT_FIT/2 + BATT_WRAP + CELL_D/2
    return (BRICK_X0 + BATT_TAB, BRICK_X1 - BATT_TAB,
            [(sy*CELL_P, z0 + sz*CELL_P) for sz in (0, 1) for sy in (-1, 0, 1)])

def cell_holder(end=-1):
    """One of the two printed combs that hold the six cells while they are welded, and
    then stay in the pack inside the heatshrink.  `end` is -1 for the rear cap, +1 for the
    front; the two are identical and print as one part, qty 2.

    It is a union of six CH_WALL-thick collars on the cell pitch, CLIPPED to the array's
    own envelope and then bored.  The clip is the whole design: a frame around the outside
    of the array does not fit this robot - the module has 0.60 mm to the deck and its side
    walls are 0.5 mm off the deck screws' nut bosses at |y| = 35.2 - so there is no
    material outboard of any cell in either y or z, and the module grows by the separator
    gaps alone.  What is left at each outer face is a 7.9 mm flat where the collar has
    been cut back to the cell's tangent, and a 21.3 mm cell cannot leave through it.

    Assembly, which is the point of the part: stand one cap on the bench, drop the six
    cells into it vertically one at a time - each finds its own bore, so there is no
    six-at-once alignment - then push the second cap down onto the far ends.  Both
    terminal planes are then flush and open, and the nickel lies flat across them.

    Not structure, and not in fea.py: nothing on the robot loads it and it carries only
    the cells' own weight in shear against the case."""
    x0, x1, axes = cell_axes()
    xa = x0 if end < 0 else x1 - CH_LEN
    s = None
    for y, z in axes:
        c = cyl(CELL_D/2 + CH_WALL, CH_LEN, (xa, y, z), axis=(1, 0, 0))
        s = c if s is None else s.union(c)
    # clip flush with the outermost cells' tangent planes - this is what keeps the part
    # from costing the module anything but the pitch
    yc = CELL_P + CELL_D/2
    zs = [z for _, z in axes]
    s = s.intersect(bxc(xa, xa + CH_LEN, -yc, yc,
                        min(zs) - CELL_D/2, max(zs) + CELL_D/2))
    # ... and only now the bores.  Cut last, after every union that adds material near
    # them - the ordering rule chassis_bottom paid for.
    for y, z in axes:
        s = s.cut(cyl(CELL_D/2 + CH_FIT/2, CH_LEN + 2, (xa - 1, y, z), axis=(1, 0, 0)))
    # THE FOUR CORNER CELLS EACH ORPHAN AN ARC, AND NO CHECK IN THIS FILE SAW IT.
    # A corner cell is clipped on two sides at once, and the bore (CH_FIT/2 wider than the
    # clip radius) breaks the collar at BOTH flats - so the quadrant between them comes
    # away as a loose crescent.  The part was five solids, and `isValid()` and `Volume()
    # > 0` were both perfectly happy with it: this is the OCC-silence note again, in the
    # one form build() could not read.  Keep the body, drop the crescents, and check the
    # amount dropped is crescent-sized rather than something structural.
    solids = sorted(s.val().Solids(), key=lambda so: so.Volume(), reverse=True)
    orphan = sum(so.Volume() for so in solids[1:])
    # four crescents, ~79 mm3 each: CH_LEN long, CH_WALL thick, and as wide as the flat
    # the clip leaves.  Anything more than that, or a fifth piece, is a different defect.
    if len(solids) > 5 or orphan > 400.0:
        raise RuntimeError(f"cell_holder: {orphan:.0f} mm3 came away in "
                           f"{len(solids)-1} pieces - that is not the corner crescents")
    return W(solids[0])

def cell_holders():
    """Both caps as one solid - what the mass budget and both sim exporters carry, and
    what holder_clear() probes.  PARTS holds ONE cap at qty 2; this is the pair in place."""
    return cell_holder(-1).union(cell_holder(+1))

def cells_solid():
    """The six bare cells as one solid - a payload, like the brick and the BMS.  The
    holder has to clear these or it does not go on."""
    x0, x1, axes = cell_axes()
    s = None
    for y, z in axes:
        c = cyl(CELL_D/2, x1-x0, (x0, y, z), axis=(1, 0, 0))
        s = c if s is None else s.union(c)
    return s

def holder_clear():
    """mm3 the cell holder shares with (the case + lid) and with the six cells.

    Both are payload-class blindnesses, exactly like module_clear(): the cells are not
    parts, and the holder is a part that lives entirely inside another part's cavity, so
    a cap drawn 0.5 mm too wide would print, assemble in CAD and crush the pack without
    isValid() or interference() saying a word.  Returns (case_mm3, cells_mm3); both zero,
    and inf if a boolean failed - see overlap()."""
    h = cell_holders().val()
    box = battery_case().val().fuse(battery_lid().val())
    return tuple(overlap(h, w) for w in (box, cells_solid().val()))

def brick_com():
    """Centroid of the six wrapped cells - what BATTERY_KG hangs on in both sim exporters.
    It is NOT the module's centre: the BMS sits at the rear, so the brick is pushed
    2.75 mm forward of it.  Both exporters used to build this box from a literal
    (0, 0, BODY_Z0+3+BATT_H/2), which was the cradle's centre and is now nothing's."""
    return ((BRICK_X0+BRICK_X1)/2.0, 0.0, BATT_ZI0 + BATT_FIT/2.0 + BRICK_H/2.0)

def bms_com():
    """Centroid of the BMS, standing on edge against the module's rear wall, on the ribs
    that put its top edge at the lid."""
    return ((BMS_X0+BMS_X1)/2.0, 0.0, BMS_Z0 + BMS_W/2.0)

def brick_solid():
    """The six wrapped cells as one solid - a payload, not a part, but the thing the case
    exists to hold.  See module_clear()."""
    return bxc(BRICK_X0, BRICK_X1, -BRICK_W/2, BRICK_W/2,
               BATT_ZI0+BATT_FIT/2, BATT_ZI0+BATT_FIT/2+BRICK_H)

def bms_solid():
    """The BMS as a solid, on its ribs against the rear wall.  Also a payload."""
    return bxc(BMS_X0, BMS_X1, -BMS_L/2, BMS_L/2, BMS_Z0, BMS_Z0+BMS_W)

def module_clear():
    """mm3 of the two things INSIDE the battery module that are inside its printed walls.

    This is the foot bolt's lesson in a new place: the brick and the BMS are payloads, so
    `interference()` cannot see them, `isValid()` is happy either way, and the case can be
    drawn round a pack it would crush without one boolean in this file objecting.  It has
    already caught one - the lid was first drawn with a locating lip round the whole
    opening, 2 mm deep, into an interior the wrapped brick fills to 0.2 mm a side.

    Returns (brick_mm3, bms_mm3); both must be zero, and inf if a boolean failed."""
    box = battery_case().val().fuse(battery_lid().val())
    return tuple(overlap(box, w.val()) for w in (brick_solid(), bms_solid()))

def batt_clear():
    """Air between the battery module's lid and the deck's underside.

    The module is a printed part, so interference() covers it against the chassis - but
    the gap that decides the design is the one it does NOT share solid with, exactly like
    the IMU's used to be.  This is where a strip of foam goes; it may be small, never
    negative."""
    return BODY_Z1 - BATT_Z1

# =====================================================================================
# PART: chassis_top / lidar_mount
# =====================================================================================
def chassis_top():
    z0, z1 = BODY_Z1, BODY_Z1+DECK_T
    s = bxc(-BODY_L/2, BODY_L/2, -BODY_W/2, BODY_W/2, z0, z1)
    for x, ay, _ in DECK_SCREWS:
        for y in (-ay, ay):
            s = s.cut(cyl(M3_CLR, 20, (x, y, z0-1))).cut(cyl(3.2, 2.2, (x, y, z1-2.2)))
    # Orange Pi 5 Pro standoffs.  M2.5 through the board, through the standoff, into a nut
    # in a slot opening outboard in y - fitted before the board goes on, and still the only
    # face you can reach once the deck is on the tray.  The M2.5 nut, not M3: the board's
    # own holes are 2.5, and its 5.0 across-flats leaves 2.2 mm of standoff wall.
    # ... and the rear pair needs a deck to stand on before it gets one: see OPI_TAB_*.
    # The guard is the point - the tab exists only while the hole pattern hangs the
    # standoff off the end, so a re-measured OPI_HOLES deletes it with no edit here.
    if OPI_TAB_X < -BODY_L/2:
        for sy in (-1.0, 1.0):
            ty = sy*OPI_HOLES[1]/2
            s = s.union(bxc(OPI_TAB_X, -BODY_L/2, ty-OPI_TAB_W, ty+OPI_TAB_W, z0, z1))
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
    s = s.cut(bxc(IMU_WINDOW[0], IMU_WINDOW[1], -34, 34, z0-1, z1+1))
    s = s.cut(bxc(58, 60, -26, 26, z0-1, z1+1))
    # IMU standoffs, on the deck's TOP face on the centreline - see the IMU_* block.  The
    # board used to hang under two tabs in the window below; a cased battery needs that
    # 3.6 mm and the Orange Pi's standoff gap up here is 7 mm of air nothing else uses.
    # Component face DOWN into the gap under the board, so nothing it carries reaches the
    # Pi.  The window moved off the centreline (IMU_WINDOW) to leave solid deck here, and
    # the two M2.5 form their own thread IMU_TAP_L deep through the standoff and into that
    # deck - there is no nut, and the reason is in the IMU_* block.
    for sx in (-1.0, 1.0):
        hx = sx*IMU_HOLE_P/2
        s = s.union(cyl(IMU_STAND_R, IMU_STAND_H, (hx, IMU_Y, z1)))
        s = s.cut(cyl(M25_TAP, IMU_TAP_L, (hx, IMU_Y, z1+IMU_STAND_H-IMU_TAP_L)))
    for y in (-BODY_W/2, BODY_W/2-DECK_LIP_W):               # stiffening lips
        s = s.union(bxc(-BODY_L/2, BODY_L/2, y, y+DECK_LIP_W, z1, z1+DECK_LIP_H))
    for x, ay, _ in DECK_SCREWS:              # ... notched at every screw: a socket head
        foot = (x == GPS_X and ay == GPS_Y)   # at |y| = 41 stands 0.8 mm proud of its
        dz = DECK_LIP_H if foot else DECK_LIP_NOTCH       # counterbore and reaches into
        x0 = -BODY_L/2-1.0 if foot else x-DECK_LIP_CUT    # the lip.  It used to be the
        for sy in (-1.0, 1.0):                            # mid pair only; all four sit at
            y0 = sy*(BODY_W/2-DECK_LIP_W)                 # 41 now - and at the GPS mast's
            s = s.cut(bxc(x0, x+DECK_LIP_CUT,             # own pair it goes FULL DEPTH,
                          y0, y0+sy*DECK_LIP_W,           # out to the deck's rear end,
                          z1, z1+dz))                     # for GPS_PAD_R.  See DECK_LIP_*.
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
    s = bxc(IMU_X-L/2, IMU_X+L/2, IMU_Y-W_/2, IMU_Y+W_/2,
            IMU_Z0-T-IMU_STACK, IMU_Z0)
    # ... minus the two standoffs' footprints out of the COMPONENT layer only.  A board
    # has clear annuli round its mounting holes - that is what a mounting hole is - and
    # without this the envelope reports the deck's own standoffs as an interference with
    # the board that is bolted to them.  The PCB above them is untouched.
    for sx in (-1.0, 1.0):
        s = s.cut(cyl(IMU_STAND_R+CLR, IMU_STACK+0.2,
                      (sx*IMU_HOLE_P/2, IMU_Y, IMU_Z0-T-IMU_STACK-0.1)))
    return s

def imu_clear():
    """The IMU board against the Orange Pi above it.  Returns (overlap_mm3, gap_mm).

    It used to look DOWN, at the battery pack - that was the thin wall when the board hung
    in the deck window.  The board now sits on the deck's top face inside the Pi's own
    standoff gap, so the payload it can foul is the Pi, and neither is a part:
    interference() sees a printed solid and this sees the two boxes it cannot.

    Note this is deliberately measured against the Pi's BOARD (one OPI_STAND_H above the
    deck), not against OPI_BOX, whose floor is the deck itself.  The board shares the
    standoff gap with that envelope on purpose; OPI_BOX stays what it is for - the mass
    box, and the keep-out gps_mount's arms are shaped around."""
    z = BODY_Z1 + DECK_T + OPI_STAND_H
    pi = bxc(OPI_X-OPI_BOX[0]/2, OPI_X+OPI_BOX[0]/2,
             -OPI_BOX[1]/2, OPI_BOX[1]/2, z, z+OPI_BOX[2])
    return overlap(imu_module().val(), pi.val()), z - IMU_Z0

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

    THE FRONT WALL IS TIED AT BOTH ENDS, and until 2026-09-11 it was tied at neither:
    the board pocket ran the full length of the part, severed the wall from the skirt, and
    camera_mount came back as three bodies (the channel, plus the wall in two loose pieces
    of 529.3 and 104.9 mm3).  PART_SOLIDS said 3, so the count check passed on a wall that
    had fallen off.  The pocket now stops CLR short of the board's far end - 2.06 mm of
    end wall, which is also the board's stop in y - and the channel's top rises CAM_CAP
    over the pocket between the lens relief and the board's near end, where the board is
    inserted through and nothing may close across it.  The argument, the measurements and
    what else was tried are in the CAM_CAP block.

    IT PRINTS ON ITS BACK, on the skirt: that face is the part's one big flat, and stood
    up the right way the whole 90 mm channel is a 20 mm wall on a 4 mm foot."""
    L, Wd, T = CAM_BOARD
    o, n, _ = camera_frame()
    y0, y1 = cam_span()
    e0, e1 = min(CAM_END), max(CAM_END)
    ct, st = math.cos(math.radians(CAM_TILT)), math.sin(math.radians(CAM_TILT))
    top  = CAM_Z + Wd/2*ct                                        # the board's upper edge
    bot  = CAM_Z - Wd/2*ct                                        # ... and its lower one
    ptop = CAM_Z + (Wd/2+CLR)*ct + CLR*st          # ... and the pocket's own ceiling, at
    deck = BODY_Z1 + DECK_T                        # its front lip: the highest the cut
    s = bxc(CAM_BACK, CAM_FRONT, e0, e1, CAM_LEDGE, bot)          # below reaches
    s = s.union(bxc(CAM_BACK, CAM_FRONT, e0, e1, bot, deck))      # ... and the skirt, in
    s = s.union(bxc(CAM_BACK_HI, CAM_FRONT, e0, e1, deck, top))   # two steps past the deck
    # ... and the roof that ties the front wall's near half back to the skirt over the
    # board, from the lens relief to the board's own end.  See the CAM_CAP block.
    s = s.union(bxc(CAM_BACK_HI, CAM_FRONT, CAM_LENS_D/2+CAM_LENS_REL, y1,
                    top, ptop+CAM_CAP))
    # everything in front of the board below its upper half has to go: 0.5 mm to the fork
    s = s.cut(cam_box(-CLR, 40.0, e0-1, e1+1, -Wd, 1.0))
    # ... and so does everything in front of the lens, all the way across the holder
    s = s.cut(cam_box(-CLR, 40.0, -CAM_LENS_D/2-CAM_LENS_REL, CAM_LENS_D/2+CAM_LENS_REL,
                      -Wd, Wd))
    # the board's own pocket.  Open at the NEAR end, because that is the way the board
    # goes in; closed CLR short of its FAR end, which is both the board's stop in y and
    # what ties the long half of the front wall back to the skirt.
    s = s.cut(cam_box(-T-CLR, CLR, y0-CLR, e1+1, -Wd/2-CLR, Wd/2+CLR))
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
        a = a.cut(cyl(M3_CLR, 20, (HUB_BC/2*math.cos(th), HUB_BC/2*math.sin(th), HUB_TOP_Z-1)))
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
        # overlap() returns inf when the boolean fails, so a failed angle falls out of
        # `free` and the joint reads SMALLER.  It used to read -1.0 and land in `free`,
        # i.e. a boolean that never ran was exported as travel the leg does not have.
        if overlap(m.val(), static.val()) < 1.0: free.append(a)
    if not free: return (0, 0)
    best = cur = [free[0], free[0]]
    for a in free[1:]:
        if a - cur[1] <= step: cur[1] = a
        else:
            if cur[1]-cur[0] > best[1]-best[0]: best = cur
            cur = [a, a]
    if cur[1]-cur[0] > best[1]-best[0]: best = cur
    return tuple(best)

# The window each joint is swept over.  A free range that REACHES its window is not a
# mechanical limit, it is the end of the scan, and export_sim.py says so rather than
# exporting it as if the geometry had stopped the joint.
SCAN_WINDOW = {"hip_roll": 90, "hip_pitch": 150, "knee": 150}
ROM_STEP    = 10                 # degrees; the default sweep, see CLAUDE.md step 7

def rom_scan_all(step=ROM_STEP):
    """{joint: (lo_deg, hi_deg)} for the front-left leg, over real solids.

    THE ONE PLACE THE THREE SCANS ARE SPELLED OUT.  It was written twice - here and in
    export_sim.joint_rom() - and the two copies drifted: the exporter's hip_roll swung a
    bare hip_bracket() against chassis_bottom + cradle_front + servo_dummy alone, with no
    gps_mount, no camera_mount, no camera module, no thrust bolts and no fork screws, so
    --rom-step exported a wider roll joint than mini_dog.py printed.  Same class as the
    servo mass and the MJ_* constants: a number that exists twice is a number that will
    disagree with itself, so both callers come through here.

    gps_mount goes in MIRRORED: it stands over the rear pair of deck bosses and this scan
    swings the FRONT-left leg, so mirroring it forward is exactly the rear-leg scan
    against the real one - the legs are mirror images and the roll axis is x.
    The clamp screws go in on the STATIC side of every scan: they belong to the sleeve,
    which belongs to the proximal part, and it is the distal fork that sweeps over them.
    ... and the hub screw HEADS go on the MOVING side: they belong to the fork, and at
    the hip the passive arm's four sweep FORK_GAP from the gusset (see head_clear())."""
    if not PARTS:
        build()
    w = SCAN_WINDOW
    return {
        "hip_roll": rom_scan(hip_bracket().union(mv(fork_screws(), ROLL_LOC)),
                             PARTS["chassis_bottom"][0]
                                             .union(PARTS["cradle_front"][0])
                                             .union(mirX(gps_mount()))
                                             .union(camera_mount())
                                             .union(camera_module())
                                             .union(mv(servo_dummy(), ROLL_LOC))
                                             .union(mv(thrust_bolts(), ROLL_LOC)),
                             (ROLL_X, ROLL_Y, ROLL_Z), axis=(1,0,0),
                             lo=-w["hip_roll"], hi=w["hip_roll"], step=step),
        "hip_pitch": rom_scan(thigh().union(mv(fork_screws(), PITCH_LOC)),
                              hip_bracket().union(mv(servo_dummy(), PITCH_LOC))
                                           .union(mv(thrust_bolts(), PITCH_LOC))
                                           .union(mv(fork_screws(), ROLL_LOC)),
                              (PITCH_X, LEG_Y, PITCH_Z),
                              lo=-w["hip_pitch"], hi=w["hip_pitch"], step=step),
        "knee": rom_scan(shin().union(mv(fork_screws(), KNEE_LOC)),
                         thigh().union(mv(servo_dummy(), KNEE_LOC))
                                .union(mv(thrust_bolts(), KNEE_LOC))
                                .union(mv(fork_screws(), PITCH_LOC)),
                         (PITCH_X, LEG_Y, KNEE_Z),
                         lo=-w["knee"], hi=w["knee"], step=step),
    }

# The base link's printed parts, in one place because THREE files need the same list:
# interference() here, export_sim.py's URDF/MJCF, and ../ros2/.../generate_model.py.  It
# was three hand-copied tuples until the cradles were split out, which is the arrangement
# CLAUDE.md's "masses and densities live once" note exists to stop - a part missing from
# one of them is a robot that weighs different amounts in different simulators.
BASE_MESHES = ("chassis_bottom", "cradle_front", "cradle_rear", "chassis_top",
               "lidar_mount", "gps_mount", "camera_mount")
BODY_PARTS = BASE_MESHES + ("battery_case", "battery_lid")
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
            v = overlap(PARTS[a][0].val(), PARTS[b][0].val())
            if v > INTERF_TOL:
                bad.append((a, b, v))
    return bad

def panel_clear():
    """Does anything stand in front of the rear wall's openings?  Returns
    {name: blocked_mm3}, and 0.0 is the only passing value.

    Fourth member of the family that foot_bolt_check(), thrust_clear() and fork_access()
    belong to, and it is here because the defect it catches SHIPPED: every opening on this
    wall - the bus window and all three connectors - opened into the rear hip-roll
    cradles' 2.8 mm plate, 1.2 mm behind the wall.  Nothing could see it.  interference()
    pairs the static body parts and the cradles were part of chassis_bottom, so the plate
    was chassis_bottom standing in front of chassis_bottom's own hole; isValid() is happy
    with a pocket that leads nowhere, and rom_scan only looks at what moves.

    The probe is the opening extruded outward along -x, intersected with the whole
    assembled body.  A connector is not fitted from outside so the run only has to be as
    long as the mating half stands proud, but the bus window is a cable route and gets the
    full PANEL_REACH."""
    body = None
    for nm in BODY_PARTS:
        if nm not in PARTS:
            continue
        body = PARTS[nm][0] if body is None else body.union(PARTS[nm][0])
    xw = -BODY_L/2
    pw, ph = PANEL_WIN
    tgt = [("bus window", 0.0, 0.0, pw, ph)]
    for cy, cz, (w, h) in PANEL_AT:
        tgt.append((f"panel y{cy:+.0f} z{cz:+.0f}", cy, cz, w, h))
    by, bz = PANEL_BAL_AT
    tgt.append(("balance", by, bz, *PANEL_BAL))
    out = {}
    for name, cy, cz, w, h in tgt:
        probe = bxc(xw-PANEL_REACH, xw-CLR, cy-w/2, cy+w/2, cz-h/2, cz+h/2)
        out[name] = overlap(body.val(), probe.val())
    return out

def cradle_clear():
    """Can a driver reach the four screws that hold each cradle on?  Returns
    {(end, i): blocked_mm3}; 0.0 is the only passing value.

    The heads sit on seats inboard of the end wall and are driven along +x from inside the
    tray, which is open only with the battery module out - that is what fixes the assembly
    order, not a preference.  The probe is a DRIVER_D cylinder on the screw's real line
    running CRADLE_REACH back from the seat, against the tray itself: a seat a socket
    cannot get to is a screw nobody can turn, which is the failure this repository has now
    shipped three times."""
    tray = PARTS["chassis_bottom"][0]
    x0 = CRADLE_X - WALL - CRADLE_SEAT
    out = {}
    for end, sx in (("front", 1.0), ("rear", -1.0)):
        for i, (y, z) in enumerate(cradle_bolt_axes()):
            p = cyl(DRIVER_D/2, CRADLE_REACH, (sx*x0, y, z), axis=(-sx, 0.0, 0.0))
            out[(end, i)] = overlap(tray.val(), p.val())
    return out

def cradle_head_clear():
    """mm3 of the cradle screws' heads inside the battery module, plus the air left
    between the front seat's counterbore floor and the module's front face.  Both are
    invisible to interference(): a screw head is not a part.
    """
    heads = cradle_screws()
    box = None
    for nm in ("battery_case", "battery_lid"):
        b = PARTS[nm][0]
        box = b if box is None else box.union(b)
    v = overlap(heads.val(), box.val())
    seat = CRADLE_X - WALL - CRADLE_SEAT + CRADLE_CB          # counterbore floor
    # read the module's front face off the solid rather than re-deriving it: the case's
    # walls are not the same thickness front and back, and this is the wall that matters
    return v, seat - HUB_HEAD_H - PARTS["battery_case"][0].val().BoundingBox().xmax

def gps_clear():
    """mm3 of gps_mount inside the Orange Pi's envelope - the mast's own binding
    constraint, and the one thing about it isValid() and interference() cannot see: the
    Pi is a payload, not a part, so it exists in this model only as OPI_BOX."""
    cx, _, cz = opi_com()
    L, W, H = OPI_BOX
    box = bxc(cx-L/2, cx+L/2, -W/2, W/2, cz-H/2, cz+H/2)
    return overlap(PARTS["gps_mount"][0].val(), box.val())

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

def fork_access():
    """Can a key actually reach each fork arm's screws?  Returns {(joint, side):
    blocked_mm3}, and 0.0 is the only passing value (a boolean that fails returns -1, and
    that fails too - an unknown is not a pass).

    Third member of the same family as foot_bolt_check() and thrust_clear(), and it exists
    for the same reason: `isValid()` is happy with a screw nobody can turn, interference()
    only looks at the static body, and rom_scan only looks at what moves.  A fastener whose
    ACCESS is blocked is invisible to all three, and the robot has now shipped that defect
    twice - the foot bolt that opened inside the dome, and the clamp head inside the spine.

    The probe is a DRIVER_D cylinder on the screw's own axis, run KEY_REACH, intersected
    with the part the fork bolts ONTO.  That neighbour is the whole question: the fork is
    identical at all three joints, and what differs is what happens to be sitting behind
    it - and for the roll joint that is now cradle_front alone, because the fork goes on
    with the cradle in hand.  All six arms take a straight key.

    It has not always been six.  The roll joint's inboard arm faced the chassis, and two
    versions of a way round that came and went - four coaxial bores that reached one screw
    of four, then two tilted channels through the tray's corner.  Both are in the history
    above DRIVER_D.  What made this simple was not a better hole, it was the cradle
    becoming a separate part.  Keep the probe honest about that: it must end in open air,
    not inside a bore, which is what the first version got wrong and what the
    2026-09-03 bores cost."""
    out = {}
    # The roll joint's neighbour is cradle_front ALONE, and that is the whole point of the
    # split: the fork goes on with the cradle in hand, before its four screws hold it to
    # the tray, so the tray is not in the way at that moment.  Probed against the tray as
    # well it reads 569 mm3 - which is what it read for two years, and what the two @6
    # channels through the tray's corner existed to get around.
    for joint, loc, near in (("roll",  ROLL_LOC,  PARTS["cradle_front"][0]),
                             ("pitch", PITCH_LOC, PARTS["hip_bracket_A"][0]),
                             ("knee",  KNEE_LOC,  PARTS["thigh_A"][0])):
        for side, face, sgn in (("driven",  HUB_TOP_Z + ARM_T,   +1.0),
                                ("passive", ARM_BOT_TOP - ARM_T, -1.0)):
            lines = [((HUB_BC/2*math.cos(math.radians(90*i)),
                       HUB_BC/2*math.sin(math.radians(90*i)), face),
                      (0.0, 0.0, sgn), KEY_REACH) for i in range(HUB_N)]
            probe = None
            for p, ax, L in lines:
                c = cyl(DRIVER_D/2, L, p, axis=ax)
                probe = c if probe is None else probe.union(c)
            out[(joint, side)] = overlap(mv(probe, loc).val(), near.val())
    return out

def thrust_clear():
    """Furthest anything the thrust clamp adds gets from the joint axis, against the
    SPINE_R0 the distal fork sweeps.  Returns (r, margin); margin has to stay positive.

    Closed form, and deliberately so: no CadQuery, no sweep, no solids - so it costs
    nothing to print on every run and it stays readable next to the numbers it is made of.
    rom_scan sees the same thing through the boolean now that thrust_bolts() is in the
    static side of every scan, but a sweep at `step` degrees can step straight over a
    2 mm foul, and this cannot.

    Both terms matter and they move independently.  The LUG is set by THRUST_L and
    THRUST_YL and has always been fine.  The SCREW is set by THRUST_BOLT_L and the head,
    and it is the one that put the first assembled leg into its own spine."""
    x_out   = (-S_AX + CLR) - THRUST_BOLT_L - THRUST_HEAD_H
    r_screw = math.hypot(x_out, THRUST_Y + max(THRUST_HEAD_D, 3.0)/2)
    r_lug   = math.hypot(-S_AX - SLEEVE_W - THRUST_L, THRUST_YL)
    r = max(r_screw, r_lug)
    return r, SPINE_R0 - r

PARTS, REPORT = {}, {}
# How many separate bodies a part is ALLOWED to come back as.  Default 1, and the default
# is the point: a boolean that orphans a piece leaves a part that `isValid()` and
# `Volume() > 0` both pass and a slicer happily prints the debris of - cell_holder came
# back as five before its corner crescents were dealt with, and nothing here could see it.
# ONE ENTRY IS LEFT AND IT IS DELIBERATE.  Two others were here until 2026-09-11 and both
# were DEFECTS this table permitted rather than described - chassis_top at 3 (the deck
# plus the Orange Pi's rear standoff pair, two 383.8 mm3 bosses floating 0.2 mm off the
# deck's edge) and camera_mount at 3 (the channel plus the front retaining wall in two
# loose pieces of 529.3 and 104.9 mm3).  Both are geometry now: OPI_TAB_* puts a deck
# under the standoffs, CAM_CAP and the shortened board pocket tie the wall back on.  The
# lesson is in the number: a count check is only as good as the number a human typed, so
# raising an entry here is a claim about the design and has to be argued like one.
#   servo_gauge   2 - two test coupons on one plate, and this one is on purpose
PART_SOLIDS = {"servo_gauge": 2}
def build():
    hb, th, sh, ft = hip_bracket(), thigh(), shin(), foot()
    cf, cr = cradles()
    PARTS["chassis_bottom"] = (chassis_bottom(), 1, "PETG/ASA, 4 walls, 30% gyroid")
    PARTS["cradle_front"]   = (cf, 1, "PETG/ASA, 5 walls, 40% - flange face down")
    PARTS["cradle_rear"]    = (cr, 1, "PETG/ASA, 5 walls, 40% - flange face down")
    PARTS["chassis_top"]    = (chassis_top(),    1, "PETG/ASA, 4 walls, 25%")
    PARTS["lidar_mount"]    = (lidar_mount(),    1, "PETG/ASA, 4 walls, 30%")
    PARTS["gps_mount"]      = (gps_mount(),      1, "PETG/ASA, 4 walls, 30% - platform down")
    PARTS["camera_mount"]   = (camera_mount(),   1, "PETG/ASA, 4 walls, 40% - skirt down")
    PARTS["battery_case"]   = (battery_case(),   1, "PETG/ASA, 4 walls, 25% - open side up")
    PARTS["battery_lid"]    = (battery_lid(),    1, "PETG/ASA, 4 walls, 25% - flat")
    PARTS["cell_holder"]    = (cell_holder(),    2, "PETG/ASA, 3 walls, 30% - bores vertical, no support")
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
    a.add(PARTS["cradle_front"][0],   name="cradle_front",   color=grey)
    a.add(PARTS["cradle_rear"][0],    name="cradle_rear",    color=grey)
    a.add(PARTS["chassis_top"][0],    name="chassis_top",    color=grey)
    a.add(PARTS["lidar_mount"][0],    name="lidar_mount",    color=grey)
    a.add(PARTS["gps_mount"][0],      name="gps_mount",      color=grey)
    a.add(PARTS["camera_mount"][0],   name="camera_mount",   color=grey)
    a.add(PARTS["battery_case"][0],   name="battery_case",   color=dark)
    a.add(PARTS["battery_lid"][0],    name="battery_lid",    color=dark)
    a.add(cell_holders(),             name="cell_holder",    color=grey)
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
                # on the flange: 98 x 30.7 mm of flat, square to the two sleeve bores, and
                # it puts the layers across the joint's clamp rather than along it.
                "cradle_front": ((0,1,0),90), "cradle_rear": ((0,1,0),-90),
                # both flat as modelled: the case's open side is already up and the lid
                # is a plate.  Neither wants support.
                "battery_case": ((1,0,0),0), "battery_lid": ((1,0,0),0),
                # the cell holder on its face, bores vertical: a 10 mm plate with
                # six through holes, no overhang anywhere and nothing to support.
                "cell_holder": ((0,1,0),90),
                # base disc DOWN, which is what the part was drawn for and what
                # lidar_mount()'s docstring says.  It was 180 - upside down on the seat,
                # which has no flat at all: tools/orient_scan.py reads 4310 mm2 of
                # overhang and ZERO bed area that way against 2392 and 1656 mm2 this way.
                "lidar_mount": ((1,0,0),0),
                "hip_bracket_A": ((0,1,0),90),
                "hip_bracket_B": ((0,1,0),90), "thigh_A": ((1,0,0),90),
                "thigh_B": ((1,0,0),90), "shin_A": ((0,1,0),90), "shin_B": ((0,1,0),90),
                "foot": ((1,0,0),180), "servo_gauge": ((1,0,0),0),
                # upside down, on the platform's top face: it is the only flat on
                # the part (2080 mm2 against 72 on the two pads) and it turns a 40 x 52
                # unsupported ceiling into the bed itself.  See gps_mount's docstring.
                "gps_mount": ((1,0,0),180),
                # on its back skirt: the one big flat.  Stood up the right way this is a
                # 90 x 20 mm wall on a 4 mm foot.  90 + CAM_TILT lays that face on the bed.
                "camera_mount": ((0,1,0), 90.0+CAM_TILT)}

def main():
    for d in ("step", "stl"): os.makedirs(os.path.join(OUT, d), exist_ok=True)
    hb, th, sh, ft = build()
    rows = []
    print("\n  part                qty   volume   est.mass    print bbox (mm)")
    for name, (wp, qty, note) in PARTS.items():
        shp = wp.val()
        # Volume, not just isValid().  A boolean that fails on a degenerate contact - two
        # coincident faces, a tangency - comes back inverted rather than broken: OCC raises
        # nothing and isValid() still says True, and the part is then a sliver with negative
        # volume.  See chassis_bottom's ordering note for the one that shipped.
        # ... and ONE solid.  A part that comes back in pieces passes both of the checks
        # above - cell_holder did, as five - and a slicer will happily print the debris
        # next to the part.  See cell_holder()'s corner-crescent note.
        ok = (shp.isValid() and shp.Volume() > 0.0
              and len(shp.Solids()) == PART_SOLIDS.get(name, 1))
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
            v = overlap(psolid, PARTS[nm][0].val())
            if v > INTERF_TOL:
                bad.append((pname, nm, v))
    for na, nb, v in bad:
        print(f"  !! INTERFERENCE  {na} x {nb}  {v:.1f} mm3")
    if not bad:
        print(f"  body clear: {' / '.join(BODY_PARTS)} + camera + imu share no solid")
    # ... and the two payload gaps interference() cannot see at all: the IMU against the
    # Orange Pi it now shares a standoff gap with, and the battery module's lid against
    # the deck.  The module IS a part, so its solids are covered above; what is not is
    # the air over it, which is what the whole redesign was spent on.
    iv, igap = imu_clear()
    if iv > INTERF_TOL or iv < 0:
        print(f"  !! INTERFERENCE  imu x Orange Pi  {iv:.1f} mm3")
    else:
        print(f"  imu clear: {igap:+.2f} mm of air between the board and the Pi")
    bgap = batt_clear()
    if bgap < 0:
        print(f"  !! BATTERY  the module's lid is {-bgap:.2f} mm into the deck")
    else:
        print(f"  batt clear: {bgap:+.2f} mm of air between the module's lid and the deck")
    kv, mv_ = module_clear()
    if max(kv, mv_) > INTERF_TOL or min(kv, mv_) < 0:
        print(f"  !! BATTERY MODULE  case/lid into the pack: brick {kv:.1f} mm3,"
              f" BMS {mv_:.1f} mm3")
    else:
        print(f"  module clear: the case and lid touch neither the wrapped brick nor"
              f" the BMS ({kv:.1f} / {mv_:.1f} mm3)")
    hc, hcell = holder_clear()
    if max(hc, hcell) > INTERF_TOL or min(hc, hcell) < 0:
        print(f"  !! CELL HOLDER  into the case {hc:.1f} mm3, into the cells {hcell:.1f} mm3")
    else:
        print(f"  holder clear: both caps clear the case and all six cells"
              f" ({hc:.1f} / {hcell:.1f} mm3), {CELL_GAP:.1f} mm of air between cells")
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
    # The clamp screw against the fork spine it lives under.  This is the invariant the
    # first assembled leg broke: every printed solid passed, and the joint still bound,
    # because the thing sticking out was a screw and no screw was in the model.
    r, margin = thrust_clear()
    if margin < 0:
        print(f"  !! THRUST CLAMP  reaches r = {r:.2f}, inside the {SPINE_R0:.1f} the fork"
              f" spine sweeps - the joint binds on its own clamp screws")
    else:
        kind = "set screw" if THRUST_HEAD_H <= 0 else f"head @{THRUST_HEAD_D:.1f}"
        print(f"  clamp clear: M3 x {THRUST_BOLT_L:.0f} {kind} reaches r = {r:.2f} vs the"
              f" spine's {SPINE_R0:.1f} ({margin:+.2f} mm)")
    # The hub screws themselves: the passive head against the gusset, the tip against the
    # case.  Same kind of defect as the clamp screw - the part in the way is hardware.
    gap, sp, sd = head_clear()
    if min(gap, sp, sd) < 0:
        print(f"  !! HUB SCREWS  head gap {gap:+.2f}, tip spare passive {sp:+.2f} /"
              f" driven {sd:+.2f} mm - a negative one is a screw that fouls")
    else:
        print(f"  head clear:  ISO 7380 M3 x {HUB_SCREW_L:.0f} head {gap:+.2f} mm off the"
              f" gusset; tip {sp:+.2f} (passive) / {sd:+.2f} (driven) mm short of the case")
    # ... and whether a driver can reach the screws that hold the legs on at all.
    acc = fork_access()
    bad = {k: v for k, v in acc.items() if v > INTERF_TOL or v < 0}
    for (joint, side), v in bad.items():
        print(f"  !! FORK ACCESS  {joint}/{side} arm: {v:.0f} mm3 of the key's run to open"
              f" air is solid - those four screws cannot be fitted")
    if not bad:
        print(f"  fork access: all six arms open to a straight key within"
              f" {KEY_REACH:.0f} mm - the roll pair with the cradle in hand, before its"
              f" {len(CRADLE_BOLT)} screws go into the tray")
    pc = panel_clear()
    bad = {k: v for k, v in pc.items() if v > INTERF_TOL or v < 0}
    for k, v in bad.items():
        print(f"  !! PANEL  the {k} opening has {v:.0f} mm3 of solid in front of it -"
              f" nothing can be plugged in or routed out there")
    if not bad:
        print(f"  panel clear: bus window {PANEL_WIN[0]:.0f}x{PANEL_WIN[1]:.0f} and all"
              f" three connectors open to air over {PANEL_REACH:.0f} mm")
    hv, hgap = cradle_head_clear()
    if hv > INTERF_TOL or hv < 0:
        print(f"  !! CRADLE HEAD  {hv:.0f} mm3 of screw head inside the battery module -"
              f" counterbore CRADLE_CB is too shallow")
    cc = cradle_clear()
    bad = {k: v for k, v in cc.items() if v > INTERF_TOL or v < 0}
    for (end, i), v in bad.items():
        print(f"  !! CRADLE BOLT  {end} screw {i}: {v:.0f} mm3 of tray in the driver's"
              f" {CRADLE_REACH:.0f} mm run - that screw cannot be turned")
    if not bad:
        print(f"  cradle bolts: {len(CRADLE_BOLT)} x M3 x {CRADLE_BOLT_L:.0f} per end,"
              f" heads clear over {CRADLE_REACH:.0f} mm from inside the tray"
              f" (battery module out), {hgap:+.2f} mm of air to the pack")
    v = gps_clear()
    if v > INTERF_TOL or v < 0:
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
    carried = (N_SERVO*SERVO_KG + BATTERY_KG + BMS_KG + ELECTRONICS_KG + LIDAR_KG
               + GPS_KG + CAMERA_KG + IMU_KG)*1000.0
    print(f"\n  printed mass  ~{tm:.0f} g   + {N_SERVO} servos {N_SERVO*SERVO_KG*1000:.0f} g"
          f" + 3S2P cells ~{BATTERY_KG*1000:.0f} g + BMS ~{BMS_KG*1000:.0f} g"
          f" + Orange Pi/wiring ~{ELECTRONICS_KG*1000:.0f} g"
          f" + LiDAR ~{LIDAR_KG*1000:.0f} g + GPS ~{GPS_KG*1000:.0f} g"
          f" + camera ~{CAMERA_KG*1000:.0f} g + IMU ~{IMU_KG*1000:.0f} g"
          f"  ->  ~{(tm+carried)/1000:.2f} kg")
    print(f"  ROM scan (coarse, {ROM_STEP} deg steps, real solids):")
    rom = rom_scan_all()
    for k, v in rom.items():
        print(f"    {k:10s} free {v[0]:+4d} .. {v[1]:+4d} deg  (0 = leg straight down)")
    with open(os.path.join(OUT, "bom.json"), "w") as f:
        json.dump({"parts": rows, "rom_deg": rom,
                   "stance": {"wheelbase": 2*PITCH_X, "track": 2*LEG_Y,
                              "thigh": L_THIGH, "shin": L_SHIN}}, f, indent=2)

if __name__ == "__main__":
    main()
