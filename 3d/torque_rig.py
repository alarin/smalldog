"""
torque_rig.py - the printed fixture that measures the ST3215's torque in NEWTON-METRES.

    .venv/bin/python torque_rig.py       # -> out/torque/{step,stl} + the numbers

Why this exists, and why it is not another arm for `bench_rig.py`
----------------------------------------------------------------
Every torque this project has measured so far is a torque *per register count* -
per amp of PRESENT_CURRENT, or per unit of PRESENT_LOAD - and those two channels
now agree with each other to 0.3 % and disagree with the datasheet by 2.2x
(PLAN.md step 2c).  Friction survived that, because `tau_c` and `mu_load` are
RATIOS measured in the same register units and converted through a gravity anchor
that is known exactly.  `k_u` did not: the fit's implied stall was 4.23 N*m, the
friction-cancelled ladder said 7.4, the vendor said 2.94, and nothing on the
bench could tell which - `rl/` trained against a servo that peaked near 4.4 N*m
while `ros2/` clamped the same joint at 2.94, a 44 % disagreement about how
strong the robot is, decided by which sim you loaded.

**THAT QUESTION IS SETTLED.**  This rig answered it on 2026-09-10: the stall is
**4.50 N*m**, and `SERVO_STALL_NM` in `mini_dog.py` has been that number since,
which re-baselined `fea.py`'s stall column and all three sim arms (3d/CLAUDE.md
steps 4 and 6).  What follows is kept in the present tense because the rig still
has to be printed and run the same way for every following measurement - the
duty ladder still caps the push, the frame is still checked against DESIGN_NM -
but read the candidate list as history with one entry now confirmed.

A force reading breaks the tie, because a scale is not a register.  Torque is
then F x r, from a mass and a length, with no electrical parameter anywhere in
it.  And taken alongside the bus log from the same push it pins the two suspect
scalings as well: tau/i gives k_t in N*m/A, which is PRESENT_CURRENT's 6.5 mA LSB
- the number ST3215_STS3215_measured_parameters.md has always listed as
"confirmed only against vendor numbers, never against a shunt" - and tau/(d*U)
gives PRESENT_LOAD's per-mille scaling.  One evening settles k_u, k_t, R and the
gearbox efficiency together.

**The rig is SELF-REACTING, and that is the whole design.**  `bench_rig.py`'s
stand prints `!! IT WILL GO OVER unclamped` at 0.33 N*m; this test asks for up to
7.4, which is 22x that, and no clamp the bench owns should have to be trusted
with it.  So the scale sits in the frame's own throat: the arm presses DOWN on
the scale, the scale presses down on the jaw, the jaw carries it back through the
column into the sleeve, and the sleeve holds the servo case.  The loop closes
inside the part.  The bench feels the rig's weight and nothing else, so a reading
cannot be spoiled by the rig creeping, and the failure mode where a stand tips
mid-push with 4 kg of stored energy in it does not exist.

That is also why this is a separate frame rather than a third `bench_rig` arm.
Nothing here swings, `AXIS_H` = 190 is pure liability at this load, and the two
rigs want opposite things: that one needs air under the axis for a free swing,
this one needs the shortest possible path from the contact back to the case.

Like `bench_rig.py`, this file imports `mini_dog` ONE WAY - the sleeve, the
thrust clamp and the hub pattern, so the servo is held exactly as the robot holds
it - and `mini_dog` has never heard of it.  No `PARTS` entry, no mass in the
robot's budget, no `fea.py` or sim consumer, so `3d/CLAUDE.md` steps 4-6 do not
apply.  The moment anything here acquires a mass in the robot's budget it is a
robot part and they do.

IT DELIBERATELY NEVER STALLS, and that is not a compromise
----------------------------------------------------------
The scale on this bench reads 2 kg.  At `ARM_R` that is 3.3 N*m, which is above
the vendor's stall but below both of the other candidates - so a real stall test
is not merely unavailable, it would put 2.4 to 4.5 kg through a 2 kg instrument
and break it.  The rig is interlocked against that: see `duty_ladder()`, and cap
`TORQUE_LIMIT` before the first push.

None of which costs anything, because the quantity wanted was never the endpoint.
Against a blocked output omega = 0, so the back-EMF term vanishes and
tau = k_t * d * U / R exactly - a straight line through the origin.  `k_u` is its
SLOPE, and a slope measured over 0..3.3 N*m is the same slope as one measured at
the top.  Reading it at partial duty is also better conditioned than a single
stall point would have been: five points on a line, with a residual that says
whether the line is straight, against one number with nothing to check it.

What the arm length is for
--------------------------
`ARM_R` sets only how much torque fits under the scale's ceiling; it does not
change the structural demand, because the bending moment at the frame root is
F x r = tau whatever the radius is.  Longer is therefore better and the bed is
what stops it: at 170 mm the frame is 249 mm across a 256 mm bed, and 2 kg buys
3.3 N*m of range.
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import cadquery as cq                                                # noqa: E402
import mini_dog as md                                                # noqa: E402
from mini_dog import (HUB_TOP_Z, S_AX, S_W, SLEEVE_LEN,              # noqa: E402
                      SLEEVE_W, THRUST_L, bxc, cyl)
import bench_rig as br                                               # noqa: E402
from bench_rig import hub_bolts, hub_face, wedge                     # noqa: E402

OUT = os.path.join(HERE, "out", "torque")

# =====================================================================================
# parameters
# =====================================================================================
# Same frame as bench_rig: servo-local +Z is the (horizontal) axis and servo-local +X
# points straight DOWN, so the arm hangs at q = 0 and q = +90 deg lays it along +Y.
# The jaw is therefore at +Y and the column at -Y, out of the arm's way.
ARM_R      = 170.0      # axis to the anvil's contact line.  A SCALE-RANGE choice, not a
                        # structural one - and pushed as long as the bed allows, because
                        # every extra millimetre buys torque range under a fixed ceiling
SCALE_MAX_KG = 2.0      # the instrument on this bench.  **VERIFY against yours**: it
                        # sets the whole measuring range, and exceeding it does not give
                        # a bad reading, it gives a broken scale
SCALE_HEADROOM = 0.80   # never plan a push past this fraction of the scale
THROAT     = 60.0       # axis down to the jaw's top face.  Has to swallow the scale
                        # plus the anvil screw's adjustment; SCALE_H below is what
                        # it was sized for
SCALE_H    = 28.0       # MEASURED, to the platform, on the scale actually in use.  The
                        # one dimension here that belongs to a part nobody in this repo
                        # owns, so it is measured rather than assumed; if you swap the
                        # scale, re-measure and re-run - the fix for one that does not
                        # fit is to raise THROAT and reprint, never to shim, because a
                        # shim puts a soft joint in the load path
FRAME_Z    = SLEEVE_LEN/2                       # 16.5, exactly as bench_rig: the part
                        # prints flat on this face, one zmin, no support anywhere
JAW_T      = 14.0       # jaw plate thickness.  Deeper than bench_rig's 8 mm base
                        # because this one is a working beam and not a foot
JAW_Y1     = ARM_R + 26.0                     # out past the contact, so the scale has
                        # somewhere to sit on both sides of it
COL_Y1     = -S_W/2-SLEEVE_W                    # -15.36, the sleeve's -y face
COL_Y0     = COL_Y1-38.0                        # column depth in y - the bending flanges
COL_X0     = -S_AX-SLEEVE_W                     # flush with the sleeve's top face
COL_WALL   = 8.0        # flange left either side of a lightening window
COL_RIB    = 10.0
GUSSET     = 46.0       # column-to-jaw wedge, how far it runs OUT along the jaw
CASE_FLOOR = 4.0        # air kept under the servo case before the gusset starts
STRAP_Z    = 7.0        # each end strap that closes the holder's cable window - see
                        # torque_frame().  The gap left between the pair is
                        # SLEEVE_LEN - 2*STRAP_Z = 19 mm, which is the connector's road
                        # out; shrink it and the servo cannot be wired

#: How tall the gusset may rise before it meets the servo.  Derived, never typed:
#: the case's +x face is at S_L - S_AX, and the frame's clear space starts CASE_FLOOR
#: below it.
GUSSET_UP  = THROAT - (md.S_L - S_AX) - CASE_FLOOR

ARM_ROOT_W = 30.0       # beam width at the hub disc ...
ARM_TIP_W  = 24.0       # ... tapering to this at the anvil
ARM_T_Z    = 16.0       # arm thickness along the axis.  Thicker than bench_rig's arms:
                        # this one carries the full stall torque in bending, not 0.3 N*m
ANVIL_D    = 6.4        # M6 clearance.  The screw is the adjustment that takes up
                        # whatever is left between the arm at q = 90 and the scale's
                        # platform, and it is NOT threaded into the plastic: the whole
                        # measured force runs through it, so it lands in two M6 nuts,
                        # jammed against each other on the arm's top face.  That is the
                        # same rule the robot's own fasteners follow - 3d/CLAUDE.md,
                        # "Nothing in a torque path threads into plastic"
ANVIL_BOSS = 15.0       # local pad round the anvil hole, so the screw bears on more
                        # than the tapered beam's width

#: The torques worth telling apart, N*m.  Not decoration: they are what the rig is
#: sized around, and the report prints what each reads on the scale.
# MEASURED 2026-09-10: 4.50 N*m.  The rig did its job, so these three are history
# now - kept because duty_ladder() sizes the safe cap against the WORST of them, and
# a re-run on a servo that has not been measured wants that same conservatism.
CANDIDATES = {
    "vendor spec": 2.94,
    "fit_bam k_u": 4.23,
    "MEASURED, this rig": 4.50,
    "friction-cancelled ladder": 7.40,
}
DESIGN_NM  = 9.0        # what the frame and arm are checked against - past the highest
                        # candidate, because the point of the rig is that we do not yet
                        # know which candidate is right


# =====================================================================================
# parts
# =====================================================================================
def torque_frame():
    """sleeve + thrust clamp on a short column, with a jaw reaching out under the arm.

    A C, and the throat is the measurement: the scale goes in it, so the force the
    arm applies comes straight back through the jaw to the case it is pushing
    against.  Prints flat with the bore vertical, like bench_rig's stand, for the
    same reason - no support anywhere, and the column's layers lie in the plane it
    bends in, which at 9 N*m of design load is not a nicety.
    """
    s = md.sleeve(length=SLEEVE_LEN, window=True, lighten=True, clamp=True)

    # Fill the thrust lug out to the sleeve's full z so it lands on the bed rather
    # than printing on scaffold - bench_rig.bench_stand() explains this at length.
    # Only the two bands OUTSIDE the existing lug, or the union refills the nut
    # channels and the bolt clearances.
    for z0, z1 in ((-FRAME_Z, -md.THRUST_Z), (md.THRUST_Z, FRAME_Z)):
        s = s.union(bxc(-S_AX-SLEEVE_W-THRUST_L, -S_AX-SLEEVE_W+2.0,
                        -md.THRUST_YL, md.THRUST_YL, z0, z1))

    # CLOSE THE HOLDER'S BOTTOM.  mini_dog's sleeve cuts a cable window through the
    # +x wall over the sleeve's whole length, and +x points DOWN here - so on the
    # robot that window faces sideways and is fine, while on this rig it is a
    # CONN_W-wide slot in the floor with the servo sitting over it.  The ring is
    # therefore open exactly where this rig loads it hardest: everything the arm
    # applies arrives at the case as a torque about the axis, and an open ring
    # carries torsion through two C-shaped walls instead of a closed tube.
    #
    # Two straps, at the z ENDS, restoring the wall over STRAP_Z of the window at
    # each end and leaving the middle clear - the connector is the fat part of the
    # cable and it escapes through the gap between them.  Adding material, never
    # cutting: the same rule as the thrust clamp, and for the same reason
    # (3d/CLAUDE.md, "Do not slit the sleeve" - the clamp that survives is the one
    # that adds material).  Closing the whole window would strand the connector
    # inside the sleeve.
    for sg in (1, -1):
        s = s.union(bxc(md.S_L-S_AX-1, md.S_L-S_AX+SLEEVE_W,
                        -md.CONN_W/2, md.CONN_W/2,
                        sg*(FRAME_Z-STRAP_Z), sg*FRAME_Z))
    # Re-cut the case AFTER the straps, not before.  The window was cut from 1 mm
    # inside the case outward, so a strap that simply fills it back in reaches 1 mm
    # into the cavity and the servo no longer goes in - invisible to isValid(), and
    # the kind of thing that is only found with the part in one hand and the servo
    # in the other.  Adding material and then re-cutting the envelope is the same
    # ordering rule the rest of this repo follows for through-paths.
    s = s.cut(md.servo_case(md.CLR))

    # column: sleeve's -y flank down to the jaw
    s = s.union(bxc(COL_X0, THROAT+JAW_T, COL_Y0, COL_Y1, -FRAME_Z, FRAME_Z))

    # lightening windows through z, so they print as plain vertical holes.  What is
    # left is an I: the two y-flanges take the bending, the ribs take the shear.
    x, ys, ye = COL_RIB, COL_Y0+COL_WALL, COL_Y1-COL_WALL
    while x + 22.0 < THROAT-COL_RIB:
        w = min(30.0, THROAT-COL_RIB-x)
        s = s.cut(bxc(x, x+w, ys, ye, -FRAME_Z-1, FRAME_Z+1))
        x += w + COL_RIB

    # the jaw, and a gusset into the column.  The gusset is why the jaw can be 14 mm.
    s = s.union(bxc(THROAT, THROAT+JAW_T, COL_Y0, JAW_Y1, -FRAME_Z, FRAME_Z))
    # The column-to-jaw gusset, and its height is NOT free.  The corner wants as
    # deep a brace as it can get, but the servo's own case hangs into this quadrant:
    # the case's +x face is the floor of the space, so a wedge that rises past it
    # cuts straight through the servo.  A 46 mm brace did exactly that - 5.8 cm3 of
    # frame inside the case envelope, invisible to isValid() and to every other
    # check here, because the servo is not a part.  GUSSET_UP is what is left below
    # it; if THROAT ever shrinks, this goes to zero before the throat does.
    s = s.union(wedge([(THROAT, COL_Y1), (THROAT, COL_Y1+GUSSET),
                       (THROAT-GUSSET_UP, COL_Y1)], -FRAME_Z, FRAME_Z))
    return s


def torque_arm():
    """the horn the scale reads, with an adjustable anvil at ARM_R.

    Points at servo-local +X like bench_rig's arms, so it hangs at q = 0 and lies
    along +Y - over the jaw - at q = +90.  That is the only attitude this rig is
    used in, and the scale is what stops it going further: released, gravity pulls
    the arm from +90 back toward 0 and the anvil is already resting on the platform,
    so there is nowhere to fall.  Put the arm on the scale BEFORE enabling torque
    and there is no drop in the procedure at all.
    """
    z0, z1 = HUB_TOP_Z, HUB_TOP_Z+ARM_T_Z
    a = hub_face(z0, z1)
    x0, x1 = ARM_R-ANVIL_BOSS, ARM_R+ANVIL_BOSS
    a = a.union(wedge([(0.0, -ARM_ROOT_W/2), (x0, -ARM_TIP_W/2), (x1, -ARM_TIP_W/2),
                       (x1, ARM_TIP_W/2), (x0, ARM_TIP_W/2), (0.0, ARM_ROOT_W/2)],
                      z0, z1))
    a = a.union(bxc(x0, x1, -ANVIL_BOSS, ANVIL_BOSS, z0, z1))   # pad round the screw
    a = hub_bolts(a, z0)                                        # after the union, always
    a = a.cut(cyl(ANVIL_D/2, ARM_T_Z+2, (ARM_R, 0.0, z0-1)))
    return a


PARTS = {}


def build():
    PARTS["torque_frame"] = (torque_frame(), 1,
                            "PETG, 5 walls, 40% gyroid - flat on -z, no support")
    PARTS["torque_arm"] = (torque_arm(), 1, "PETG, 5 walls, 50% - it carries the torque")
    ok = True
    for name, (wp, qty, note) in PARTS.items():
        sh = wp.val()
        good = sh.isValid() and sh.Volume() > 0
        ok &= good
        bb = sh.BoundingBox()
        print(f"  {'ok  ' if good else '!! INVALID'} {name:<12} "
              f"{sh.Volume()/1000:7.1f} cm3   bbox "
              f"{bb.xlen:6.1f} x {bb.ylen:6.1f} x {bb.zlen:6.1f} mm   {note}")
        if max(bb.xlen, bb.ylen) > br.BED[0] or min(bb.xlen, bb.ylen) > br.BED[1]:
            print(f"       !! does not fit the {br.BED[0]:g} mm bed")
            ok = False
    return ok


# =====================================================================================
# checks
# =====================================================================================
def throat_clear():
    """Does the scale fit, does the arm clear the frame, and how long is the screw?

    Three questions, and the middle one is the only one `interference()` could ever
    answer.  The scale is not a part - exactly as the Orange Pi is not one in
    mini_dog.py - so nothing else here can see a throat that cannot be loaded.

    The screw length is the part that was wrong the first time.  The gap from the
    platform up to the AXIS is not what the screw spans: the arm is in the way, so
    the screw only bridges from the arm's underside down.  Reporting the axis gap
    flattered a 45 mm scale with 15 mm of adjustment where the real figure was 3.

    And it was wrong a SECOND time, the same way: the underside was taken as
    ARM_TIP_W/2 = 12, which is the beam, but the widest thing at the anvil is the
    ANVIL_BOSS pad at 15 - and after the q=+90 rotation that direction is "down".
    So read the underside off the ROTATED SOLID inside a slab at the anvil, the way
    cradle_head_clear() reads the battery module's front face, rather than off
    whichever constant looks like the right one.  It costs 3 mm, i.e. M6 x 35.
    """
    arm = PARTS["torque_arm"][0].rotate((0, 0, 0), (0, 0, 1), 90.0).val()
    frame = PARTS["torque_frame"][0].val()
    shared = arm.intersect(frame).Volume()
    platform = THROAT - SCALE_H            # x of the scale's top face, +x being down
    # the arm's lowest material AT THE ANVIL, measured: local +X points along +Y after
    # the rotation, so slab the solid about the anvil's own line and take its xmax.
    slab = bxc(-THROAT, THROAT, ARM_R-ANVIL_BOSS, ARM_R+ANVIL_BOSS, -FRAME_Z-50, FRAME_Z+50)
    under = arm.intersect(slab.val()).BoundingBox().xmax
    reach = platform - under               # arm's underside to the platform
    screw = int(round((reach + 18) / 5.0)) * 5
    print(f"  throat:     {THROAT:.0f} mm axis to jaw, {SCALE_H:.0f} mm of scale "
          f"-> platform sits {platform:.0f} mm below the axis")
    print(f"  anvil:      {reach:.0f} mm from the arm's underside ({under:.0f} mm below "
          f"the axis) to the platform -> M6 x {screw:d}, two nuts")
    print(f"  arm at q=90:{shared:9.1f} mm3 shared with the frame "
          f"({'clear' if shared < 1.0 else '!! FOULS'})")
    if reach < 4.0:
        print("       !! no room for the screw - raise THROAT and reprint")
    if platform < 2.0:
        print("       !! the scale is taller than the throat")
    return shared < 1.0 and reach >= 4.0 and platform >= 2.0


def holder_closed():
    """Is the cable window actually shut at the ends, and can the servo still go in?

    Two failure modes that no other check here can see, because the servo is not a
    part.  A strap that misses leaves the ring open and nothing says so; a strap
    that overreaches fills the case cavity and nothing says that either - isValid()
    is happy with both.  So probe the real solid: a thin box in the window's mouth
    inside each strap band must be FULL, and the servo's own envelope must be
    EMPTY.
    """
    f = PARTS["torque_frame"][0].val()
    x0, x1 = md.S_L-S_AX, md.S_L-S_AX+SLEEVE_W
    band = bxc(x0+0.5, x1-0.5, -md.CONN_W/2+1, md.CONN_W/2-1,
               FRAME_Z-STRAP_Z+0.5, FRAME_Z-0.5).val()
    want = band.Volume()
    got = f.intersect(band).Volume()
    shut = got > 0.98*want
    case = f.intersect(md.servo_case(md.CLR).val()).Volume()
    gap = SLEEVE_LEN - 2*STRAP_Z
    print(f"  holder floor: strap fills {got/want:5.1%} of the window's mouth "
          f"({'shut' if shut else '!! STILL OPEN'}), "
          f"{gap:.0f} mm left for the connector")
    print(f"  case cavity:  {case:7.2f} mm3 of frame inside the servo's envelope "
          f"({'clear' if case < 1.0 else '!! THE SERVO WILL NOT FIT'})")
    return shut and case < 1.0 and gap > 12.0


def strength():
    """Section modulus at the two roots, against DESIGN_NM.

    Not a substitute for fea.py - it sees no stress concentration and no interlayer
    plane - but both members here are prismatic beams in pure bending, which is the
    one case a hand section is honest about.  fea.py does not cover this part and
    should not: it is not a robot part.
    """
    out = []
    for name, b, d in (("arm at the hub", ARM_T_Z, ARM_ROOT_W),
                       ("frame column", 2*FRAME_Z, COL_Y1-COL_Y0)):
        Z = b * d * d / 6.0
        out.append((name, Z, DESIGN_NM * 1000.0 / Z))
        print(f"  {name:<16} Z = {Z:7.0f} mm3 -> {out[-1][2]:5.1f} MPa at "
              f"{DESIGN_NM:g} N*m")
    return out


def duty_ladder():
    """What the scale reads, and the duty each candidate reaches its ceiling at.

    This is the interlock.  The scale is 2 kg and three of the four candidates put
    more than that through it at full duty - including the 4.50 N*m this rig went
    on to measure - so the run has to be capped BEFORE the first push and the cap
    has to be set for the WORST case: a duty chosen for the 2.94 N*m candidate
    would have broken the instrument on push one.
    """
    r = ARM_R / 1000.0
    cap_kg = SCALE_MAX_KG * SCALE_HEADROOM
    cap_nm = cap_kg * 9.80665 * r
    print(f"  the scale is {SCALE_MAX_KG:g} kg; planning to {SCALE_HEADROOM:.0%} of it "
          f"= {cap_kg:.2f} kg = {cap_nm:.2f} N*m at {ARM_R:g} mm\n")
    print(f"  {'candidate':<28}{'torque':>9}{'at full duty':>14}{'safe duty':>11}")
    worst = 1.0
    for name, tau in CANDIDATES.items():
        kg = tau / r / 9.80665
        d = min(1.0, cap_nm / tau)
        worst = min(worst, d)
        flag = "" if kg <= SCALE_MAX_KG else "   << would break it"
        print(f"  {name:<28}{tau:8.2f} N*m{kg:11.2f} kg{d:10.0%}{flag}")
    print(f"\n  SET TORQUE_LIMIT <= {int(worst*1000)//50*50} (of 1000) before the first "
          f"push.\n  That is the cap for the WORST candidate, which is the only safe one "
          f"to pick\n  while which candidate is true is the open question.")
    print(f"  Measurable range: 0 .. {cap_nm:.2f} N*m, which is "
          f"{cap_nm/CANDIDATES['vendor spec']:.0%} of the vendor stall. The fit needs a "
          f"SLOPE,\n  not an endpoint, so that is enough - see the docstring.")


def protocol():
    print("""
  the run, once it is printed
  ---------------------------
  The frame needs no clamp - the loop closes inside the part - but the scale goes
  in the throat and the anvil onto its platform BEFORE torque is enabled.

  0. CAP THE SERVO FIRST.  Set TORQUE_LIMIT to the value duty_ladder() printed and
     read it back.  It is the only thing standing between a 2 kg scale and a servo
     that might be a 7.4 N*m one.  Do this before step 1, not after.
  1. Torque off, arm laid at q = +90 with the anvil resting on the platform.  Wind
     the anvil screw down until it just touches and jam the second nut.  TARE, then
     read: that is the arm's own weight through the contact, and it comes off every
     later reading.
  2. `robot/bench/torque_hold.py` against this contact - NOT
     `sweep.py --traj stall`.  This was learned the expensive way on 2026-09-10.
     traj_stall's 0.8 s bursts are right for the ELECTRICAL fit and wrong for a
     scale: the arm's inertia and the scale's own filter ringing add a roughly
     CONSTANT offset - measured at ~130 g, which is 920 g against a held 780 at
     the same rung - so it corrupts the slope as well as the level.  Hold for 8 s
     and read it settled.  The thermal argument that justified bursts does not
     apply at a capped duty: at TORQUE_LIMIT 350 the motor sees 1.5 W, not the
     ~32 W of an uncapped locked rotor.
     Note the torque DECAYS 4-7 % over an 8 s hold - the winding warms, R rises,
     and at fixed duty the current falls with it.  Record the COLD value: that is
     what a leg gets in a transient.
  3. Walk TORQUE_LIMIT up in steps, reading the scale at the peak of each burst and
     noting which burst it was.  STOP at the planned ceiling even if the scale has
     more printed on it.  Duty and current come off the bus in the same csv, so
     each reading is one (force, duty, current) triple.
  4. Repeat at 12 / 10 / 8 V.  Torque per duty*volt and torque per amp must both be
     voltage-invariant.  If they are not, the contact is moving.

  what it settles
  ---------------
  Fit a line through (d*U, tau).  Its SLOPE is k_u in N*m per volt with no register
  scaling anywhere in it, so k_u*12 is the stall - and therefore whether rl/ or
  ros2/ has the right servo, which they currently disagree about by 44 %.  The
  slope of (i, tau) is k_t in N*m/A, which pins PRESENT_CURRENT's 6.5 mA LSB - a
  number never checked against anything but the datasheet.  Together they close
  PLAN.md step 2c.  The INTERCEPT is not waste either: at a blocked output it is
  the friction the gearbox eats before anything reaches the arm, which is a third
  independent read on tau_c.

  four ways to get a wrong number
  -------------------------------
  * the arm not square to the platform.  tau = F*r only if the force is
    perpendicular; q = +90 off the servo's own encoder is what makes it square,
    which is why the anvil is a screw and not a printed boss - the screw takes up
    the height, the encoder sets the angle.
  * a soft scale, or one that is not FLAT ON ITS FEET.  A kitchen scale weighs the
    load between its platform and its feet, and the jaw here is 33 mm wide against
    a scale 120-190 mm across - so it bridges a beam under its middle with its feet
    in air, rocks, and reads several percent off.  Put a rigid plate across the jaw,
    wide enough to carry the feet.  Do NOT solve it by moving the scale to the bench
    and clamping the frame: that hands the reaction to the clamp, the stand rotates,
    and it leans its own weight onto the scale - +70 g at every rung, measured.
    The C-frame is self-reacting precisely so that no clamp is in the load path.
  * trusting the encoder to tell you the fixture is solid.  IT CANNOT.  It reads the
    output shaft relative to the servo's own CASE, so a case turning in the sleeve or
    a whole frame rotating is invisible to it.  Zero encoder creep is necessary and
    nowhere near sufficient.
  * the servo's own protection.  PROTECTION_CURRENT 310 and OVERLOAD_TORQUE 80 sit
    under this test as well as the TORQUE_LIMIT you set.  If protection trips first
    you have measured the protection - check the duty in the csv actually reaches
    the cap.
  * reading the ceiling as the stall.  This rig stops at a fraction of full duty by
    design.  The stall is the slope extrapolated to d = 1, never a number the scale
    was shown.""")

def main():
    print(f"\ntorque_rig.py - torque in newton-metres, {ARM_R:g} mm arm\n")
    ok = build()
    print()
    ok &= throat_clear()
    ok &= holder_closed()
    print()
    strength()
    duty_ladder()
    protocol()
    os.makedirs(os.path.join(OUT, "step"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "stl"), exist_ok=True)
    for name, (wp, qty, note) in PARTS.items():
        cq.exporters.export(wp, os.path.join(OUT, "step", f"{name}.step"))
        cq.exporters.export(wp, os.path.join(OUT, "stl", f"{name}.stl"))
    print(f"\n  -> {os.path.relpath(OUT)}/{{step,stl}}")
    print("  PASS" if ok else "  !! FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
