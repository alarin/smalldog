#!/usr/bin/env python3
"""bench_rig.py - the printed fixture `robot/bench/sweep.py` needs, and nothing else.

    .venv/bin/python bench_rig.py          # -> out/bench/{step,stl}, the numbers below

`robot/README.md` says "Build it first. The parts list is in the project notes" and
then never gives one, so this is it: a stand that holds one ST3215 with its axis
horizontal, and two arms that bolt to its driven hub.  The bench needs exactly three
things of the fixture and this file exists to give it them:

  * the case held rigidly, with no play.  The ST3215 has no threaded side holes, so
    the only interface that exists is `mini_dog.sleeve()` and its two M3 thrust bolts
    - the same one the robot uses.  It is imported, not re-cut: a bench that holds the
    servo differently from the robot fits a servo the robot does not have.
  * the axis HORIZONTAL, and enough air under it.  Every trajectory in sweep.py is a
    gravity experiment - `fit_bam.py` regresses measured current against m*g*r*sin(q) -
    so q = 0 has to be the hanging position and the arm has to be able to fall through
    it.  That is what sets AXIS_H, and it is the only reason this stand is tall.
  * `J_arm` KNOWN, because the free swing reports `J_m = J_total - J_arm`.  It is a
    difference of two numbers, so the arm's own inertia is measured off the real solid
    here and printed below, not estimated at the bench.

The one idea the whole layout rests on
--------------------------------------
The arm lives in the 4 mm slab z = 20.3 .. 24.3 - the driven hub's outer face plus
ARM_T - and every part of the stand stays at z <= SLEEVE_LEN/2 = 16.5.  The two never
share a z, so the arm sweeps a FULL CIRCLE and the stand can have any shape it likes
below it.  3.8 mm of axial gap is what buys that; it is the number to check first if
either the sleeve length or the hub offset ever moves.

Why the arms are single-shear horns, which the robot forbids
------------------------------------------------------------
`3d/CLAUDE.md` is emphatic that nothing may load the servo through a single-shear horn.
That rule is about the gait: 12 joints, ~36 N landing reactions, reversing at every
step, for the life of the robot.  Here one static weight hangs on one arm for about
twenty minutes.  Bolting to the driven hub alone costs a factor of ~2 in the bolt
pattern and buys the full-circle swing above - `fork()` spans z = -21.9 .. 24.3 and
would collide with the stand at exactly the angle the free swing decays through.  It
also keeps the light arm light, which is the whole point of the light arm.  Sized
anyway: 0.35 kg at 90 mm is 0.31 N*m into the ARM_R disc, ~21 N per bolt pair on the
@14 circle, and 0.7 MPa of in-plane bending in the beam.

Not part of the robot: nothing here is in `mini_dog.PARTS`, nothing here has a mass in
the robot's budget, and `fea.py` / `export_sim.py` / the ROS 2 generator never see it.
It imports `mini_dog` one way only.
"""
from __future__ import annotations

import math
import os
import sys

import cadquery as cq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import mini_dog as md                                                # noqa: E402
from mini_dog import (ARM_R, ARM_T, HUB_BC, HUB_N, HUB_TOP_Z, M25_CLR, S_AX, S_L, S_W,
                      SLEEVE_LEN, SLEEVE_W, THRUST_L, bxc, cyl)      # noqa: E402
from export_sim import MP                                            # noqa: E402

OUT = os.path.join(HERE, "out", "bench")

# =====================================================================================
# parameters
# =====================================================================================
# In use: servo-local +Z is the (horizontal) axis, servo-local +X points straight DOWN.
# So the case hangs under the axis, the thrust lug and its two M3 jack screws end up on
# top where a hex key reaches them, and the cable window at x = +34 faces the floor.
AXIS_H     = 150.0      # axis to the base's underside.  > ARM_L_R + the tip hardware.
BASE_T     = 8.0
BASE_Y0    = -78.0      # behind the column ...
BASE_Y1    = +58.0      # ... and out under the arm: this half is what stops it tipping
BASE_Z     = SLEEVE_LEN/2                       # 16.5 - never more, see the docstring
CLAMP_D    = 7.0        # two slots for an M6 or a G-clamp.  It needs them: see below.
CLAMP_RUN  = 14.0      # obround, not open-ended: CLAMP_RUN/2 + CLAMP_D/2 has to
                        # stay inside BASE_Z or the slot severs the base into strips

COL_Y1     = -S_W/2-SLEEVE_W                    # -15.36, the sleeve's -y face
COL_Y0     = COL_Y1-34.0                        # 34 mm of depth = the bending flanges
COL_X0     = -S_AX-SLEEVE_W                     # flush with the sleeve's top face
COL_Z      = BASE_Z     # the column, the gussets and the base are all exactly as thick
                        # in z as the sleeve, and that is not styling: the part prints
                        # flat on this face, so anything thinner floats above the bed and
                        # buys itself support under its whole length.  One zmin, no
                        # support anywhere - which is also why the windows below go
                        # through z rather than through y.
COL_HEAD_X = 42.0       # head tie: sleeve flank to column, over this much of the case
COL_WALL   = 7.0        # flange left either side of a lightening window
COL_RIB    = 9.0

ARM_S_R    = 45.0       # light arm: the direct J_m measurement.  Small on purpose -
                        # J_m = J_total - J_arm is a difference, so J_arm wants to be a
                        # few % of it.  Not smaller, though: fit_bam.py needs
                        # m*g*r*sin(q0) to beat tau_c before it can read the release at
                        # all, and tau_c is itself a guess (0.05 N*m), so this is sized
                        # for ~2x that and no more.
ARM_L_R    = 90.0       # heavy arm: the holds, where a big torque is exactly what is
                        # wanted.  0.35 kg here is 0.31 N*m, ~11 % of the 2.94 stall.
ARM_S_BOLT = 5.30       # M5 clearance at the light tip
ARM_L_BOLT = 8.40       # M8 clearance at the heavy tip
TIP_W      = 22.0       # tip pad width; the weight bolts flat to the +z face
TIP_L      = 20.0
TIP_R      = 16.0       # how far the tip hardware may reach past the bolt axis
ROOT_W     = 26.0       # beam width at the hub disc, tapering to TIP_W at the tip

# what to hang on each arm.  Not geometry - the numbers to type into sweep.py, and the
# reason the tips are sized the way they are.  Both are WEIGHED, not assumed.
ARM_MASS   = {"bench_arm_s": 0.25, "bench_arm_l": 0.35}

# Fill factor = printed mass / solid-volume mass, MEASURED the way mini_dog's PRINT_FILL
# is: sliced alone in OrcaSlicer for the Qidi Q2 (0.2 layer, 4 walls, 30 % gyroid, no
# support) and read off the plate.  Not a detail - mini_dog's own default of 0.92 is for
# thin-walled parts where the walls set the mass, and the stand is the opposite: a chunky
# part where the infill does, so 0.92 overstated it by 100 g.
BENCH_FILL = {"bench_stand": 0.54, "bench_arm_s": 0.72, "bench_arm_l": 0.70}

G = 9.80665
TAU_C_EST  = 0.05       # rl/params/st3215.json's Coulomb friction prior, the number
                        # the free swing's driving torque has to beat.  A prior, not a
                        # measurement - which is exactly why the arms carry margin on it.


# =====================================================================================
# helpers
# =====================================================================================
def rho(name):
    """kg/mm3 for a bench part.  Its own table, not mini_dog's: nothing here is in the
    robot's mass budget and nothing here should ever appear in it."""
    return md.PETG_RHO * BENCH_FILL.get(name, md.PRINT_FILL_MEAN) * 1e-6


def wedge(pts, z0, z1):
    """a polygon in the local XY plane, extruded through z - the whole stand is these"""
    return cq.Workplane("XY", origin=(0, 0, z0)).polyline(pts).close().extrude(z1-z0)


def hub_face(z0, z1):
    """the driven hub's bolt pattern, exactly as fork() cuts it: @6.4 over the central
    output-shaft screw so a driver still reaches it, and 4 x M2.5 clearance on the @14
    circle.  The screws are M2.5 x 6 and thread into the stock aluminium - see the note
    in fork(); a longer one bottoms out on the case and the vendor FAQ says that burns
    servos."""
    d = cyl(ARM_R, z1-z0, (0, 0, z0))
    d = d.cut(cyl(3.2, 40, (0, 0, z0-10)))
    for i in range(HUB_N):
        th = math.radians(90*i)
        d = d.cut(cyl(M25_CLR, 40, (HUB_BC/2*math.cos(th), HUB_BC/2*math.sin(th), z0-10)))
    return d


# =====================================================================================
# parts
# =====================================================================================
def bench_stand():
    """sleeve + thrust clamp, on a column, on a clamped base.  Prints flat with the
    bore vertical: no support anywhere, and the column's layers lie in the plane it
    bends in."""
    s = md.sleeve(length=SLEEVE_LEN, window=True, lighten=True, clamp=True)

    # The thrust lug is only 2*THRUST_Z = 14 of the sleeve's 33, so on the robot's parts
    # it prints as a box hanging in mid-air off the -x end wall - 4.8 g of support and a
    # scaffold to dig out of the two nut channels, which are the last place on this part
    # that wants a rough surface.  Here there is no reason to carry it: fill the lug out
    # to the sleeve's full z and it lands on the bed.  Only the two bands OUTSIDE the
    # existing lug, or the union would refill the nut channels and the bolt clearances.
    for z0, z1 in ((-BASE_Z, -md.THRUST_Z), (md.THRUST_Z, BASE_Z)):
        s = s.union(bxc(-S_AX-SLEEVE_W-THRUST_L, -S_AX-SLEEVE_W+2.0,
                        -md.THRUST_YL, md.THRUST_YL, z0, z1))

    # head: full-depth tie between the sleeve's -y flank and the column
    s = s.union(bxc(COL_X0, AXIS_H, COL_Y0, COL_Y1, -COL_Z, COL_Z))

    # The sleeve's -y cooling window is now buried under the column, and the right thing
    # to do about it is nothing: the union fills it solid.  Re-cutting it as a relief was
    # tried and it is the one mistake this orientation punishes - a 20 x 34 pocket with a
    # flat ceiling at z = 6 is 700 mm2 of overhang and 4.4 g of support, in a part that
    # otherwise needs none.  The servo still breathes: the +y window is untouched, the
    # cable window at +x is 15 x 33, and the 35 mm case stands 1 mm proud of the 33 mm
    # sleeve at both ends, so the pocket is open on the axis at both ends.
    #
    # Lighten the head through z instead, which costs nothing to print.
    s = s.cut(bxc(4.0, COL_HEAD_X-COL_RIB, COL_Y0+COL_WALL, COL_Y1-COL_WALL,
                  -BASE_Z-1, BASE_Z+1))

    # lightening windows, through z, so they print as plain vertical holes.  What is
    # left is an I: the two 7 mm y-flanges take the bending, the ribs take the shear.
    x, ys, ye = COL_HEAD_X+COL_RIB, COL_Y0+COL_WALL, COL_Y1-COL_WALL
    while x + 24.0 < AXIS_H-COL_RIB:
        w = min(38.0, AXIS_H-COL_RIB-x)
        s = s.cut(bxc(x, x+w, ys, ye, -BASE_Z-1, BASE_Z+1))
        x += w + COL_RIB

    # base, and two gussets into it.  The gussets are the reason the base can be 8 mm.
    s = s.union(bxc(AXIS_H, AXIS_H+BASE_T, BASE_Y0, BASE_Y1, -BASE_Z, BASE_Z))
    for y0, y1 in ((COL_Y0, max(COL_Y0-34.0, BASE_Y0)),
                   (COL_Y1, min(COL_Y1+34.0, BASE_Y1))):
        s = s.union(wedge([(AXIS_H, y0), (AXIS_H, y1), (AXIS_H-40.0, y0)], -BASE_Z, BASE_Z))

    # clamp slots.  It DOES need clamping: the heavy arm out at +y is 0.25 N*m about the
    # base's front edge against ~0.2 N*m of the stand's own weight, so the margin is a
    # G-clamp and not the footprint.  Obround, through the base, either side of the column.
    for y in (COL_Y0-18.0, COL_Y1+22.0):
        for e in (0, 1):
            s = s.cut(cyl(CLAMP_D/2, BASE_T+2, (AXIS_H-1, y, -CLAMP_RUN/2+e*CLAMP_RUN),
                          axis=(1, 0, 0)))
        s = s.cut(bxc(AXIS_H-1, AXIS_H+BASE_T+1, y-CLAMP_D/2, y+CLAMP_D/2,
                      -CLAMP_RUN/2, CLAMP_RUN/2))
    return s


def bench_arm(reach, bolt_d):
    """a horn on the driven hub.  The beam points at servo-local +X, which is DOWN in
    use, so the arm hangs at q = 0 and sweep.py's angles are measured from the bottom -
    which is the convention m*g*r*sin(q) is written in."""
    z0, z1 = HUB_TOP_Z, HUB_TOP_Z+ARM_T
    a = hub_face(z0, z1)
    x0, x1 = reach-TIP_L, reach+TIP_R
    a = a.union(wedge([(0.0, -ROOT_W/2), (x0, -TIP_W/2), (x1, -TIP_W/2),
                       (x1, TIP_W/2), (x0, TIP_W/2), (0.0, ROOT_W/2)], z0, z1))
    a = a.cut(cyl(bolt_d/2, ARM_T+2, (reach, 0.0, z0-1)))     # the weight bolts here
    return a


PARTS = {}
def build():
    PARTS["bench_stand"]  = (bench_stand(), 1, "PETG, 4 walls, 30% gyroid - flat, no support")
    PARTS["bench_arm_s"]  = (bench_arm(ARM_S_R, ARM_S_BOLT), 1, "PETG, 4 walls, 40% - light arm")
    PARTS["bench_arm_l"]  = (bench_arm(ARM_L_R, ARM_L_BOLT), 1, "PETG, 5 walls, 40% - heavy arm")
    return PARTS


def axis_inertia(shape, rho):
    """kg*m^2 about the servo axis (local z through the origin).  MP.of gives it about
    the part's own com, so this is that plus the parallel-axis term - the arm's disc is
    on the axis but its beam is emphatically not."""
    mp = MP.of(shape, rho)
    cx, cy, _ = mp.c
    return mp.m, (mp.I[2][2] + mp.m*(cx*cx + cy*cy)) * 1e-6


def checks():
    """The three things that can silently be wrong here, checked the way mini_dog.py
    checks `body clear:` - a print is expensive and none of this is visible in a render:

      the bore   the head tie unions a 34 mm block onto the sleeve's -y flank, and a
                 union that reached through the wall would fill the case pocket.
      the hubs   the sleeve is 33 long and the hubs sit at -16.95 / +20.30, i.e. 0.45
                 outside it.  If the stand ever touched one, the servo could not turn.
      the swing  the whole layout rests on the arm and the stand never sharing a z.
                 Assert it against the real solids over a full turn rather than trust
                 the arithmetic in the docstring."""
    st = PARTS["bench_stand"][0].val()
    bad = []
    for name, (wp, _, _) in PARTS.items():
        if not wp.val().isValid():
            bad.append(f"!! INVALID  {name}")
    bore = st.intersect(md.servo_case().val()).Volume()
    hub = st.intersect(md.hubs().val()).Volume()
    worst = max(PARTS["bench_arm_l"][0].rotate((0, 0, 0), (0, 0, 1), a).val()
                .intersect(st).Volume() for a in range(0, 360, 10))
    for what, v in (("bore", bore), ("hubs", hub), ("swing", worst)):
        bad += [f"!! {what.upper()} FOULED  {v:.1f} mm3"] if v > 1.0 else []
    print(f"\n  bore clear: {bore:.1f} mm3   hubs clear: {hub:.1f} mm3   "
          f"swing clear over 360 deg: {worst:.1f} mm3")
    for b in bad:
        print("  " + b)
    return not bad


def report():
    print(f"\n  part            mass    J about axis   print bbox (mm)")
    for name, (wp, qty, note) in PARTS.items():
        sh = wp.val()
        m, J = axis_inertia(sh, rho(name))
        bb = sh.BoundingBox()
        ok = "" if sh.isValid() else "   !! INVALID"
        print(f"  {name:<14}{m*1000:6.1f} g   {J:10.3e}    "
              f"{bb.xlen:5.1f} x {bb.ylen:5.1f} x {bb.zlen:5.1f}{ok}")
        del qty, note

    print(f"\n  axial gap sleeve -> arm: {HUB_TOP_Z - SLEEVE_LEN/2:.1f} mm"
          f"   (the arm's full-circle swing depends on this being > 0)")
    for name, reach in (("bench_arm_s", ARM_S_R), ("bench_arm_l", ARM_L_R)):
        clr = AXIS_H - (reach + TIP_R)
        flag = "" if clr > 10.0 else "   !! the arm hangs into the base"
        print(f"  {name}: reach {reach:.0f} mm, hanging clearance to the base "
              f"{clr:.0f} mm{flag}")

    # will it stand up?  The heavy arm out at +y is a real tipping moment about the
    # base's front edge, and the answer is "only just" - which is what the clamp slots
    # are for.  Printed rather than reasoned about, because it moves with every one of
    # AXIS_H, BASE_Y1, ARM_L_R and the tip mass.
    m_st, _ = axis_inertia(PARTS["bench_stand"][0].val(), rho("bench_stand"))
    c = MP.of(PARTS["bench_stand"][0].val(), rho("bench_stand")).c
    over = m_st*(BASE_Y1-c[1]) + md.SERVO_KG*(BASE_Y1-0.0)
    tip = ARM_MASS["bench_arm_l"]*(ARM_L_R-BASE_Y1)
    print(f"\n  tipping, heavy arm horizontal at +y: {tip*G/1000:.3f} N*m over the base's"
          f" front edge\n  against {over*G/1000:.3f} N*m of stand + servo holding it down"
          f"  ->  {over/tip:.2f}x."
          f"\n  Under ~2x, so it gets clamped: two slots in the base, M6 or a G-clamp.")

    print(f"\n  what to hang on each arm, and what sweep.py is then told")
    print(f"  {'arm':<14}{'tip':>7}{'radius':>8}{'m*g*r':>8}{'/tau_c':>8}"
          f"{'J_arm share':>13}{'period':>9}")
    lines = []
    for name, reach in (("bench_arm_s", ARM_S_R), ("bench_arm_l", ARM_L_R)):
        rh = rho(name)
        mp = MP.of(PARTS[name][0].val(), rh)
        m0, J0 = axis_inertia(PARTS[name][0].val(), rh)
        r = reach/1000.0
        mt = ARM_MASS[name]
        # the printed arm has weight of its own, at its own radius.  It is 1-2 % of the
        # tip's torque, which is small but not nothing, and folding it into an effective
        # tip mass costs one line - sweep.py's --mass only ever enters as m*g*r.
        rcom = math.hypot(mp.c[0], mp.c[1])/1000.0
        meff = mt + m0*rcom/r
        J = J0 + mt*r*r
        tau = meff*G*r
        T = 2*math.pi*math.sqrt((md.MJ_ARMATURE + J)/tau)
        share = J/(md.MJ_ARMATURE + J)
        flag = "" if share < 0.35 else "  !! over fit_bam's third"
        print(f"  {name:<14}{mt*1000:5.0f} g{reach:8.0f}{tau:8.3f}"
              f"{tau/TAU_C_EST:8.1f}x{share*100:11.0f} %{T:8.2f} s{flag}")
        lines.append(f"  python bench/sweep.py --traj all --mass {meff:.3f} "
                     f"--radius {r:.3f} --arm-inertia {J0:.6f} --volts 12.6")
    print(f"\n  ({'tau_c'} is only estimated at {TAU_C_EST} N*m, hence the margin column:"
          f"\n   fit_bam.py cannot read a release whose drive m*g*r*sin(q0) - tau_c is not"
          f"\n   comfortably positive.  J_arm share is J_load/(J_m + J_load), the ratio it"
          f"\n   warns on above 35 %, against MJ_ARMATURE = {md.MJ_ARMATURE} kg*m2 - itself"
          f"\n   the guess the free swing exists to replace.)")
    print(f"\n  from robot/, once the weights are on.  WEIGH them, and weigh the printed"
          f"\n  arm too - --arm-inertia below is a slicer fill factor away from the truth,"
          f"\n  and it is the one number here the fit subtracts rather than fits:")
    for l in lines:
        print(l)


def main():
    for d in ("step", "stl"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    build()
    ok = checks()
    for name, (wp, qty, note) in PARTS.items():
        sh = wp.val()
        cq.exporters.export(cq.Workplane(obj=sh), os.path.join(OUT, "step", name + ".step"))
        bb = sh.BoundingBox()                       # print orientation IS the model frame
        cq.exporters.export(cq.Workplane(obj=sh.translate(cq.Vector(0, 0, -bb.zmin))),
                            os.path.join(OUT, "stl", name + ".stl"))
        del qty, note
    report()
    print(f"\n  -> {os.path.relpath(OUT, HERE)}/{{step,stl}}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
