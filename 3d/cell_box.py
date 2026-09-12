#!/usr/bin/env python3
"""cell_box.py - a storage box for the robot's SPARE cells: eight 21700 in 2 x 4, lying
down, and a lid.

    .venv/bin/python cell_box.py          # -> out/cell_box/{step,stl}

Not a robot part: nothing in `mini_dog.PARTS`, no mass in the budget, no FEA, no sim.
It imports `mini_dog` one way only, for the cell it has to hold: CELL_D / CELL_L are
the Molicel P42A's datasheet maxima over the wrap, so a cell that fits the robot's
`cell_holder` fits here.

How a cell comes out: the trough under each cell stops PIT_L short of one end and the
floor drops away there.  Press that end down, the other end pivots up out of the box
(6 mm down at one end is ~15 mm up at the other) and you take it by the raised end.
Nothing has to be pinched between neighbours, so the dividers can be a print line.

The lid is a cap: the box wall steps to an inner lip LIP_H tall and the lid's skirt
sits on the step, flush outside, held by friction on the lip.  Prints open side up,
no support, either part.
"""
from __future__ import annotations

import os
import sys

import cadquery as cq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mini_dog import CELL_D, CELL_L, bxc, cyl      # noqa: E402

OUT = os.path.join(HERE, "out", "cell_box")

NX, NY   = 2, 4                       # cells along x (lying), rows across y
POCKET_W = CELL_D + 1.0               # 22.3: a loose drop-in, not the holder's push fit
POCKET_L = CELL_L + 1.5               # 71.7, along the cell
DIV      = 1.6                        # divider between pockets
WALL     = 3.0                        # outer wall = lip + clearance + lid skirt
FLOOR    = 1.6
PIT_D    = 6.0                        # floor drop under the press end
PIT_L    = 20.0                       # how far the trough stops short of that end
DIV_H    = 5.0                        # divider above the cell's equator
RIM_UP   = 1.0                        # rim above the cell top, lid off
LIP_T    = 1.4                        # inner lip the lid grips
LID_T    = 2.0
LID_H    = 6.0                        # skirt = lip height
LID_FIT  = 0.2                        # per side, lip to skirt

R_CELL = POCKET_W / 2
Z_AXIS = FLOOR + PIT_D + R_CELL                 # cell axis, at rest
Z_TOP  = Z_AXIS + R_CELL + RIM_UP               # inner wall top
IN_X   = NX * POCKET_L + (NX - 1) * DIV
IN_Y   = NY * POCKET_W + (NY - 1) * DIV
OUT_X, OUT_Y = IN_X + 2 * WALL, IN_Y + 2 * WALL
Z_STEP = Z_TOP - LID_H                          # where the wall thins to the lip


def pocket_origin(i, j):
    return (WALL + i * (POCKET_L + DIV), WALL + j * (POCKET_W + DIV))


def box():
    b = bxc(0, OUT_X, 0, OUT_Y, 0, Z_STEP)
    b = b.union(bxc(WALL - LIP_T, OUT_X - WALL + LIP_T,
                    WALL - LIP_T, OUT_Y - WALL + LIP_T, Z_STEP, Z_TOP))
    # interior above the dividers, then the pockets
    b = b.cut(bxc(WALL, OUT_X - WALL, WALL, OUT_Y - WALL, Z_AXIS + DIV_H, Z_TOP + 1))
    for i in range(NX):
        for j in range(NY):
            x0, y0 = pocket_origin(i, j)
            yc = y0 + R_CELL
            b = b.cut(cyl(R_CELL, POCKET_L, (x0, yc, Z_AXIS), axis=(1, 0, 0)))
            b = b.cut(bxc(x0, x0 + POCKET_L, y0, y0 + POCKET_W, Z_AXIS, Z_TOP + 1))
            # the press end is the outer end of each pocket, so both columns tilt outward
            px0, px1 = (x0, x0 + PIT_L) if i == 0 else (x0 + POCKET_L - PIT_L, x0 + POCKET_L)
            b = b.cut(bxc(px0, px1, y0, y0 + POCKET_W, FLOOR, Z_AXIS + 1))
    return b


def lid():
    ix = IN_X + 2 * LIP_T + 2 * LID_FIT
    iy = IN_Y + 2 * LIP_T + 2 * LID_FIT
    l = bxc(0, OUT_X, 0, OUT_Y, 0, LID_T + LID_H)
    l = l.cut(bxc((OUT_X - ix) / 2, (OUT_X + ix) / 2,
                  (OUT_Y - iy) / 2, (OUT_Y + iy) / 2, LID_T, LID_T + LID_H + 1))
    return l


PARTS = {"cell_box": box, "cell_lid": lid}


def main():
    for d in ("step", "stl"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    ok = True
    print(f"  cell {CELL_D} x {CELL_L}, {NX} x {NY} lying, pocket {POCKET_W} x {POCKET_L}")
    print(f"  box {OUT_X:.1f} x {OUT_Y:.1f} x {Z_TOP:.1f}, with lid {Z_TOP + LID_T:.1f} tall")
    print(f"  lever: {PIT_D:.0f} mm down at the press end -> "
          f"{PIT_D * (POCKET_L - PIT_L) / PIT_L:.0f} mm up at the other")
    for name, fn in PARTS.items():
        sh = fn().val()
        good = sh.isValid() and sh.Volume() > 0 and len(sh.Solids()) == 1
        ok &= good
        bb = sh.BoundingBox()
        print(f"  {name:<9} {sh.Volume() / 1000:6.1f} cm3  {bb.xlen:.1f} x {bb.ylen:.1f} x {bb.zlen:.1f}"
              + ("" if good else "   !! INVALID"))
        cq.exporters.export(cq.Workplane(obj=sh), os.path.join(OUT, "step", name + ".step"))
        cq.exporters.export(cq.Workplane(obj=sh), os.path.join(OUT, "stl", name + ".stl"))
    print(f"  -> {os.path.relpath(OUT, HERE)}/{{step,stl}}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
