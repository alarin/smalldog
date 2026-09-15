#!/usr/bin/env python3
"""test_pack.py - a screw-terminal holder for three 18650, 3S1P: the TEMPORARY pack for
bench tests before the robot's own 3S2P 21700 module exists.  No spot welding: nickel
strip tabs are clamped under M3 screws, the series links and the balance lead are ring
terminals under the same screws, and cell pressure comes from a sliding end plate pushed
by two jack screws.

    .venv/bin/python test_pack.py         # -> out/test_pack/{step,stl}

Not a robot part: nothing in `mini_dog.PARTS`, no mass in the budget, no FEA, no sim.
It imports `mini_dog` one way only, for the primitives and `nut_slot()`.

Layout.  Cells lie side by side along x, alternating (PLUS_FIX): cell 1 and 3 with + at
the FIXED end (x = 0), cell 2 with + at the plate end - put a cell in backwards and its
button lands on a strip recessed for a flat end, which is a short through the rim, so
mark the + ends on the print with a pen before the first cell goes in.  Each cell end
meets its own tab - a strip TAB_L long standing in a shallow groove on the end plate's inner face, clamped at the top
by an M3 from the inside into a nut in the plate (nut slot open at the top edge).  The
series jumpers are then short: at the fixed end c1+ -> c2- and at the plate end c2+ ->
c3-, ring terminal to ring terminal under the tab screws; pack - is c1- (plate end), pack
+ is c3+ (fixed end); balance taps B0..B3 are those same four screws.  Nothing is soldered
to a cell.

Pressure.  The plate end is a separate PLATE that drops into a notch in both side walls
and slides along x by TRAVEL; two M3 jack screws through nuts in the OUTER wall push it
onto the cells (the thrust clamp's idea).  Drop the cells in loose, then tighten until the
tabs press.  Cells from 64.5 to 69 mm long fit: unprotected flat-top 18650 are 65.0-65.3,
button-tops ~67-69.  Do not use protected cells - the PCB tail is not a contact.

BMS.  The robot's own 3S board (mini_dog.BMS_*, same-port per POWER.md) rides on a LID
that spans the troughs and doubles as the cell retainer: a 3 mm plate on the side walls
with a rim round the board and four tie slots, two 2.5 mm ties round the whole body.
Its five leads go to the tab screws under the rings: B- to c1- (plate end), B1 to the
c1+/c2- screw (fixed wall), B2 to the c2+/c3- screw (plate end), B+ to c3+ (fixed wall);
P-/C- is the pack's negative output and c3+ its positive.  Nothing else changes.

Prints open side up, no support, all three parts.  Flat-top cells only, wrap intact: a bare
can shoulder against the PLA divider is fine, against a neighbour it is a short.
"""
from __future__ import annotations

import os
import sys

import cadquery as cq

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from mini_dog import BMS_H, BMS_L, BMS_W, M3_CLR, TIE_SLOT, bxc, cyl, nut_slot   # noqa: E402

OUT = os.path.join(HERE, "out", "test_pack")

CELL_D  = 18.6                        # 18650 over the wrap, maximum
CELL_L  = (64.5, 69.0)                # what the travel accepts, shortest .. longest
N       = 3
R_TR    = CELL_D / 2 + 0.5            # trough radius: a loose drop-in
DIV     = 1.6                         # divider between troughs
SIDE    = 2.4                         # side wall
FLOOR   = 2.0
WALL_T  = 6.0                         # end walls and the plate: a nut inside, 1.5 skins
GAP     = 5.0                         # air behind the plate at rest = the jack travel
TRAVEL  = GAP
LIP     = 4.0                         # dividers and side walls above the cell axis
TAB_W   = 8.0                         # nickel strip: 8 x 0.15
TAB_L   = 24.0
TAB_DZ  = 13.5                        # tab screw above the cell axis: clear of the rim
                                      # (9.3) and the M3 head (2.75) by 1.4
STRIP_G = (TAB_W + 0.6, 0.1, 0.4)     # the strip's groove: width, depth at a - end, at a
                                      # + end.  A + end's button stands ~0.7 proud of the
                                      # crimped rim, and the rim is the CAN - negative.
                                      # There the strip sits 0.25 below the plastic, so
                                      # the rim seats on plastic and only the button
                                      # reaches the nickel.  At a - end the whole flat
                                      # can end is the contact, and the strip stands
                                      # 0.05 proud.
PLUS_FIX = (True, False, True)        # which cell has its + at the fixed wall
NOTCH   = 1.5                         # the plate's tabs into the side walls
PLATE_FIT = 0.3                       # plate to notch, along x and y
LID_T   = 3.0                         # roof over the cells
LID_CLR = 0.3                         # the lid's arches over the trough radius
LID_RIM = 2.0                         # locating rim round the BMS, height and width
LID_FIT = 0.4                         # rim to board, per side

PITCH  = 2 * R_TR + DIV
WIDTH  = 2 * SIDE + N * 2 * R_TR + (N - 1) * DIV
Z_AXIS = FLOOR + R_TR
Z_LIP  = Z_AXIS + LIP
Z_TOP  = Z_AXIS + TAB_DZ + 4.0                  # end walls: the tab nut's slot needs it
X_FIX  = WALL_T                                 # fixed wall's inner face
X_PL0  = X_FIX + CELL_L[1] + 0.5                # plate's inner face at rest
X_OUT  = X_PL0 + WALL_T + GAP                   # outer wall's inner face
LENGTH = X_OUT + WALL_T
CELL_Y = [SIDE + R_TR + i * PITCH for i in range(N)]
JACK_Y = [SIDE + 2 * R_TR + DIV / 2 + i * PITCH for i in range(N - 1)]


def tabs(s, x_face, sign, plus):
    """The three tab grooves and tab screws on an inner face at x_face; sign = which
    way is 'inward' (+1 for the fixed wall, -1 for the plate); plus = per cell, whether
    its + end meets this face (groove depth, see STRIP_G).  The screw runs from the
    inside, head on the strip, into a nut slot open at the top edge."""
    gw, g_minus, g_plus = STRIP_G
    z_scr = Z_AXIS + TAB_DZ
    for y, pl in zip(CELL_Y, plus):
        gd = g_plus if pl else g_minus
        s = s.cut(bxc(x_face, x_face - sign * gd, y - gw / 2, y + gw / 2,
                      Z_AXIS - TAB_L + TAB_DZ + 2.0, Z_TOP + 1))
        s = s.cut(cyl(M3_CLR, WALL_T + 2, (x_face + sign, y, z_scr), axis=(-sign, 0, 0)))
        # nut centred in the wall: 1.5 mm of skin on either face
        s = s.cut(nut_slot((x_face - sign * (WALL_T / 2 - 1.5), y, z_scr),
                           (0, 0, 1), up=(-sign, 0, 0), run=Z_TOP - z_scr + 1))
    return s


def body():
    b = bxc(0, LENGTH, 0, WIDTH, 0, Z_LIP)
    b = b.union(bxc(0, X_FIX, 0, WIDTH, 0, Z_TOP))                  # fixed end wall
    b = b.union(bxc(X_OUT, LENGTH, 0, WIDTH, 0, Z_TOP))             # outer wall
    x_pl = X_PL0 - TRAVEL                                           # plate, fully in
    for y in CELL_Y:                                                # troughs, open top
        b = b.cut(cyl(R_TR, x_pl - X_FIX, (X_FIX, y, Z_AXIS), axis=(1, 0, 0)))
        b = b.cut(bxc(X_FIX, x_pl, y - R_TR, y + R_TR, Z_AXIS, Z_TOP + 1))
    # the plate's run and the bay behind it, down to the floor it slides on
    b = b.cut(bxc(x_pl, X_OUT, SIDE, WIDTH - SIDE, FLOOR, Z_TOP + 1))
    # ... and its notch NOTCH deep into both side walls, PLATE_FIT longer than the run
    b = b.cut(bxc(x_pl, X_PL0 + WALL_T + PLATE_FIT, SIDE - NOTCH, WIDTH - SIDE + NOTCH,
                  FLOOR, Z_TOP + 1))
    b = tabs(b, X_FIX, +1, PLUS_FIX)
    for y in JACK_Y:                                                # jack screws
        b = b.cut(cyl(M3_CLR, WALL_T + 2, (LENGTH + 1, y, Z_AXIS), axis=(-1, 0, 0)))
        b = b.cut(nut_slot((X_OUT + WALL_T / 2 + 1.5, y, Z_AXIS), (0, 0, 1),
                           up=(-1, 0, 0), run=Z_TOP - Z_AXIS + 1))
    return b


def plate():
    w = WIDTH - 2 * (SIDE - NOTCH) - PLATE_FIT
    p = bxc(X_PL0, X_PL0 + WALL_T, (WIDTH - w) / 2, (WIDTH + w) / 2, FLOOR, Z_TOP)
    return tabs(p, X_PL0, -1, [not q for q in PLUS_FIX])


def lid():
    """The BMS carrier and cell retainer: sits on the side walls between the fixed wall
    and the plate at full travel.  Board flat, component side up, between two rims along
    its long edges (the ends are open: wires out one, the other is where it slides in);
    two ties over the board and round the whole body hold board, lid and cells, and the
    grooves across the rims are where they sit."""
    # 4 mm short of each end wall: the tab screws' heads and rings sit there, at the
    # lid's height.  The board overhangs the lid's ends by ~2.5 mm; the rims are on the
    # long sides, so nothing depends on the ends.
    x0, x1 = X_FIX + 4.0, X_PL0 - TRAVEL - 4.0
    # The cells stand R_TR - LIP above the side walls, so the lid is a block that sits
    # on walls and dividers with an arch over each cell and LID_T of roof above them.
    z0 = Z_LIP
    z1 = Z_AXIS + R_TR + LID_CLR + LID_T
    l = bxc(x0, x1, 0, WIDTH, z0, z1)
    for y in CELL_Y:
        l = l.cut(cyl(R_TR + LID_CLR, x1 - x0 + 2, (x0 - 1, y, Z_AXIS), axis=(1, 0, 0)))
    bl, bw = BMS_L + 2 * LID_FIT, BMS_W + 2 * LID_FIT
    bx0, by0 = (x0 + x1 - bl) / 2, (WIDTH - bw) / 2
    for y in (by0 - LID_RIM, by0 + bw):
        l = l.union(bxc(bx0, bx0 + bl, y, y + LID_RIM, z1, z1 + LID_RIM))
    ta, tb = TIE_SLOT
    for x in (bx0 + 10.0, bx0 + bl - 10.0):                   # tie grooves, over the rims
        l = l.cut(bxc(x - ta / 2, x + ta / 2, -1, WIDTH + 1, z1 + LID_RIM - 1.0, z1 + LID_RIM + 1))
    return l


def main():
    for d in ("step", "stl"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    print(f"  {N} x 18650 in a row, troughs @{2 * R_TR:.1f} on a {PITCH:.1f} pitch")
    print(f"  body {LENGTH:.1f} x {WIDTH:.1f} x {Z_TOP:.1f}; cells {CELL_L[0]}..{CELL_L[1]} long"
          f" ({TRAVEL:.0f} mm of jack travel)")
    print(f"  tabs: {N * 2} x nickel strip {TAB_W:.0f} x {TAB_L:.0f}, one hole @3.4 at"
          f" {TAB_L - 2.0:.0f} from the contact end; M3 x 8 button + nut, {N * 2}"
          f"; jack screws M3 x 16 x {len(JACK_Y)} + nut")
    print(f"  lid: arches over the cells, {LID_T:.0f} mm roof; BMS {BMS_L} x {BMS_W} x {BMS_H} flat on"
          f" top, rim {LID_RIM:.0f}, two 2.5 mm ties round the body")
    for name, obj in (("body", body()), ("plate", plate()), ("lid", lid())):
        sh = obj.val()
        good = sh.isValid() and sh.Volume() > 0 and len(sh.Solids()) == 1
        bb = sh.BoundingBox()
        print(f"  {name:<6} {sh.Volume() / 1000:6.1f} cm3  {bb.xlen:.1f} x {bb.ylen:.1f} x"
              f" {bb.zlen:.1f}  {'ok' if good else '!! INVALID'}")
        cq.exporters.export(obj, os.path.join(OUT, "step", name + ".step"))
        cq.exporters.export(obj, os.path.join(OUT, "stl", name + ".stl"))
    print(f"  -> {os.path.relpath(OUT, HERE)}/{{step,stl}}")


if __name__ == "__main__":
    main()
