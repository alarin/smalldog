# The mac's queue — worked 2026-09-03

Three machines share this repository and nothing else (`rl/CLAUDE.md`, "The machine
split"):

| machine | what it is for |
|---|---|
| **this one**: a mac | the CAD — `3d/`, and everything generated out of it |
| Windows + WSL2, RTX 3070 | training — `rl/`, plus the pure-Python MuJoCo sim. See [`WSL.md`](WSL.md) |
| an Orange Pi 5 Pro | the robot — `robot/runtime`, step 7 |

Five commits changed `3d/` from a session that had no CadQuery. The ladder in
`3d/CLAUDE.md`, "Verifying a change", has now been run against all five. **Four of them are
green. `7e30e75` is not, and it never could have been** — see the last section, which is
the only part of this file still live.

## What the ladder said

| step | result |
|---|---|
| 0 baseline | `fea.py --all` at `b4d7e4a`, kept for step 4 |
| 1 `mini_dog.py` | **found `chassis_bottom` at −0.3 cm³. Fixed** — see below. After the fix: every part valid, ROM unmoved at −90/+90, −90/+90, −110/+110, `body clear`, `gps clear`, `imu clear +0.80`, `foot bolt` ok |
| | `clamp clear: M3 x 10 set screw reaches r = 21.57 vs the spine's 23.0 (+1.43 mm)` — exactly as predicted |
| 2 bboxes | not one part's outline moved |
| 3 render | three PNGs; the hip was measured rather than eyeballed, and that is where the live defect came from |
| 4 `fea.py --all` | **no inter-layer SF dropped anywhere.** `hip_bracket_A` improved (stand4 46.3 → 46.7, stand2 23.1 → 23.3, land3g 7.7 → 7.8, stall 2.4 → 2.4); `thigh_A` and `shin_A` identical to a decimal. The bigger bearing area wins, as argued |
| 5 `export_sim.py --check` | `4 feet down, upright +1.00`, base z 187 mm, terrain the same, camera axis (+0.99 −0.00 +0.10), urdf/mjcf leg mass agree |
| 6 ROS 2 + consumers | regenerated; `check_model.py` and `--terrain` both **0 FAIL, 4 warn** (all pre-existing "GUESSED" warnings); `walk.py --dry-run --profile` 1050 ticks at 50 Hz, 0 late; `calib/safety/loop --selftest` all ok |
| 7 `bench_rig.py` | passes |

Mass came out **−4.15 g**, not the −1.2 g this file predicted: the hub holes are 1.3 g of
it and the access bores are 2.9 g, so "plus a little from the access bores" was the wrong
way round. 2.499 → 2.495 kg.

**The mass control was run, because −4 g is inside the band `3d/CLAUDE.md` calls
hypersensitive.** Same seed, the unchanged model beside the changed one:

| | before, 2.499 kg | after, 2.495 kg |
|---|---|---|
| flat trot | 781.7 mm | 782.4 mm |
| course | 5/7, corridor 2892 mm | 5/7, corridor 2893 mm |
| terrain, seeds 7…12 | 649 ±48 mm | 625 ±89 mm |

The flat arm moved 0.7 mm and the course is the same obstacle count and the same
millimetre, so the cliff was not crossed. The terrain sweep's two means differ by 24 mm
against spreads of 48 and 89 — the same distribution, which is what indistinguishable has
to look like.

## The defect that was fixed: `chassis_bottom` was a −257 mm³ sliver

`7e30e75` cut its four ⌀6 bores *before* the rear connector pads were unioned on, and the
pads start at exactly the plane the bores end on — `xw+WALL` = −60.2 against the bores'
`BODY_L/2-WALL` = 60.2. They share **no volume at all**; the pad's inner face was welded
straight onto the bore's circular opening, and OCC's fuse on that contact returned an
inverted solid. `isValid()` said True, so nothing downstream noticed: 240 cm³ of chassis
became a 2.5 × 14.9 × 6.9 mm sliver, the ROM read `hip_roll +0 .. +0`, every
`interference()` pair fired at once, and the robot came out 280 g light.

The cut now happens last, after everything that adds material near it, and `build()` fails
a part whose volume is not positive. Both are written up in `3d/CLAUDE.md`.

## Still live — `7e30e75`'s bores do not do what they were cut to do

Measured on the repaired solid, front-left hip, ⌀2.5 key swept over tilt and direction:

* **The deck boss at (±52, ±38) stands in front of three of the four screws.** It is
  `DECK_BOSS_R` = 5.8 and full height, z −22…+25, and its outer face is at x = 57.8 — 2.4 mm
  inboard of the bore mouth at 60.2. On the screw axes it overlaps the key by 5.05 mm at
  y = 36 (both z = ±7) and 2.05 mm at y = 43. Only the y = 29 screw is clear, and it is
  clear straight-on with a 25 mm key, 0.00 mm³ blocked. `7e30e75` checked the bores against
  the sleeve, the battery cradle and the BMS bay; the bosses are the one thing directly in
  line and they were not checked.
* **The outboard bore breaks out through the side of the robot.** Its axis is at y = 43 and
  at ⌀6 it reaches y = 46.00 — the tray's outer skin, exactly. Not "tangent inside the wall
  where it is full width": the front wall's *outer* surface is at y = 46 for the bore's whole
  2.8 mm of x. Measured at x = 61.6, the skin left between bore and outside is **0.00 mm at
  z = 0**, 0.04 at z = ±0.5, 0.17 at ±1.0, 0.40 at ±1.5. That is an open slot with a knife
  edge, not a tunnel.
* **An M3 head cannot pass that wall on that axis at all.** ⌀5.5 centred at y = 43 reaches
  45.75 against the skin at 46 — 0.25 mm, at zero clearance. So "⌀6 also passes an M3 head,
  so the screws go in this way" is false for the screw that motivated the bores.
* **The bore's tilt budget is half what the commit claims.** 23.9° = atan(3.5/7.9) assumes
  the key pivots at the bore's middle. It pivots at the *screw*, at the far end, so the
  mouth-end swing is (D−d)/2 over L: atan(1.75/7.9) = **12.5°**. Measured, the key is clean
  in the bore to 12° and fouling by 16°. 12.5° is not enough to clear the tray's side wall,
  which the key must cross for its first ~4.7 mm.

`fork_access()` prints `all six arms break out within 7.5 mm` through all of this, because
`DRIVER_REACH` = 7.5 against a 7.9 mm bore keeps the probe **inside the bore** — by 0.4 mm,
as `7e30e75` itself notes. It therefore tests that the bore exists and nothing else. It is
a check that passes on a screw nobody can turn, which is the exact failure mode
`foot_bolt_check()` and `thrust_clear()` were written to end.

Three ways out, and picking one is a design decision, not a verification one:

1. **move the deck bosses.** `DECK_SCREWS`' outer pair at x = ±52 is the only thing in the
   way of two of the three; y = 38 → 30 or so would clear y = 36 and y = 43 both. Cheap in
   CAD, but it moves a deck screw and its nut slot, and the mid pair's clipping comment in
   `chassis_bottom` says that pattern has already been fought over once;
2. **give up on driving those screws through the tray** and pre-place them in the fork arm,
   which makes the bore a key-clearance hole rather than a head-clearance one — but 12.5° of
   tilt still does not clear the side wall for the outboard one, so this fixes three
   screws, not four;
3. **change what the fork bolts to at the roll joint** so the inboard arm's screws are not
   blind. This is the honest one and the expensive one.

Whichever it is, `fork_access()` has to stop probing only inside the bore — the probe needs
to continue into the tray until it reaches open air, and then it will fail until the
geometry is fixed. A red line that is true beats a green one that is not.

## Hardware notes that go with this batch

* **Thrust clamp: M3 × 10 set screws**, not cap screws. A cap head reaches r = 24.2 into an
  annulus the fork spine sweeps from 23.0 and binds the joint over ±2…38°. Found on the
  bench, then reproduced in the arithmetic. Verified here: `clamp clear` reads +1.43 mm.
* **Hub screws are M3 × 6 driven / M3 × 7 passive.** Ceilings 6.5 and 7.15; past the hub
  they bottom on the case and the vendor FAQ says that burns servos. A legs plate already
  printed does **not** need reprinting — the hole only grew, so drill the eight per joint
  out to 3.3–3.4.
* ~~**The hip's inboard fork screws are fitted from inside the tray**, through the ⌀6 bores,
  with a **ball-end** hex key tilted inboard.~~ **Not true as built** — see the section
  above. One of the four is reachable; three are behind a deck boss, and the outboard one
  has no wall left. Do not print a chassis against this note.
