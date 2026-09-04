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
green. `7e30e75` was not, and it never could have been** — the section on it is kept as
the record of why; what replaced it, on 2026-09-04, is the last section of this file.

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

## `7e30e75`'s bores did not do what they were cut to do (superseded 2026-09-04)

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
  has no wall left. Do not print a chassis against this note. **Replaced** by the two corner
  channels of the last section: inboard arm first, ball-end key leaning 20° outboard, a
  quarter turn of the leg between screws, deck and battery in place.

## Fixed 2026-09-04: two channels through the corner, and the leg turns

The four bores are gone. What replaced them is in `3d/README.md`, "Reaching the fork
screws", and rests on two facts the bores ignored: the screw circle **turns with the leg**,
so two channels a quarter turn apart serve all four screws; and a key leaning 20°
*outboard* is in open air past the tray's front corner after 8 mm, without entering the
tray — where the battery cradle's end stop closes every straight path within 18 mm anyway.
`fork_access_channels()` cuts them on the assembled chassis, last; `fork_access()` now
follows the same two lines for their whole 55 mm, i.e. to open air, and a failed boolean
counts as blocked. `fork_channels_cover()` is the closed-form turn check.

The ladder, with the unchanged tree as the control beside it (same seeds, same day):

| step | result |
|---|---|
| 1 `mini_dog.py` | every part valid, volumes positive; ROM −90/+90, −90/+90, −110/+110; `body clear`, `imu clear +0.80`, `foot bolt` ok, `clamp clear +1.43` |
| | `fork access: five arms open within 40 mm; roll/passive through 2 @6 channels leaning 20 deg, every screw within 90 deg of one` |
| 2 bboxes | unchanged; `chassis_bottom` 240.18 → 239.8 cm³ (−0.5 g, the bores' 2.9 g put back and the channels taken out) |
| 3 render | three PNGs plus a corner close-up with the key in the channels |
| 4 `fea.py --all` | `hip_bracket_A` 46.7 / 23.3 / 7.8 / 2.4, `thigh_A` 18.7 / 9.4 / 3.1 / 1.2, `shin_A` 58.8 / 29.4 / 9.8 / 4.8 inter-layer — identical to 2026-09-03; nothing in the FEA set was touched, and `chassis_bottom` is not in it |
| 5 `export_sim.py --check` | `4 feet down, upright +1.00`, base z 187, 2.495 kg, terrain the same, camera axis (+0.99 −0.00 +0.10), urdf/mjcf leg mass agree |
| 6 ROS 2 | regenerated |

| | control, unchanged tree | after |
|---|---|---|
| flat trot | 782.4 mm | 781.8 mm |
| terrain, seeds 7…12 | 625 ±95 mm | 631 ±55 mm |
| course | 5/7, corridor 2893 mm | 4/7, corridor 2639 mm |

The flat arm moved 0.6 mm and the terrain sweep's means differ by 6 mm against spreads of
95 and 55 — the same distribution. The course flipped from 5/7 to 4/7 (the wall at
2600 mm), which is the flip `3d/CLAUDE.md` already documents on the *unchanged* tree: it is
a report, not a pass/fail, and −0.5 g on the base link is what it flips on. Total mass is
2.495 kg to three decimals either side.

What the chassis gives up: a ⌀6 hole through the front corner post at axis height, and a
⌀6 hole through the lower half of each corner deck boss (z 4…10, under its M3 hole at
z ≥ 11) that runs on out through the side wall into the vent. `fea.py` cannot put a number
on either; the side wall carries 22 × 18 vents already.

**A chassis already printed does not need reprinting.** Drill both, ⌀6, from outside,
aimed 20° inboard of the body's long axis at the screw head — "out" enters the side wall
3 mm behind the front face at axis height, "top" enters through the side vent 22 mm behind
the front face and 18 mm below the top edge — and stop at the gusset face (8 and 28 mm in):
past it is the servo.

## Same day: the screws arrived, and their heads were not in the model

ISO 7380 M3 × 6 button heads, 2 mm hex socket — so the key for the channels is **2 mm**,
not 2.5. And the head is 1.65 mm tall against an inboard arm that had **0.6 mm** of air
to the gusset: the hip would have bound on its own screw heads before it turned a degree,
and no check saw it, because like the thrust clamp's cap head the part in the way was
hardware. Fix: `FORK_GAP` 0.6 → 1.1 (the gusset face moves back half a millimetre), a
`FORK_CB` = 1.2 counterbore in the passive arm, `fork_screws()` on the moving side of every
`rom_scan`, and `head_clear()` printing the margins. The counterbore also makes the
passive screw an M3 × 6 — it was an M3 × 7, a length nobody stocks. Reach into the case:
0.5 mm spare passive, 0.8 driven (which engages 2.0 of its hub's 2.5).

The ladder again, against the channel commit `89e5149` as the control:

| step | result |
|---|---|
| 1 `mini_dog.py` | all valid; ROM **with the heads swept** still −90/+90, −90/+90, −110/+110; `head clear: +0.65 mm off the gusset; tip +0.50 / +0.80 short of the case`; every other line unchanged |
| 2 mass | `chassis_bottom` 239.8 → 239.1 cm³ (the gusset's half millimetre), `hip_bracket` 25.1 → 25.0 (the counterbores); total 2.495 → 2.493 kg |
| 4 `fea.py --all` | `hip_bracket_A` 46.8 / 23.4 / 7.8 / 2.4 inter-layer against 46.7 / 23.3 / 7.8 / 2.4 — the counterbore is on the passive arm, the stall peak is not; `thigh_A` 18.8 / 9.4 / 3.1 / 1.2, `shin_A` 60.0 / 30.0 / 10.0 / 4.9, both a rounding step up with the 2 g |
| 5 `export_sim.py --check` | `4 feet down, upright +1.00`, base z 187, camera axis (+0.99 −0.00 +0.10), urdf/mjcf leg mass agree |
| 6 ROS 2 | regenerated — every hip, thigh and shin mesh moved, because `fork()` is one part and the counterbore is in all twelve |

| | `89e5149` | after |
|---|---|---|
| flat trot | 781.8 mm | 781.1 mm |
| terrain, seeds 7…12 | 631 ±55 mm | 657 ±35 mm |
| course | 4/7, corridor 2639 mm | **3/7, corridor 2209 mm** |

Flat and terrain are the same walker. The course has now read 5/7, 4/7 and 3/7 on three
models 0.5 g and 2 g apart, all within the day, on the same seed — a monotone-looking
sequence of three chaotic samples. `3d/CLAUDE.md` already says the unchanged tree reads
both 5/7 and 4/7, and the terrain sweep, which is the arm that is read over seeds, moved
+26 mm on a ±35 spread. It is flagged here, not acted on: the fix for a course number is
a gait change, and there is no CAD change on the table that would put 2 g back.
