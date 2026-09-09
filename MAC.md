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

## 2026-09-09: the battery became a module, and the IMU moved to make room

The ask was safety: no loose cells in the dog, the pack in heatshrink and a plastic case.
The design that followed is not a wrapper round the old cradle — it replaced it. Six cells
welded into a 3 × 2 brick, heatshrunk, the BMS beside them, all of it in `battery_case` +
`battery_lid`, dropped into a seat recess in the tray floor as one payload. The cradle's
four fins, two end stops, strap slots and the front BMS bay are gone.

**The binding constraint was height, and it was not close.** From the tray floor at −22 to
the old pack ceiling at 21.4 there were 43.4 mm for 42.6 mm of cell: 0.8 mm for a floor, a
lid and a fit. What sat in the 3.6 mm above the pack was the IMU, so the IMU is what moved
— onto the deck's top face, on two standoffs inside the Orange Pi's 7 mm standoff gap. The
deck's cable window moved off the centreline (`IMU_WINDOW`, x −34…−12) to leave solid deck
under the board. Three knock-ons, all forced, none cosmetic:

* the module reaches x = −46.15, so the ESP32/URT-1 bay moved to `ESP_X` = −51 and `BATT_X`
  was pushed forward to 5.0 until that strip was the 8.9 mm it had before;
* the four corner deck screws moved from |y| = 38 to 41 — measured on the solid, at 38
  their bosses took a 1.9 mm bite out of the module's corners — so all eight are at 41 now,
  and `DECK_SCREWS` carries its nut-channel direction explicitly because |y| no longer
  tells the two pairs apart;
* the BMS went inside the module, so `BMS_KG` = 55 g is split out of `ELECTRONICS_KG` and
  hangs at `bms_com()` instead of being averaged into the Pi's box 46 mm away and 40 mm up.

Two defects were found and fixed during the build, both by measurement rather than by
reading:

* **the lid was not captured at all.** It was drawn to tuck under an inward tongue on the
  rear wall, and a tongue cannot work when the lid is flush with the module's top — there
  is nowhere at the lid's own height for it to be. It butted against the tongue instead.
  It is a groove in the wall's inner face now, and the lid's rear tab is only as wide as
  the interior, because full width put its two rear corners inside the side walls.
* **the lid's locating lip was a slab, not a frame** — 12 cm³ of plastic doing nothing but
  weigh 14 g, on a robot whose flat trot moves on 11.

### The ladder

| step | result |
|---|---|
| 0 baseline | `fea.py --all` at `c051e66`, in a worktree, kept for step 4 |
| 1 `mini_dog.py` | every part valid, ROM unmoved at −90/+90, −90/+90, −110/+110, `body clear` (now covering `battery_case` and `battery_lid` too — the pack is inside `interference()` for the first time), `imu clear: +3.40 mm` to the Pi, `batt clear: +1.40 mm` to the deck, every fastener probe unchanged to the hundredth |
| 2 bboxes | `battery_case` 102.3 × 68.2 × 46.4, `battery_lid` 96.2 × 68.2 × 3.2 — both well inside a 256 bed |
| 3 render | three PNGs, looked at |
| 4 `fea.py --all` | **not one SF moved, in-plane or inter-layer.** `hip_bracket_A` 46.8 / 23.4 / 7.8 / 2.4, `thigh_A` 18.8 / 9.4 / 3.1 / 1.2, `shin_A` 60.0 / 30.0 / 10.0 / 4.9 — all three identical to the baseline to a decimal, at 2.462 kg against the baseline's 2.4619. `fea.robot_mass()` had to be taught `BMS_KG` first, or it would have lost 55 g of robot when that constant was split out of `ELECTRONICS_KG` — the load cases scale with the total |
| 5 `export_sim.py --check` | `4 feet down, upright +1.00`, base z 187 mm, terrain the same, camera axis (+0.99 +0.00 +0.10), urdf/mjcf leg mass agree |
| 6 ROS 2 | regenerated; see below |

Mass came out **+2.3 g** (2.4619 → 2.4642 kg on `fea.robot_mass()`): the deleted cradle
almost exactly pays for the case. What did move is the **base link's centre of mass**,
+2.75 mm forward and −1.09 mm down — the cells sit 9.3 mm ahead of the module's centre
because the BMS is behind them, and 55 g of BMS came down out of the Pi's box.

### What the gait did, with a control beside it

| | control, `c051e66` | after |
|---|---|---|
| flat trot | 790.8 mm, z 167.8 | **595.2 mm, z 174.0** |
| terrain, seeds 7…12 | 537 ±70 mm | 498 ±76 mm |
| course | 5/7, corridor 2778 mm | 5/7, corridor 2876 mm |

**Read the terrain and course rows, not the flat one.** The two terrain means differ by
39 mm against spreads of 70 and 76 — about half a standard deviation, and the standard
error of the difference is 42 mm, so the two sweeps are one distribution. The course
cleared the same five obstacles and went 98 mm further. The flat trot lost 196 mm and the
body rides 6.2 mm higher, which is the exact signature `3d/CLAUDE.md` documents for this
walker crossing its bifurcation on a few grams: it is the operating point moving, not the
robot getting worse. Both causes here are real and both are the module's — +2.3 g, and
+2.75 mm of forward CoM.

It is flagged, not acted on. The forward CoM is worth a second look if the flat trot ever
becomes the number that matters: putting the BMS in the *front* zone instead of the rear
would take about 4 mm back out of it. It is at the rear because that is where the connector
panel is, and it is the only arrangement where the pack's leads and the BMS's leave by one
grommet slot without crossing the cells.

### The IMU move, measured rather than argued

`rl/checks/imu_placement.py` runs on the mac — it needs MuJoCo and the walker, not ROS — so
it was run here rather than left for the WSL2 box.

A board bolted where the model's site is reads **0.000 m/s², 0.0°**, which is the whole
point of the site living in `mini_dog.py`: the site moved with the board, so there is no
error to carry. What the move does change is the *observation itself*. Measured with
`--at 0,0,-7.6` — an accelerometer at the old site's height while the model's site is at
the new one — the difference is **p50 0.13, p95 0.60, max 4.28 m/s², worst apparent tilt
23.6°**.

Read that as a re-baseline, not a defect. It is the size of the step between the old
observation and the new one, so **any policy trained against the pre-2026-09-09 model is
trained on a different signal** and `rl/` has to retrain rather than fine-tune. For scale,
the same check puts a board left on the deck beside the Pi at 30.6 m/s² and 72.2°, and 50
mm straight up at 28.2 and 70.8° — the new mount is nowhere near that band, and it is
still on the centreline, which was the part of the original argument that mattered.


## 2026-09-09, later: the hip-roll cradles became bolted parts

The ask was maintenance: *"can we connect servo holders and rest of case with 4 bolts and
nuts? it's pain in the ass to reprint and rebolt servos"*. It started from a different
question — a report that the cutouts on the back of the dog were blocked — and the two
turned out to be the same defect.

**They were blocked, all of them, and nothing here could see it.** The two rear hip-roll
cradles met across the centreline and formed one continuous 2.8 mm plate at
x = −64.2…−67.0 over |y| ≤ 49.11 and |z| ≤ 15.36, with 1.2 mm of air behind the wall.
Measured on the solid: bus window 100 % blocked, XT30 100 %, balance lead 100 %, the
XT60's top 3.61 mm — and the XT mating halves stand 1.5 mm proud of x = −63, i.e. to
−64.5, already 0.3 mm *inside* the plate. Not one could ever have been plugged in.
`interference()` pairs the static body parts and the cradles **were** `chassis_bottom`, so
this was one part standing in front of its own hole.

### What was built

`cradle_front` and `cradle_rear`, one off each, each carrying both of that end's roll
servos, four M3 × 12 per end through a spigot register into a locally thickened tray wall.
One part per end and not four quarters because the two halves' rails overlap on the
centreline — four quarters are not four separable bodies. `chassis_bottom` goes from
213 × 110 × 50 mm / 208 g to **126 × 92 × 50 / 127 g**; the cradles are 38 g each.

The flange is a **frame**, and its opening is what lets the connector panel out. The panel
was relaid inside it: bus window 22 × 16 (was 32 × 20), XT30 at y = +32, balance at
y = −32, and the XT60 *above* the cradle at z = +20, because 16.5 mm of width does not
survive the window and the two bolt bosses.

Then the second half of the ask, raised mid-run: **the fork-access channels are not needed
any more.** They existed because the roll joint's inboard arm faced the chassis with
`FORK_GAP` of air and the fork could only go on after the cradle was welded to the tray.
With the cradle bolted, the fork goes on with it *in hand* and all four screws are in open
air — 0 mm³ against the cradle alone against 569 mm³ against the cradle on the tray. So the
two ⌀6 channels through the tray's front corner and the corner deck bosses are gone, eight
holes with them, and `fork_access()` reads *all six arms* rather than five. One feature
replaces them: `FORK_DRIVER_R`, the driver's swept circle relieved through the flange,
derived from `HUB_BC` and `DRIVER_D`. Without it the outermost screw at y = 42.9 fouls the
frame's rib at 43.11 by 38 mm³ — which is what the check read the moment the channels
stopped relieving it by accident.

### Three probes added, and all three earned their place on the first run

* `panel_clear()` — every rear-wall opening along −x against the assembled body. This is
  the one that found the defect above.
* `cradle_clear()` — a driver's run to each of the eight cradle screws. Found the
  ESP32/URT-1 divider rib standing 0.5 mm in the two lower rear screws' runs; its half
  width was `BMS_W`/2 + 1.5, and the BMS moved inside the battery module weeks ago, so the
  number was already stale. Now `ESP_RIB_Y`.
* `cradle_head_clear()` — the eight ISO 7380 heads against the battery module. Found them
  0.4 mm inside the pack: the front seat's face is at x = 57.4, the module's front at
  56.15, and a head is 1.65 tall. `CRADLE_CB` = 2 mm of counterbore; +1.60 mm now. Same
  class as the thrust clamp's cap head and the hub screws' heads — the part in the way is
  hardware, so `isValid()`, `interference()` and `rom_scan` are all blind to it.

### The ladder

| step | result |
|---|---|
| 0 baseline | `fea.py --all` on the unchanged tree, kept for step 4 |
| 1 `mini_dog.py` | every part valid, volumes positive; ROM unmoved at −90/+90, −90/+90, −110/+110; `body clear` (now nine parts), `imu clear +3.40`, `batt clear +1.40`, `module clear`, `foot bolt` ok, `clamp clear +1.43`, `head clear +0.65`, `fork access: all six arms`, `panel clear`, `cradle bolts … +1.60 mm of air to the pack` |
| 2 bboxes | `chassis_bottom` 213.0 × 110.2 × 50.0 → **126.0 × 92.0 × 50.0**; `cradle_front`/`_rear` 30.7 × 110.2 × 45.5. Nothing else moved |
| 3 render | three PNGs, looked at |
| 4 `fea.py --all` | no inter-layer SF dropped: `hip_bracket_A` 46.9 / 23.4 / 7.8 / 2.4, `thigh_A` 18.9 / 9.4 / 3.1 / 1.2, `shin_A` 60.1 / 30.1 / 10.0 / 4.9 — all within a rounding step of the baseline, marginally up on −6 g. **`cradle_front` is new in the set**: 4.0 / 2.0 / **0.7** / 3.0 |
| 5 `export_sim.py --check` | `4 feet down, upright +1.00`, base z 187 mm, 2.488 kg, terrain the same, camera axis (+0.99 +0.00 +0.10), urdf/mjcf leg mass agree, 13 STL |
| 6 ROS 2 | regenerated; only `base_link.stl` moved — no leg mesh changed |

### The gait, with a control beside it

| | control, unchanged tree, 2.493 kg | after, 2.487 kg |
|---|---|---|
| flat trot | 595.2 mm | 556.6 mm |
| terrain, seeds 7…12 | 498 ±81 mm | **520 ±67 mm** |
| course, seeds 7 / 8 / 9 | 5/7 2876, 3/7 2023, 4/7 2441 — all upright | 1/7 1506, 5/7 2872, 4/7 2582 — all upright |

The terrain sweep is the arm to read and its two means differ by 22 mm against spreads of
81 and 67 — one distribution. The flat trot moved 39 mm on −6 g, inside the band
`3d/CLAUDE.md` calls hypersensitive. The course is a report and it has to be read over
seeds: **an intermediate version of this change, 8 g lighter, read 1/7 and went DOWN on
seed 7** while its terrain sweep (501 ±89) was indistinguishable from the control. On one
seed that looks like a regression and is not. On three it is 5/3/4 against 1/5/4, all
upright, and the mean corridor is 2447 against 2320.

### The one thing that is not clean

**`cradle_front` is the weakest printed part on the robot.** Interlayer SF **1.35** at
`land3g` on `--orient`'s build-direction index, against 3.08 for `thigh_A`, 8.52 for
`hip_bracket_A` and 19.36 for `shin_A`. Read `--orient` and not the `SF xy / z` column —
the crude column assumes the worst stress orientation and says 0.7.

It is **not a regression**. This geometry was inside `chassis_bottom`, which `fea.py` has
never covered, so it had never been analysed at all — and the first run found two real
defects in it: a **1.4 mm notch** where the narrow strap stopped at x = 72.1 while the wide
one starts at 73.5, and a **27.6 mm step** where the width jumps from `STRAP_Y` = 21.5 to
49.11 in one plane. Both are re-entrant corners in the strap that carries the whole leg.
Closing the notch took the peak von Mises 53.7 → 26.4 MPa and the deflection 1.58 → 1.01
mm; tapering the step over `STRAP_TAPER` holds it. The part now reads 29.7 MPa at 1.00 mm,
mesh-converged (26.4 at 2.0 mm, 26.5 at 1.5).

The remainder is structural and has no cheap fix: everything the leg puts into the sleeve
funnels back to the flange through a section capped at `STRAP_Y`, and the space that would
widen it is swept by the fork over the ±90° roll ROM — a version with full-width straps
and only the arm's disc cut deflects 0.27 mm instead of 1.00, and reads `hip_roll
+0 .. +0`. **Three fixes were tried and all three made the peak worse**: filling the neck
solid (35.3 MPa), closing it with a `WALL` web (31.8), thickening the straps (29.5→). Do
not re-try them.

It is also printed in its *worst* build direction on purpose. `--orient` ranks flange-down
1.35, lying flat 1.67, on end 2.15; flange-down is the only one that puts the two servo
bores vertical, and a press fit wants that more than it wants the 1.67. On end is 110 mm
tall on a 263 mm² footprint and is not a real option.

`fea.py` grew a second traction patch for this part (`solve(..., load2_pred, force2)`): a
stall torque written as "a force at a lever about the joint axis" divides by zero when the
load patch is concentric with that axis, which the cradle's bore is, so its stall case is a
genuine couple. Its ground cases keep the whole bore — half a bore is a different, more
local load, and read 3 MPa hotter when the two were conflated.

## 2026-09-09, later still: the fitted actuator went into the model

`PLAN.md` step 3. Four constants in `mini_dog.py` section 4, all four now measurements off
a real ST3215 rather than the "plausible values" the block used to admit to:

| | was | now | where it came from |
|---|---|---|---|
| `MJ_KP` | 25.0 | **40.9** N·m/rad | friction-cancelled stiffness at 12 V |
| `MJ_FRICTIONLOSS` | 0.02 | **0.184** N·m | Coulomb, two routes agreeing (0.184 / 0.186) |
| `MJ_DAMPING` | 0.12 | **1.37** N·m·s/rad | the *total* speed-proportional torque |
| `MJ_ARMATURE` | 0.008 | **0.0165** kg·m² | free swing at the corrected `tau_c` |

`MJ_DAMPING` is deliberately the total (`b_v + k_u·k_e`) and not the viscous part: the
bench cannot split them — both cost a motor voltage proportional to ω and neither depends
on supply — and a MuJoCo `position` actuator has nowhere to put back-EMF anyway. Same
reasoning puts the Coulomb term in `frictionloss`, which is also the only place in this
project where STATIC friction can be modelled at all: `rl/actuator.py`'s `tanh(w/v_eps)`
is exactly zero at rest and cannot hold a joint (`PLAN.md` step 2b).

### The ladder

| step | result |
|---|---|
| 1 `mini_dog.py` | unchanged: every part valid, ROM −90/+90, −90/+90, −110/+110, `body clear` (nine parts), `imu clear +3.40`, `batt clear +1.40`, `module clear`, `foot bolt` ok, `clamp clear +1.43`, `head clear +0.65`, `fork access: all six arms`, `panel clear`, `cradle bolts +1.60` |
| 2 bboxes | unchanged — no geometry moved |
| 3 render | skipped, justified: no geometry moved |
| 4 `fea.py --all` | **skipped, and provably so.** `fea.py` reads no `MJ_*` (checked), the four constants appear nowhere in `out/bom.json`, and neither mass nor geometry changed. Strength cannot have moved |
| 5 `export_sim.py --check` | identical to the control: `4 feet down, upright +1.00`, base z 187 mm, 2.488 kg, terrain the same, camera axis (+0.99 +0.00 +0.10), urdf/mjcf leg mass agree, 13 STL |
| 6 ROS 2 | regenerated; **only `mujoco/defaults.xml` moved**, and only those four attributes — no mesh, no mass, no limit |

### The gait, with a control beside it

Control is the same tree with the old `MJ_*`, i.e. the numbers this file recorded a few
hours earlier for the cradle change.

| | control, old `MJ_*` | after, 2.487 kg |
|---|---|---|
| flat trot | 556.6 mm | **487.0 mm** |
| terrain, seeds 7…12 | 520 ±67 mm | **340 ±39 mm** |
| course, seeds 7 / 8 / 9 | 1/7 1506, 5/7 2872, 4/7 2582 | **2/7 1903, 2/7 1705, 0/7 1005** — all upright |

Mass is identical to the gram, so none of this is the mass cliff. The terrain sweep is the
arm to read and it moved **180 mm against a 32 mm standard error on the difference** —
about 5.6 σ, and the first re-baseline in this file that is *not* one distribution.

### Why that is the model getting honest, and not a regression

One number settles it. A joint can turn no faster than where the torque ceiling meets the
damping, `(forcerange − frictionloss)/damping`:

- old, 0.12 / 0.02 → **24.3 rad/s**
- new, 1.37 / 0.184 → **2.01 rad/s**
- ST3215 vendor no-load speed → 4.71 rad/s
- measured on the bench under the 1 kg arm → **~1.8 rad/s**

The old model let every joint swing **five times faster than the servo's own no-load
speed** and thirteen times faster than the bench can actually drive it, and the hand-tuned
gait had settled into exactly that headroom. The new value lands on the measured ceiling.
So the shorter distances are the legs no longer being allowed to do something the hardware
cannot, which is the whole point of fitting the actuator.

**What follows is a re-tune, not a revert.** The gait in `ros2/tools/standalone_sim.py`
is asking for swing speeds the servo does not have. Do not put `MJ_DAMPING` back.

One consequence worth stating plainly for `rl/`: any policy trained against the old model
learned to spend joint speed that does not exist, so it has to retrain rather than
fine-tune — the same conclusion the IMU move reached earlier today, for a different reason.
