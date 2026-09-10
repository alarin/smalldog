# CLAUDE.md

Parametric CAD for a 12-DOF printed quadruped ("mini robot dog") built around 12 ×
Feetech/Waveshare ST3215 bus servos. Pure code-CAD — there is no GUI model and no feature
history outside the scripts. `README.md` is the human-facing spec; read it before changing
geometry.

## Environment

Always use the project venv — CadQuery, gmsh, sfepy and VTK are installed there only:

```bash
.venv/bin/python mini_dog.py     # rebuild → out/step, out/stl, out/mini_dog_assembly.step, out/bom.json
.venv/bin/python render.py       # → out/view_{iso,front,side}.png
.venv/bin/python fea.py --all    # strength, every part, every load case
.venv/bin/python export_sim.py --check   # → out/sim/{mini_dog.urdf,mini_dog.xml,meshes/}
```

Run everything from the repo root; scripts use paths relative to the CWD (`ref/…`) or to
their own location (`out/…`).

## Layout

| path | role |
|---|---|
| `mini_dog.py` | **the model.** All geometry, all parameters, the ROM scan, the exporters. |
| `fea.py` | linear static FEA (gmsh + sfepy). Imports `mini_dog` — never duplicates dimensions. |
| `export_sim.py` | ROS 2 URDF + MuJoCo MJCF export. Imports `mini_dog` — every length,
  mass and joint limit is read from it. |
| `camera.py` | the IMX415 module **as a sensor**: MJCF `<camera>` + the two URDF frames, shared by both sim exporters. No CAD, like `lidar.py`. |
| `terrain.py` | procedural MuJoCo heightfield ground **and the ramp/wall/log obstacle course bedded into it**. Imported by *both* sim exporters; no CAD in it. |
| `lidar.py` | the Unitree L2 as a *sensor*: the MJCF site and `<custom>` numerics both sim exporters emit, and the `mj_multiRay` scanner that turns them into a point cloud. No CAD in it either. |
| `render.py` | offscreen VTK renders of `out/stl/*.stl`. |
| `bench_rig.py` | the printed fixture `robot/bench` runs on: a stand that holds **one** ST3215 with its axis horizontal, and two arms for its driven hub. Imports `mini_dog` one way only - the sleeve, the thrust clamp, the hub pattern, the densities - and writes `out/bench/`. **Not part of the robot**: no `PARTS` entry, no mass in the budget, no `fea.py` or sim consumer, so a change here needs none of steps 4-6 below. |
| `torque_rig.py` | the printed fixture that measures the servo's torque in **newton-metres** — a self-reacting C-frame with a kitchen scale in its throat. Imports `mini_dog` the same one way `bench_rig.py` does and writes `out/torque/`. **Not part of the robot**, so steps 4-6 below do not apply to it either. It exists because every torque measured so far is a torque per *register count*, and the two register channels agree with each other and disagree with the datasheet by 2.2x (`PLAN.md` step 2c); a scale is not a register. |
| `tools/` | one-off measurement/diagnostic scripts, not part of the build (see `tools/README.md`). |
| `ref/` | vendor downloads: ST3215 STEP/PDF/wiki, Waveshare ROBOTIC DOG STEP, and `camera/` - the IMX415 module's dimensions, transcribed, with the two uncertain readings flagged. Read-only inputs. |
| `out/` | **generated — never edit by hand, never treat as source.** |
| `../ros2/` | a *second* consumer of this CAD: `smalldog_description/scripts/generate_model.py` imports `mini_dog` and regenerates the whole ROS 2 description. Not optional — see step 6. |
| `mini_dog_codex_handoff.md` | the original design brief and its hard constraints. |

## How to change the model

Every dimension lives in the constant block at the top of `mini_dog.py` (lines ~28–65).
Change it there and re-run — do not patch a part function with a literal, and do not edit
STEP/STL in `out/`. `fea.py` derives its load cases and clamp regions from `mini_dog`'s
joint frames, so a parameter change propagates to the analysis for free; keep it that way.

Part functions (`chassis_bottom`, `hip_bracket`, `thigh`, `shin`, `foot`, `lidar_mount`,
`servo_gauge`) each return a CadQuery `Workplane` in **robot coordinates**. `PARTS` maps
name → (workplane, qty, note) and drives both the export loop and the BOM;
`PRINT_ORIENT` maps name → (axis, angle) applied only to the STL so it sits printable on Z=0.

## Invariants — do not break these

- **Units are mm** everywhere: geometry, FEA (N, MPa), exports.
- **Servo interface is measured, not assumed.** The numbers in the `S_*` / `HUB_*` block
  come from `ref/ST3215-3D/ST3215.step` via `tools/measure*.py`. If one looks wrong,
  re-measure — do not adjust it to make a part fit. **And the STEP is not the last word:**
  `HUB_BOLT_D` is M3, measured on a hub that arrived, against the ⌀2.5 the STEP shows.
  A vendor number that a real part contradicts loses; mark it in the README's source
  column so the next reader knows which rows have met hardware.
- **The ST3215 case has no threaded side holes.** The whole sleeve+fork architecture
  exists because of that. Nothing may load the servo through a printed thread or a
  single-shear horn.
- **No metal parts, no machining, no external bearings** (628/685 etc.). FDM only, plus
  the stock aluminium hubs. This is a hard constraint from the handoff doc.
- **Nothing in a torque path threads into plastic.** Every screw that carries load —
  every joint, fork, hub, sleeve and foot — and is not going into the stock aluminium
  lands in a nut, and every one of those nuts sits in a `nut_slot()` — a channel of the
  nut's across-flats width, so its two walls stop the nut turning. When you add one,
  the `ang` argument is not cosmetic: it has to point at a face that is still reachable
  at the moment in the assembly order when that nut goes in, and that is what fixes the
  assembly order in `README.md` (the LiDAR pedestal has to be bolted to the deck before
  the deck goes on the tray, because its screw heads end up inside the tray). At the
  servo hubs the screw threads into the stock aluminium plate (both are tapped M3) —
  there is no room for a nut there, and a hex pocket would eat over half the fork arm
  right under the screw head.
- **Off the torque path, a screw may thread straight into the plastic** — a case lid, an
  electronics cover, a cable clamp, a bracket that holds nothing but itself. Nothing the
  legs load qualifies, and the test is not "is it small", it is "does a load path run
  through this screw": if the answer is yes or you are unsure, it gets a `nut_slot()`.
  The hole is then **not** `M3_CLR`/`M25_CLR` — a thread-forming screw needs a *smaller*
  hole, `M3_TAP` / `M25_TAP` in the same constant block (⌀2.6 / ⌀2.1, and they are radii
  like the clearance pair), with ≥ 1 × D of boss wall all round and 2 × D of engaged
  length. Cut them with the constants, never a literal, and never re-tap the same hole
  twice — a formed thread in FDM plastic is a one-assembly thread, so a lid that comes
  off repeatedly still wants a nut or an insert. Both radii are **UNVERIFIED**: they are
  the standard band for the thread minus what an FDM hole loses, not a measurement, so
  print a coupon before the first part depends on one. Added 2026-09-09.
- **A blind fastener path is invisible to every check in this repo, so the foot bolt has
  its own.** `isValid()` sees nothing wrong with a bolt hole that never breaks the
  surface, `interference()` only looks at the static body parts, and `rom_scan` only
  looks at what moves — so `foot()` shipped from the start with its ⌀3.4 hole cut from
  `zf-6` and its head pocket from `zf-1`, both written as if the dome's radius were 6
  rather than `FOOT_D/2` = 13. The result was **7 mm of solid TPU under the entry**: the
  hole opened inside the part, and the pocket's one annulus faced *up*, so even a bolt
  that could get in had nothing to pull against. It was specified as M3 × 16, and the
  span from the sole to the nut's far face is 28 mm — no head position could ever have
  reached, which is the tell that the number was never checked against the solid.
  `foot_bolt_check()` now probes the real solid along the axis and `build()` prints a
  `foot bolt:` line; treat `!! FOOT BOLT` as a failure like `!! INTERFERENCE`.
  **The same hole in the checks has now cost three fasteners, so there are three probes.**
  `thrust_clear()` is the clamp screw against the annulus the distal fork sweeps, and
  `fork_access()` is a key-sized cylinder on each fork screw's real line, run **until it
  stands in open air**, against the part the fork bolts onto. All six arms take a straight
  key now, and the thing that made the roll pair reachable was not a hole, it was the
  **assembly order**: the cradle is a bolted part, so the fork goes on with it in hand and
  those four screws are in open air (0 mm³ against the cradle alone, 569 against the cradle
  bolted to the tray). One feature serves it — `FORK_DRIVER_R`, the driver's swept circle
  relieved through the flange — and it is derived from `HUB_BC` and `DRIVER_D`, not typed.
  **Two earlier answers are deleted and both are worth remembering.** Four coaxial ⌀6 bores
  into the tray reached one screw of four and passed anyway, because the probe stopped
  inside the bore; a probe has to end in air, not in a hole. Two ⌀6 channels leaning 20°
  through the tray's front corner then *worked* — and still cost the chassis eight holes,
  through the corner posts and the corner deck bosses. An access problem that keeps needing
  cleverer holes is usually telling you the part boundary is in the wrong place. Deleted
  2026-09-09; `README.md`, "Reaching the fork screws", keeps the measurements.
  When you add a fastener, ask the three questions separately: does the hole reach,
  does the head clear, and can a driver get to it. The
  length follows from `FOOT_CB_Z`, `FOOT_NUT_Z` and `M3_NUT_H` — if you move any of
  those, `FOOT_BOLT_L` and the BOM line in `README.md` move with them. Fixed 2026-08-31.
- **The hip-roll cradles are BOLTED parts, not part of the tray, and the joint is four
  screws plus a register.** `cradle_front` and `cradle_rear`, one off each, each carrying
  both of that end's roll servos - one part per end because the two halves' rails overlap
  on the centreline, so four quarters are not four separable bodies. This took
  `chassis_bottom` from 213 x 110 x 50 and 208 g to 126 x 92 x 50 and 127 g, and it is what
  makes a tray change stop meaning "reprint four cradles and unbolt four servos".
  **The register is not decoration**: four M3 carry 22 N of axial and 28 N of shear at
  worst, which is nothing, but they do not replace 795 mm2 of welded face for STIFFNESS -
  the roll axis is 27 mm outboard of the joint and the foot 187 mm below, so a tenth of a
  millimetre of slop is 0.7 mm at the ground. Each screw runs through a `CRADLE_REG_D`
  spigot into a pocket in a locally thickened tray wall. **Where the screws may go is
  decided by the fork**: the flange is only `CRADLE_T` = 4 mm deep (its outer face is
  `FORK_GAP` behind the fork's rear arm) and an M3 nut does not fit in 4 mm, so the nut
  bosses run out to `CRADLE_BOSS_X` - which is legal only because past that face the one
  thing in the way is the arm's `ARM_R` disc, the spine's 23...34 annulus being swept only
  on the outboard side. `rom_scan` checks the arm, `cradle_clear()` checks a driver reaches
  the heads, `cradle_head_clear()` checks the heads against the battery module. A version
  that ran the straps full width and cut only the arm's disc read `hip_roll +0 .. +0`;
  `STRAP_Y` is a hard ROM limit, not a guess. Added 2026-09-09.
- **`cradle_front` is in `fea.py`'s set and it is the weakest printed part on the robot.**
  Interlayer SF **1.35** at `land3g` on `--orient`'s build-direction index, against 3.08
  for `thigh_A`, 8.52 for `hip_bracket_A`, 19.36 for `shin_A`. Read `--orient`, not the
  `SF xy / z` column, for this part: the crude column assumes the worst stress orientation
  and says 0.7. **It is not a regression** - this geometry was inside `chassis_bottom`,
  which `fea.py` has never covered, so nobody had ever looked, and the first run found two
  real defects in it (a 1.4 mm notch in the strap and a 27.6 mm step in its width; peak
  53.7 -> 29.7 MPa). The rest is structural and does not have a cheap fix: everything the
  leg puts into the sleeve funnels back through a section capped at `STRAP_Y`, and the
  space that would widen it is swept by the fork. Filling the neck solid, closing it with
  a web and thickening the straps were each tried and each made the peak WORSE - do not
  re-try them without reading this line. It is also printed in its WORST build direction
  on purpose: flange-down is the only one that puts the two servo bores vertical, and a
  press fit wants that more than the 1.67 lying flat would score.
- **`solve()` in `fea.py` takes a second traction patch, and only one part uses it.** A
  stall torque expressed as "a force at a lever about the joint axis" divides by zero when
  the load patch is CONCENTRIC with that axis, which the cradle's bore is. Its stall case
  is a real couple - `couple=(pred1, pred2)` in `part_specs()`. Its ground cases keep the
  whole bore: putting a ground reaction on half of it is a different, more local load and
  it read 3 MPa hotter when the two were conflated. Added 2026-09-09.
- **Nothing goes into 23 < r < 34 of a joint axis over the sleeve's length.** The distal
  fork's spine sweeps that annulus, and the hip bracket's inboard web already comes to
  r = 22.0. It is the binding constraint on the sleeve thrust clamp, and **screws count**:
  the lug corner sits at 21.6 and always cleared, but an M3×10 cap head reaches 24.2 and
  bound the first assembled leg over ±2…38° of travel. The fastener is specified headless
  for that reason — `THRUST_HEAD_*` — and `thrust_clear()` prints the margin on every run
  (`clamp clear:`), with `thrust_bolts()` in the static side of all three `rom_scan` calls.
  A check over the printed solids alone cannot see this: the part sticking out is hardware.
  The hub screws were the same defect a second time: an ISO 7380 head is 1.65 mm tall and
  the hip's inboard arm had 0.6 mm of air to the gusset. `fork_screws()` now puts all
  eight heads on the moving side of every `rom_scan`, the passive arm is counterbored
  `FORK_CB` against `FORK_GAP`, and `head_clear()` prints the head's air and the tip's
  distance to the case on every run. Found 2026-09-04, when the screws arrived.
- **Do not slit the sleeve.** Its `-x` end wall is the tube's only crossing of y=0 — the
  cable window has already eaten the `+x` one — so a C-clamp slit there opens the whole
  sleeve-plus-link box section. This was tried and measured: `thigh_A` inter-layer SF at
  stall 1.1 → 0.6, deflection 2.65 → 12.2 mm, and tying the `+x` wall back across the
  window recovered none of it (0.5). The clamp that survives is the one that adds material
  and takes the play out in thrust (`THRUST_*`), not the one that cuts the ring.
- **The camera's slot at the nose is four measured walls, and three of them belong to
  other parts,** and since 2026-09-09 one of them is a BOLTED part: `CAM_LEDGE` is
  `CRADLE_Z1`, so the camera's floor and its two nuts are on `cradle_front`, not on the
  tray. Floor `CAM_LEDGE` = 15.36 (the front cradle's own top face),
  ceiling the LiDAR pedestal's base disc at z = 29 (hence `LIDAR_BASE_FLAT`), back the
  chassis front face at x = 63, front the hip-roll fork's rear arm at
  `ROLL_X + FORK_Y0` = 68.1 — a disc that sweeps r <= 34 about the roll axis over the whole
  roll ROM. Only the lens may go past 68.1, and only because on the centreline it stays
  36.3 mm from either roll axis. Moving `LIDAR_BASE_R`, `LIDAR_X`, `SLEEVE_W`, `ARM_T` or
  `roll_module`'s rails moves one of those walls; `interference()` and `rom_scan` will say
  so, but the fix is in the camera block, not in a part function. A 90 x 15 mm board in a
  5 x 15 mm slot has ~1 mm everywhere: treat every one of these as load-bearing.
- **Every opening in the rear wall opened into solid plastic, and no check could see it.**
  The two rear cradles met across the centreline as one continuous 2.8 mm plate 1.2 mm
  behind the wall: bus window 100 % blocked, XT30 100 %, balance lead 100 %, the XT60's top
  3.61 mm - and the XT mating halves stand `PANEL_LIP_T` proud of the wall, i.e. already
  0.3 mm INSIDE the plate. `interference()` pairs the static body parts and the cradles
  WERE `chassis_bottom`, so this was one part standing in front of its own hole; a part
  cannot interfere with itself. The block that positioned the panel reasoned about deck
  bosses 30 mm further away, and was stale about those too. `panel_clear()` now probes
  every opening along -x against the assembled body and prints on every run; treat
  `!! PANEL` as a failure like `!! INTERFERENCE`. The layout is now inside the cradle
  flange's frame opening and around the four bolt bosses, and **the XT60 does not fit
  there** - it goes above the cradle at z = +20, in the 9.64 mm band between `CRADLE_Z1`
  and `BODY_Z1`, which is why its z is not a round number. Fixed 2026-09-09.
- **The servo envelope must not be cut from the link that bolts to that servo's hubs.**
  `servo_envelope(hub=False)` exists for exactly that case, and each forked part passes
  its own joint to `env_all(no_hub=...)`. Sweeping the hub discs out of a part that has
  to sit on them is how all three forks once ended up with no bolt circle at all.
- **Masses and densities live once, in `mini_dog.py` section 4.** `fea.py`, `export_sim.py`
  and `../ros2/.../generate_model.py` all read them from there and keep no copies. They
  used to, and the servo mass silently diverged (55 g here, 60 g in the ROS 2 model — 60 g
  of robot). Anything the robot carries belongs in that block, including payload that has
  no printed part: the ground load cases in `fea.py` scale with the total.
- **The MuJoCo joint feel lives there too** — `MJ_DAMPING`, `MJ_ARMATURE`,
  `MJ_FRICTIONLOSS`, `MJ_KP`, `MJ_DAMPRATIO`, in the same section 4 block, read by
  both sim exporters. They were duplicated once and diverged, exactly like the servo
  mass above; `rl/checks/check_model.py` is what caught it. Three of the five are
  estimates and say so, but `MJ_ARMATURE` = 0.008 is not: it is the ST3215's
  reflected rotor inertia, the same number as `rl/actuator.py`'s `Params.J_m`. When
  the bench fits the real actuator, these become its initial guess — never a second
  opinion sitting beside it.
- **The foot's contact lives there too, and it was the same defect a fifth time** —
  `MJ_FOOT_CONDIM` = 4, `MJ_FOOT_FRICTION` = (1.2, 0.002, 0.001), `MJ_FOOT_PRIORITY` = 1,
  read by both exporters. They had diverged (torsion 0.05 in `export_sim.py` against 0.02
  in the ROS 2 generator) while *both* shipped MuJoCo's default `condim="3"`, which reads
  only the first friction column — so the torsion number was in both files, different in
  each, and used by neither, and a planted foot was a frictionless pivot. **It is 4 and
  not 6:** condim 6 adds the rolling dimension and takes the terrain sweep from 657 ±35 mm
  to 615 ±124, and that is the dimension and not its coefficient — condim 6 with roll set
  to zero still reads 612 ±72, against condim 4's 653 ±28. Do not "upgrade" it. The
  torsion figure is derived from the real patch (⅔·a·μ·f_n, a = 1.4…2.1 mm at 4 MPa in
  95A TPU), not chosen. **And check a contact model by reading `d.contact[i].friction`,
  never the XML**: MuJoCo uses the elementwise *max* of the pair unless one geom has
  priority, so after the constants agreed the two exporters still met the ground at
  0.02 against 0.005, because their floors differ. `MJ_FOOT_PRIORITY` is what makes the
  foot's own numbers win — `solref`/`solimp` included, which is why this re-baselined
  step 6 below. Fixed 2026-09-08. `ros2/tools/foot_contact.py` is the measurement, and
  `ros2/README.md`'s "The foot's contact patch" is the write-up, including the three
  ways of growing the actual patch that were built, measured and **not** adopted.
- **The servo's stall torque and no-load speed are read from here too**, by both
  exporters — `SERVO_STALL_NM` = 2.94 and `SERVO_NOLOAD_RADS` = 4.71. This was the
  same defect a third time and the worst-stated of the three: `export_sim.py` read
  them, `generate_model.py` kept rounded copies (`J_EFF = 3.0`, `J_VEL = 4.7`), so
  the two exporters were not duplicating a constant, they were emitting **robots with
  different servo strength** — and which sim you loaded decided how strong the servo
  was. Fixed 2026-08-31 with the re-baseline in the same commit; `rl/` loads the ROS 2
  model, so that was the one that mattered.
- **The cells are held on a pitch by two printed combs, and the combs cost the module
  nothing but that pitch.** `cell_holder`, x2, one at each end of the cells: they exist
  because welding six loose cylinders into a brick by hand is the assembly step this
  design had no answer for. The obvious holder - a frame round the outside of the array -
  does not fit, and that is measured, not felt: the module had 1.4 mm to the deck and its
  side walls are 0.5 mm off the deck screws' nut bosses, whose inner faces stand at
  |y| = 35.2 over the module's whole height. So the combs are **clipped flush with the
  outermost cells' own tangent planes in both y and z** - no material outboard of any cell
  anywhere - and the module grows by the separator gaps alone, +0.8 in z and +1.6 in y.
  The cells stay captured: an 0.8 mm collar clipped at the cell's tangent leaves an 8.4 mm
  flat and a 21.3 mm cell does not leave through 8.4 mm. `CELL_GAP` is the only number
  that costs height (0.8 -> deck gap 0.60; 0.6 -> 0.80) and it is NOT a print wall - the
  printed web between two bores is `CELL_GAP - CH_FIT` = 0.6. `holder_clear()` probes both
  caps against the case and against all six cells, because the cells are a payload and the
  holder lives inside another part's cavity, so neither `interference()` nor `isValid()`
  can see a cap drawn half a millimetre too wide. **And the part came back as five loose
  pieces the first time** - the two clip planes meet at each corner cell and orphan a
  crescent of collar - with `isValid()` and `Volume() > 0` both passing, which is why
  `build()` now checks each part's SOLID COUNT against `PART_SOLIDS` (default 1; three
  existing parts are legitimately several bodies on one plate and are declared there).
  Added 2026-09-10.
- **The battery is a MODULE, and the tray is now its.** Six cells in two `cell_holder`
  combs, welded into a 3 x 2 brick, heatshrunk over cells AND holder, its BMS beside them,
  all of it in `battery_case` + `battery_lid`,
  which drop into a seat recess in the tray floor as one payload. The old cradle - four
  printed fins and two end stops with six loose cells between them - is gone, and so is
  every check's blindness to it: cells were a payload, so `interference()` could not see
  them, and nothing at all could see a welded tab chafing against a printed edge. The case
  IS a part, with a `PARTS` entry and a place in `BODY_PARTS`, so `interference()` now
  covers the pack against the chassis for the first time. What it is NOT is structure:
  nothing on the robot loads it, which is why its lid screws are one of the only two places
  a screw threads into plastic. **A case costs about 3.5 mm the bay never had**, and there
  was exactly one place to take it from - the IMU's slot - which is why that block moved in
  the same commit. Three knock-on constraints came with it and are all load-bearing: the
  module reaches x = -46.15, so the ESP32/URT-1 bay moved to `ESP_X` = -51 and `BATT_X` was
  pushed forward to 5.0 to keep that strip at 8.9 mm; the four corner deck screws moved
  from |y| = 38 to 41, because at 38 their bosses took a 1.9 mm bite out of the module's
  corners; and `DECK_SCREWS` now carries its nut-channel direction explicitly, because |y|
  no longer tells the two pairs apart. `batt_clear()` prints the air over the lid on every
  run - it is 1.4 mm, it is where a foam strip goes, and it may never be negative.
  **And the two things INSIDE the module are payloads, so they get their own probe**, the
  same way the foot bolt and the IMU do: `module_clear()` intersects the case and the lid
  against `brick_solid()` and `bms_solid()`, and both must be zero. It is not decoration -
  it caught two defects on the day it was written. The lid was first drawn with a locating
  lip round the whole opening, 2 mm deep, into an interior the wrapped brick fills to
  `BATT_FIT`/2 = 0.2 mm a side; and the BMS's retaining ribs stood 1.2 mm inside the brick.
  `isValid()` was happy with both, `interference()` cannot see a payload, and the parts
  would have printed. The lid now hangs one bar, over the BMS zone only, where there is
  room, and `BMS_GAP` is the number that has to swallow the rib as well as the air.
  Added 2026-09-09.
- **The IMU is on top of the deck, not under it, and that was the battery's doing.** It
  used to hang from two tabs in the deck's window in the 3.6 mm over the pack. It now sits
  on two standoffs on the deck's TOP face, on the centreline, inside the Orange Pi's own
  7 mm standoff gap, and the deck's cable window moved off the centreline (`IMU_WINDOW`)
  to leave solid deck under it. `imu_clear()` still exists but looks UP now, and measures
  against the Pi's BOARD rather than `OPI_BOX`, whose floor is the deck itself - the board
  shares that envelope's standoff gap on purpose. **The site rose from z = 23.4 to 31.0**,
  and `rl/checks/imu_placement.py` was RUN rather than argued from - it needs MuJoCo and
  the walker, not ROS, so it runs on the mac. A board bolted at the site reads
  0.000 m/s2, which is the point of the site living here. What the 7.6 mm changes is the
  OBSERVATION: `--at 0,0,-7.6` measures p50 0.13, p95 0.60, max 4.28 m/s2, worst apparent
  tilt 23.6 deg between the old height and the new one. **That is a re-baseline for `rl/`,
  not a defect** - a policy trained against the pre-2026-09-09 model saw a different signal
  and has to retrain, not fine-tune. For scale, the same check puts a board out on the deck
  beside the Pi at 30.6 m/s2 and 72.2 deg, and one 50 mm straight up at 28.2 and 70.8 - the
  new mount is nowhere near that band, and it is still on the centreline, which was the part
  of the original argument that mattered. Re-run it whenever this block or the gait moves.
  Moved 2026-09-09.
- **The IMU's mounting point is a model constant, not a mount detail** — `IMU_*` in
  section 3, reached through `imu_xyz()`, and both sim exporters read the `imu` site from
  there. This was the same defect a **fourth** time and it had been shipping: the ROS 2
  generator wrote `pos="0 0 0"` while `export_sim.py` wrote `BODY_Z1`, so the two files
  described robots whose accelerometers sat 25 mm apart — and `rl/` loads the ROS 2 one, so
  that offset went straight into the observation the policy trains on. What makes this one
  worse than the three above it is that no check could have caught it from inside `3d/`:
  it took `rl/checks/imu_placement.py`, which adds real accelerometers at candidate mounts
  through `MjSpec` and compares them against the site the model ships. Its number is the
  argument for the position, not a warning to note: ω × (ω × r) + α × r reaches 9.0 m/s²
  — 42° of apparent tilt — on the existing trot for a board out by the Pi, against 25° on
  the centreline. Re-run it whenever the mount or the gait moves, and never let the site
  and the board be chosen in different files again.
- **The IMU board's clearance is a payload gap, and interference() cannot see it.** It was
  the 3.6 mm slot under the deck until 2026-09-09; the board now sits on the deck's top
  face inside the Orange Pi's 7 mm standoff gap, so the wall that is thin is the Pi's own
  board 3.4 mm above it. Neither the Pi nor the pack is a part, which is why `imu_clear()`
  exists the way `gps_clear()` exists for `OPI_BOX`. Note what it measures against: the
  Pi's BOARD, one `OPI_STAND_H` above the deck, not `OPI_BOX`, whose floor is the deck
  itself and whose standoff gap the IMU now shares on purpose. The board and its
  components are 2.8 mm, which is why `IMU_STACK` says headerless and why
  `ref/imu/README.md` says solder to the pads. Moving `DECK_T`, `OPI_STAND_H`,
  `IMU_STAND_H` or `IMU_WINDOW` moves one of its walls; the build prints the remaining gap
  on every run, and it is allowed to be small but never negative.
- **A boolean that fails on a degenerate contact comes back INVERTED, not broken, and
  `isValid()` still says True.** This is the OCC silent-failure note above generalised off
  the shin lofts, and it shipped: `fork_access_bores()` - long since deleted, see the
  fastener-probe note above - cut its four @6 bores before the
  rear connector pads were unioned on, and those pads start at exactly the plane the bores
  end on - `xw+WALL` = -60.2 against the bores' `BODY_L/2-WALL` = 60.2. They share **no
  volume at all**; the pad's inner face is welded straight onto the bore's circular
  opening, and that contact alone was enough. `chassis_bottom` came back at **-257 mm3**,
  a 2.5 x 14.9 x 6.9 mm sliver where 240 cm3 should be, and because the volume was negative
  rather than the shape invalid *nothing downstream noticed*: `isValid()` passed, the ROM
  scan read `hip_roll +0 .. +0`, every `interference()` pair fired at once, and the total
  mass came out 280 g light. The cure is ordering - **cut a through-path after everything
  that adds material near it**, not before - and `build()` now fails a part whose volume is
  not positive, which is the check that names it in one line instead of five confusing ones.
  Read `!! INVALID` on a part you have just given a new hole as "a boolean inverted", not
  "the geometry is subtly wrong". Fixed 2026-09-03.
- `build()` checks `shape.isValid()`, **`Volume() > 0`** and **the solid COUNT** per part
  (`PART_SOLIDS`, default 1); an `!! INVALID` line in the output is a failure, not a
  warning. The count is the newest of the three and it exists because `cell_holder` came
  back as five bodies with the other two checks green - a boolean can orphan a piece
  instead of inverting, and a slicer will happily print the debris beside the part.
- **Every model change is re-checked for strength.** Any edit to `mini_dog.py` — a constant
  in the parameter block just as much as a part function — is unfinished until
  `.venv/bin/python fea.py --all` has been run and its safety factors compared against the
  numbers from before the change. A parameter you think is "cosmetic" still moves the load
  path: wall thickness, joint offsets and link lengths all change the moment arms `fea.py`
  derives from the joint frames. The same edit invalidates `out/sim/` **and the ROS 2
  package in `../ros2`** — regenerate both in the same pass (steps 5 and 6 below), so the
  printed, analysed and simulated robots never disagree. A stale `../ros2` is the easy one
  to forget: it lives outside this repo, nothing here imports it, and its meshes keep
  rendering happily with whatever geometry they were baked from.
- **The LiDAR's parameters live in `mini_dog.py` and travel inside the model.** The
  sensor's cone, point rate, frame rate, range and noise are in section 3's LiDAR block —
  **including `LIDAR_FRAME_HZ`, which used to sit in `lidar.py` as a "sim choice" and is
  not one**: the real L2 emits 12.0 clouds/s, so it is the sensor's number and it lives
  with the sensor's other numbers. `lidar.py`'s file-local constants are the ones that are
  still genuinely this-file's-guess (the Risley spin rates), and that is the line between
  the two files. The point rate was the same defect a degree worse: 21600 /s was catalogue,
  the real sensor does **62341 /s at 12 Hz**, and being 2.9x low is the direction that
  flatters the model. Measured 2026-09-05, method and capture in `ref/lidar/`; `lidar.py`
  writes them into every MJCF as `<custom><numeric name="lidar_*">`, and both consumers -
  `lidar.Scanner` and the C++ `MujocoLidar` in `../ros2/src/mujoco_ros2_control` - read
  them back out of the *compiled* model. Do not add a launch parameter, a YAML or a
  `robot_params.json` entry for any of them; the C++ node cannot import Python, and the
  model file is the one thing both sides already load. A model with no `lidar_*` numerics
  has no LiDAR, and both consumers say so instead of inventing a cone. The scan *pattern*
  is the single exception: it exists twice, in `lidar.directions()` and in
  `mujoco_lidar.cpp`, because neither can call the other. Change both, in the same pass.
- **The sim model is generated, never hand-tuned.** `out/sim/*.urdf|.xml` is output, like
  the STEPs. A link mass, a joint limit or an axis that is wrong in simulation is wrong in
  `mini_dog.py` — fix it there and re-export. Masses, inertias and limits come from the
  real solids and the ROM scan; only the MuJoCo actuator feel (`MJ_*`) and the collision
  primitives are estimates, and they are flagged as such in `export_sim.py`. The same holds
  for `../ros2/smalldog_description/{meshes,urdf,mujoco,robot_params.json}`: every one of
  those files is output of `scripts/generate_model.py`, which imports this `mini_dog.py`.
  Never hand-edit them, and never let a model edit end without re-running that script.
- The Waveshare `ROBOTIC DOG.step` supplies the **shape** of the shin (`SHIN_PROFILE`,
  measured by `tools/ref_ws_shin.py`) and nothing else. It is an aluminium-plate,
  single-shear-horn design: do not transfer its joint spacing or its absolute sections.

## Verifying a change

1. `.venv/bin/python mini_dog.py` — all parts valid, and check the ROM table it prints.
   Hip pitch collapsing to ±15° means the pitch axis lost its 30 mm drop below the roll axis.
   The run also ends with either `body clear:` or one `!! INTERFERENCE` line per pair of
   `chassis_bottom` / `chassis_top` / `lidar_mount` / `gps_mount` / `camera_mount` /
   `battery_case` / `battery_lid` / `cradle_front` / `cradle_rear` — and the camera module
   and the IMU board, which are not
   printed parts but are bolted to the same body — that
   share more than 1 mm³ of solid, and with `gps clear:` — the GPS mast against the Orange
   Pi's `OPI_BOX` envelope, which `interference()` cannot see because the Pi is a payload
   and not a part — `imu clear:` (the board against the Pi above it), `batt clear:`
   (the battery module's lid against the deck), `holder clear:` (both `cell_holder` caps
   against the case and against all six cells), `panel clear:` (every rear-wall opening
   against the assembled body) and `cradle bolts:` (a driver's run to each of the eight
   cradle screws, and the heads against the pack).
   That is a failure, not a warning — `isValid()` never sees it, and `rom_scan` only covers
   the parts that move. Keep it cheap: three static solids, no sweep.
2. Mass and print bbox in the same table: parts must fit a normal 256 mm bed.
3. `.venv/bin/python render.py` and actually look at the three PNGs.
4. **Strength, every time** — `.venv/bin/python fea.py --all` (covers `hip_bracket_A`,
   `thigh_A`, `shin_A` and `cradle_front` over all four load cases). Run it *before* the change too, or keep
   the previous run's output, so there is a baseline to compare against. Judge on the
   **inter-layer** SF (the second of the `SF xy / z` pair) — that is the one FDM parts
   actually fail at. Any part whose inter-layer SF drops is a regression: fix it or report
   the before/after numbers explicitly, do not just note that the parts are still valid.
   Run `--selftest` first if you touched `fea.py` itself, and `--orient` if you changed a
   part's `PRINT_ORIENT` entry. **For `cradle_front`, read `--orient` and not the SF
   column** — see the invariant above; the numbers are 1.35 there against 0.7 here, and
   the difference is that the SF column assumes the worst stress orientation.
   Baseline, 2026-09-10, at 2.46 kg, inter-layer: `hip_bracket_A` 46.8 / 23.4 / 7.8 / 2.4,
   `thigh_A` 18.8 / 9.4 / 3.1 / 1.2, `shin_A` 60.0 / 30.0 / 10.0 / 4.9,
   `cradle_front` 4.0 / 2.0 / 0.7 / 3.0. Three of those moved by 0.1 against the
   2026-09-09 set (46.9 / 18.9 / 60.1) and it is the cell holder's +7 g of robot mass
   arriving in the ground load cases, at safety factors of 19 to 47 - a 0.2 % move, and
   the same 0.2 % appears in `stand2` and nowhere structural. Reported rather than
   glossed because the rule above says a dropped inter-layer SF is a regression until it
   is explained.
5. **Re-export the sim model, every time** — `.venv/bin/python export_sim.py --check`.
   It rebuilds `out/sim/` and then loads both files in MuJoCo and stands the robot up for
   3 s: `4 feet down, upright +1.00` and a base height near the CAD stance is the pass.
   The `lidar` line below it scans one frame off the standing robot; the number to watch
   is the near edge of the cone on the centreline, which is what `LIDAR_TILT` buys and
   which moves whenever the pedestal, the tilt or the stance does.
   The `camera` line renders one frame from the model's own `<camera>` to
   `out/sim/camera_view.png` and reports the axis in world coordinates and how much of the
   frame is the robot itself; an axis that is not ~(+0.99 0 +0.10) means `camera.py`'s
   quaternion is wrong, which is the one bug a plausible-looking pose and fovy will hide.
   The `terrain` line right below it stands the same robot on `mini_dog_terrain.xml`
   (`terrain.py`'s heightfield); it spawns on the flat pad, so it should read the same.
   `!! it fell over`, `!! the two files disagree` or a link mass that jumped is a
   regression in the model, not in the exporter. It also cross-checks its own total
   against `fea.robot_mass()`.
6. **Regenerate the ROS 2 description, every time** — there are *two* exporters of this
   CAD, and this one lives outside the repo:

   ```bash
   cd ../ros2 && ../3d/.venv/bin/python smalldog_description/scripts/generate_model.py
   ../3d/.venv/bin/python tools/standalone_sim.py --headless
   ../3d/.venv/bin/python tools/standalone_sim.py --headless --terrain
   ../3d/.venv/bin/python tools/standalone_sim.py --course
   ```

   `standalone_sim.py --lidar` adds the L2 to any of those runs (a summary in the headless
   ones; `../ros2/tools/view.sh --terrain --lidar` draws the cloud in the viewer, and that
   wrapper exists because the passive viewer needs `mjpython` and `mjpython` cannot find
   this venv's libpython on its own); it changes no dynamics - verified, the 5 s trot comes
   out at the same millimetre with and without it - so it is a diagnostic, never the
   regression itself.

   **Re-baselined 2026-09-09 by the cradle split.** Same seeds, same day, the unchanged
   tree beside the changed one, at 2.493 kg control against 2.487 kg after:
   **flat 595.2 mm control / 556.6 after**; **terrain seeds 7..12 498 ±81 control /
   520 ±67 after** — 22 mm of means against spreads of 81 and 67, i.e. one distribution,
   and that is the arm to read; **course seeds 7/8/9 5/7, 3/7, 4/7 all upright control /
   1/7, 5/7, 4/7 all upright after** (corridor 2876/2023/2441 against 1506/2872/2582).
   Read the course over seeds or not at all — an INTERMEDIATE version of this change, 8 g
   lighter, read 1/7 and went DOWN on seed 7 while its terrain sweep was indistinguishable
   from the control, which is exactly the trap the paragraphs below describe.
   The 2026-09-08 numbers, superseded: flat 790.8, terrain 537 ±67, course 5/7 2778.

   **Re-baselined 2026-09-10 by the cell holder**, which is a MASS change and nothing else:
   no joint, limit, actuator constant or leg geometry moved, and the flat trot did not
   move either. 2.486 kg control against 2.493 kg after (+7 g: 4.6 g of `cell_holder`
   pair, 2.4 g of the wider/taller `battery_case`).

   | | control, unchanged | control **+7 g dead mass** | after |
   |---|---|---|---|
   | flat trot | 481.4 mm | 480.8 mm | 482.8 mm |
   | terrain, seeds 7…12, upright seeds | 350 ±31 mm, **0/6 down** | 343 ±16 mm, **1/6 down** (seed 7) | 357 ±16 mm, **1/6 down** (seed 8) |
   | course, seeds 7 / 8 / 9 | 2/7 1839, 2/7 1820, 2/7 1797, all upright | 0/7 382 **down**, 2/7 1747, 1/7 1082 | 1/7 1634, 0/7 419 **down**, 0/7 1010 |

   **The middle column is the whole point and it is the method this file already
   prescribes.** 7 g parked in `ELECTRONICS_KG` on the *unchanged* geometry - nothing to
   do with where the part sits - puts one of six terrain seeds on its back and collapses
   the course from 6/21 obstacles to 3/21, which is the same picture the holder produces
   (1/21, one seed down). A different seed falls in each arm, which is what chaos looks
   like; the counts match. So this is the documented mass cliff, not a geometry
   regression - and note the flat trot was blind to all of it at 481/481/483 mm, which is
   the opposite of the 2.448 -> 2.459 kg case where the flat trot was the sensitive arm.
   Read the terrain and course arms with a +Δm control beside them, not just an unchanged
   one: without the middle column this would read as the holder breaking the walker.

   The gait is due a re-tune regardless - see the fitted-actuator paragraph below, which
   says the walker is still asking for swing speeds the servo does not have - so do not
   spend mass buying these numbers back until that is done.

   **Re-baselined 2026-09-09 by the fitted actuator** (`MJ_DAMPING`/`MJ_ARMATURE`/
   `MJ_FRICTIONLOSS`/`MJ_KP`, PLAN.md step 3). No geometry, no mass and no limit moved —
   `export_sim.py --check` reads the same 2.488 kg, the same 187 mm stand height and the
   same camera axis as before — and every distance in this step still fell by a quarter
   to a third. **Compare against these, and read the paragraph after them before calling
   a drop a regression:**

   | | new baseline | control (same tree, old `MJ_*`) |
   |---|---|---|
   | flat trot | **487.0 mm** at 2.487 kg | 556.6 mm |
   | terrain, seeds 7..12 | **340 ±39 mm** | 520 ±67 mm |
   | course, seeds 7 / 8 / 9 | **2/7 1903, 2/7 1705, 0/7 1005**, all upright | 1/7 1506, 5/7 2872, 4/7 2582 |

   **This is the model getting more honest, not the robot getting worse, and there is a
   number that settles it.** A joint can turn no faster than where the actuator's torque
   ceiling meets its damping: `(forcerange − frictionloss)/damping`. At the old
   0.12/0.02 that is **24.3 rad/s** — five times the ST3215's own vendor no-load speed of
   4.71, and thirteen times the ~1.8 rad/s the bench actually measured under the 1 kg arm.
   At the new 1.37/0.184 it is **2.01 rad/s**, which lands on the measured ceiling. So
   every gait distance this project has ever recorded was measured on a robot whose legs
   could swing an order of magnitude faster than the real ones, and the walker had tuned
   itself into exactly that headroom. The terrain sweep is the arm to read and it moved
   180 mm against a 32 mm standard error on the difference — decisively not one
   distribution, unlike every re-baseline above it.

   **What follows from that is a re-tune, not a revert.** The gait in
   `ros2/tools/standalone_sim.py` was fitted against the old joint and is now asking for
   swing speeds the servo does not have; the distances should come back with it. Nothing
   here says the constants are wrong — they are four measurements off one real ST3215 at
   three supply voltages, and `mini_dog.py` section 4 carries their provenance. Do not
   "fix" this by putting `MJ_DAMPING` back.

   Everything below this paragraph is the history that led here — the mass-cliff argument
   still holds and is the reason to keep running a control, but do not compare a new run
   against its distances. The 2026-09-08 set, superseded by the table above, was: flat
   790.8 mm, terrain seeds 7..12 537 ±67 mm, course 5/7 corridor 2778 mm.

   The first rewrites `smalldog_description/{meshes,urdf,mujoco,robot_params.json}`; the
   second must end `RESULT: OK — stands and trots forward` with a travelled distance close
   to the previous run (780 mm at 2.499 kg, with the GPS mast and the camera and no LiDAR
   guard; it was 780 mm at 2.495 kg with the guard and no camera, measured as a control on
   the same seed immediately before the swap - that pair is +5 g net and the distance did
   not move, which is the only reason it can be read at all. Before the GPS mast it was
   548 mm at 2.459 kg; that jump is the operating point moving under +36 g, not the robot
   getting better, see the bifurcation note below). The third runs the
   same trot on `mujoco/scene_terrain.xml` (675 mm at the default seed at 2.499 kg; 585 mm
   at 2.495 kg on the same seed, which is one seed's worth of chaos and not a change - and
   610 +-51 mm over seeds 7..12 at 2.495 kg; it was 658 mm and 652 +-56 mm at 2.459 kg -
   the same distribution) and has to pass too — it is the only test that
   exercises ground the gait cannot see in advance, and it is far more sensitive to a mass
   or limit change than the flat one. Judge it on the travelled distance and the end-of-run
   pitch together; the terrain is seeded, so both are repeatable.

   **The flat trot is hypersensitive to total MASS at this operating point, so never read a
   distance drop as a geometry regression without a control beside it.** Measured, same
   seed, fully deterministic to 0.1 mm across runs: 2.448 kg travels 778 mm; add 11 g
   *anywhere* and it is 597 mm (that number is 11 g parked in `ELECTRONICS_KG` — nothing to
   do with where the part sits); the same 11 g as the LiDAR guard, out at the nose, gives
   547 mm. The body also rides ~5 mm higher in both 11 g cases, so the walker is sitting
   near a bifurcation rather than simply working harder. That is a gait-tuning problem, not
   a CAD one, but it means every mass change wants the *unchanged* model re-run next to it.

   **Read that as mass specifically, not as "this gait is fragile to everything".** It was
   read the broad way once and it cost a change being held for no reason: −2 % of servo
   torque, which is a far larger relative perturbation than 4 ‰ of mass, moved the flat
   trot 781.4 → 781.6 mm — 0.2 mm, both sides deterministic — and left the terrain sweep
   and the course indistinguishable (below). So the cliff is under the mass axis and the
   torque axis is flat, which is worth knowing before deferring anything else on
   "it might move the operating point". The control run is still the answer either way:
   it is cheap, and it is the only thing that tells you which axis you are on.

   That figure *is* the bare heightfield: `terrain.py` also beds a ramp/wall/log course
   into the field, but it starts at x = 0.95 m and the 5 s trot only reaches 0.66 m, so the
   course is outside this measurement by construction — checked, both arms give 652 +-56 mm
   over the same six seeds. The current pair, seeds 7..12 at 2.496 kg, is **622 +-27 mm
   before the torque fix and 618 +-37 mm after** — a 5 mm difference of means against a
   ~32 mm spread, i.e. the same distribution, which is what "indistinguishable" has to
   look like before a terrain number means anything. Keep it that way. An obstacle inside the regression run turns a
   mass-and-limits signal into an obstacle-interaction signal: with the course at x = 0.55
   the same trot read 486 +-93 mm, so a real regression would have to beat the noise the
   course adds. The fourth command is the course, and it is a *report*, not a pass/fail:
   deterministic at the default seed — **re-baselined 2026-09-05: cleared 3/7 (log,
   ramp_up, deck), corridor reach 2209 mm at 2.492 kg**, established as a *control* run on
   the unchanged model, which came out bit-identical to the changed one beside it. The
   4/7 / 2606 mm below did not reproduce on this tree, exactly as the paragraph after it
   predicts; re-baseline, do not chase. The older reading was **4/7, corridor 2606 mm at
   2.496 kg** (the control beside it, on the pre-2.94-torque
   model at the same mass, was 4/7 and 2512 mm). Older readings, kept because they show
   the spread rather than a trend: 5/7 / 2790 mm at 2.499 kg, 5/7 / 2883 mm at 2.495 kg,
   5/7 / 2896 mm at 2.459 kg, and 2713 +-291 mm with 0/6
   down across seeds. Those three do **not** reproduce on the current tree — the same
   unchanged model now reads 4/7 — so treat them as history, not as a target to get back
   to, and re-baseline rather than chase them. Fully blind it is 1353 +-755 mm and
   3/6 down, so this is the test the terrain feedback actually shows up in — mostly the
   heading hold, which went in on 2026-08-28 and took the default seed from 1 obstacle
   cleared to 5. When `terrain.py` changes,
   re-baseline by running the *unchanged* model on the *new* course before reading anything
   into a drop.

   The walker levels the body on the IMU and holds a foot where it lands, so a *single*
   terrain run is a weaker signal than it looks: at fixed settings the distance spreads
   +-70 mm across terrain seeds, and it is chaotic in mass on one seed exactly as the flat
   trot is - adding the GPS mast's last 0.1 g moved the default seed 690 -> 585 mm with the
   flat trot unmoved at 780. Read the terrain arm over a seed sweep or not at all. `--blind` reruns it fully open loop, which is the baseline
   the closed loop is judged against; a real gait change wants a seed sweep, one png per
   seed (MuJoCo caches a heightfield by file name inside a process). See
   `../ros2/README.md`, "Terrain feedback". No `colcon build` is needed — that workspace is built with
   `--symlink-install`, so the install space points at these files. `out/sim/` from step 5
   and the ROS 2 package are *not* the same model: different link decomposition (the foot
   is merged into the shin there) and different collision primitives. Both must be current.
7. `rom_scan(..., step=2)` before committing to real joint limits — the default 10° sweep
   is coarse. `export_sim.py` reads the limits out of `out/bom.json`, so re-run
   `mini_dog.py` before re-exporting, or pass `--rom-step 2` to re-scan them itself.

## Notes

- FEA meshes in `out/fea/` are cached on the STEP content hash; stale entries accumulate
  and are safe to delete (they just re-mesh).
- **`bench_rig.py` and `torque_rig.py` are the files here allowed to be outside the ritual**, and
  the reason is that the arrow only points one way: it reads `mini_dog`'s servo interface
  so the bench holds the servo exactly the way the robot does, and `mini_dog` has never
  heard of it. Keep it that way - the moment a bench part acquires a mass in the robot's
  budget or a `PARTS` entry, it is a robot part and steps 4-6 apply to it.
- `mini_dog.py` runs take a couple of minutes — the ROM scan is swept boolean interference
  against real solids, not a cheap approximation. Don't add `--fast` paths that fake it.
- `README.md` marks several dimensions **verify** (Orange Pi hole pattern, Unitree L2 bolt
  circle, BMS outline). Those are genuinely unconfirmed; keep the marker until measured.
- The shin is the one lofted part (`shin_profile` / `shin_beam`, driven by the
  `SHIN_PROFILE` table). Its shape is measured, not taste: `tools/ref_ws_shin.py` reads it
  off the Waveshare part in `ref/`, `tools/ref_calf_profile.py` cross-checks against
  Unitree/MIT/Spot. Re-run one of them before reshaping the leg.
- **OCC booleans on the shin lofts fail silently.** A tool solid that crosses one of the
  lofted spline end caps makes `cut`/`intersect`/`fuse` return its own input unchanged —
  no exception, `isValid()` still True, and every *later* boolean on that shape fails the
  same way. This is why the cavity slab stops 0.5 mm short of the cavity's end caps and
  why the profile interpolation is monotone. After any change to `shin_beam`, verify with
  section areas (`tools/section_check.py`), not with `isValid()`.
- `tools/section_check.py` slices the real solid and reports its exact area second moments
  and the bending stress per newton of foot force. It is not a substitute for `fea.py` —
  it sees no stress concentration, no fork, no interlayer plane — but a shin re-profile is
  mostly a section-modulus change, and this costs seconds instead of an hour. Use it to
  converge on a profile, then still run `fea.py`.
