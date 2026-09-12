# CLAUDE.md

Parametric CAD for a 12-DOF printed quadruped built around 12 × Feetech/Waveshare ST3215
bus servos. Pure code-CAD — no GUI model, no feature history outside the scripts.
`README.md` is the human-facing spec; read it before changing geometry.

## Environment

Always use the project venv — CadQuery, gmsh, sfepy and VTK are installed there only:

```bash
.venv/bin/python mini_dog.py     # rebuild → out/step, out/stl, out/mini_dog_assembly.step, out/bom.json
.venv/bin/python render.py       # → out/view_{iso,front,side}.png
.venv/bin/python fea.py --all    # strength, every part, every load case
.venv/bin/python export_sim.py --check   # → out/sim/{mini_dog.urdf,mini_dog.xml,meshes/}
```

Run everything from `3d/`; scripts use paths relative to the CWD (`ref/…`) or to their own
location (`out/…`). A `mini_dog.py` run takes a couple of minutes — the ROM scan is swept
boolean interference against real solids. Do not add `--fast` paths that fake it.

## Layout

| path | role |
|---|---|
| `mini_dog.py` | **the model.** All geometry, all parameters, the ROM scan, the clearance probes, the exporters. |
| `fea.py` | linear static FEA (gmsh + sfepy). Imports `mini_dog`; derives load cases and clamp regions from its joint frames. |
| `export_sim.py` | URDF + MJCF export. Imports `mini_dog`; every length, mass and limit is read from it. |
| `camera.py`, `lidar.py` | the IMX415 and the Unitree L2 **as sensors**: the sites, `<camera>` and `<custom>` numerics both sim exporters emit, and the ray-cast scanner. No CAD in them. |
| `terrain.py` | procedural heightfield + the ramp/wall/log course, imported by both sim exporters. |
| `render.py` | offscreen VTK renders of `out/stl/*.stl`. |
| `bench_rig.py`, `torque_rig.py` | printed bench fixtures. One-way consumers of `mini_dog` (sleeve, clamp, hub pattern, densities); **not robot parts** — no `PARTS` entry, no mass, no FEA or sim consumer, so steps 4–6 below do not apply. Keep the arrow one-way. |
| `tools/` | one-off measurement/diagnostic scripts (`tools/README.md`). Not part of the build. |
| `ref/` | vendor downloads and transcribed measurements of bought parts. Read-only inputs. |
| `out/` | **generated — never edit by hand, never treat as source.** |
| `../ros2/smalldog_description/scripts/generate_model.py` | the *second* exporter of this CAD. Not optional — step 6. |
| `mini_dog_codex_handoff.md` | the original brief and its hard constraints. |

## How to change the model

Every dimension lives in the constant block at the top of `mini_dog.py`. Change it there
and re-run — never patch a part function with a literal, never edit `out/`. Part functions
return a CadQuery `Workplane` in robot coordinates; `PARTS` maps name → (workplane, qty,
note) and drives export and BOM; `PRINT_ORIENT` maps name → (axis, angle), applied to the
STL only.

## Invariants

**Source and measurement**

- **Units are mm** everywhere: geometry, FEA (N, MPa), exports (which scale to SI).
- **The servo interface is measured, not assumed** (`S_*`/`HUB_*`, from the vendor STEP via
  `tools/measure*.py`). If one looks wrong, re-measure. **A real part beats the STEP**:
  `HUB_BOLT_D` is M3 on the hub that arrived, ⌀2.5 in the STEP. Mark such rows in the
  README's source column.
- **Masses, densities, the MuJoCo joint feel (`MJ_*`), the foot contact (`MJ_FOOT_*`), the
  servo's stall torque (the 2 s peak, 3.2 — the held 2.3 is a runtime budget) and no-load
  speed, the LiDAR parameters and the IMU site all live
  once, in `mini_dog.py` sections 3–4.** `fea.py`, `export_sim.py` and the ROS 2 generator
  read them and keep no copies. Every one of these was duplicated once and diverged (servo
  mass 55 vs 60 g; rounded `J_EFF`; IMU sites 25 mm apart; joint limits narrower by 52° in
  one exporter). `rl/checks/check_model.py` is what catches it.
- `MJ_DAMPING` / `MJ_FRICTIONLOSS` / `MJ_ARMATURE` / `MJ_KP` are measured on one real
  ST3215 (provenance beside each constant). `MJ_DAMPING` and `SERVO_STALL_NM` are in one
  torque calibration — the scale's — because the joint ceiling divides one by the other;
  the ladder's 1.37 is the same measurement in the ladder's (× 0.596/0.400). `MJ_ARMATURE` = 0.0165 is deliberately not
  `rl/actuator.py`'s `Params.J_m` default of 0.008 — that is a flagged vendor prior and
  `rl/params/st3215.json` carries the fit.
- `MJ_FOOT_CONDIM` = 4, not 6: condim 6 triples the terrain spread (measured; it is the
  dimension, not its coefficient). Torsion is derived from the real TPU patch.
  `MJ_FOOT_PRIORITY` = 1 makes the foot's numbers win over the floor's — MuJoCo otherwise
  takes the elementwise max of the pair. Check a contact model by reading
  `d.contact[i].friction`, never the XML.
- The Waveshare `ROBOTIC DOG.step` supplies the shin's **shape** (`SHIN_PROFILE`) and
  nothing else — it is a single-shear-horn aluminium design.

**Structure**

- **The ST3215 case has no threaded side holes.** The sleeve+fork architecture exists
  because of that. Nothing loads the servo through a printed thread or a single-shear horn.
- **No metal parts, no machining, no external bearings.** FDM only plus the stock aluminium
  hubs (handoff doc).
- **Nothing in a torque path threads into plastic.** Every load-bearing screw not going
  into the stock aluminium lands in a nut in a `nut_slot()`; the slot's `ang` must point at
  a face still reachable at that moment of the assembly order — it is what fixes the order
  in `README.md`. Hub screws thread into the tapped aluminium plates (no room for a nut).
- **Off the torque path a screw may thread into plastic** (lids, covers): hole
  `M3_TAP`/`M25_TAP` (⌀2.6/⌀2.1, radii like the `_CLR` pair, both **UNVERIFIED** — print a
  coupon first), ≥ 1 D of wall, 2 D engaged, one-assembly thread. Unsure whether a load
  path runs through it → nut.
- **Do not slit the sleeve.** Its −x end wall is the tube's only crossing of y = 0.
  Measured: `thigh_A` inter-layer SF at stall 1.1 → 0.6, deflection 2.65 → 12.2 mm. The
  clamp that survives is the thrust clamp (`THRUST_*`).
- **Nothing goes into 23 < r < 34 of a joint axis over the sleeve's length** — the distal
  fork's spine sweeps it. **Screws count**: an M3 × 10 cap head reaches r = 24.2 and bound
  the first leg; the fastener is headless (`THRUST_HEAD_*`). Hub screw heads were the same
  defect: 1.65 mm ISO 7380 heads in 0.6 mm of air. `thrust_bolts()` and `fork_screws()`
  are in every `rom_scan`; `thrust_clear()` / `head_clear()` print margins every run.
- **The servo envelope must not be cut from the link that bolts to that servo's hubs** —
  `servo_envelope(hub=False)` / `env_all(no_hub=...)`. Sweeping the hub discs out is how
  all three forks once lost their bolt circles.
- **The hip-roll cradles are bolted parts** (`cradle_front` / `cradle_rear`, one per end
  because the halves' rails overlap on the centreline): four M3 through `CRADLE_REG_D`
  spigots into pockets in the tray wall. The register is stiffness, not decoration —
  0.1 mm of slop at the flange is 0.7 mm at the ground. Nut bosses run to `CRADLE_BOSS_X`,
  legal only because past the flange the fork's arm disc is the only thing there.
  `STRAP_Y` is a hard ROM limit: full-width straps read `hip_roll +0 .. +0`.
- **`cradle_front` is the weakest printed part** (inter-layer SF **1.33** at `land3g` on
  `--orient`; the crude `SF xy / z` column says 0.7 because it assumes the worst stress
  orientation — read `--orient` for this part). It is structural: the leg's load funnels
  back to the flange through a section capped at `STRAP_Y`, and the fork sweeps the space
  that would widen it. Filling the neck solid, closing it with a web and thickening the
  straps were each tried and each made the peak worse. It prints flange-down (its worst
  direction) on purpose: the only orientation with the servo bores vertical. Its stall case
  is a real couple (`couple=` in `part_specs()`) — a lever force about an axis concentric
  with the load patch divides by zero.
- **The camera slot at the nose is four measured walls, three of them other parts**: floor
  `CAM_LEDGE` = `CRADLE_Z1` (on `cradle_front`), ceiling the LiDAR base disc
  (`LIDAR_BASE_FLAT`), back the chassis front face, front the roll fork's rear arm at
  `ROLL_X + FORK_Y0`, a disc sweeping r ≤ 34. Only the lens passes it. ~1 mm everywhere:
  the fix for a clash is in the camera block, not in a part function.
- **The battery is a module** (`cell_holder` ×2 combs → welded brick → heatshrink →
  `battery_case` + `battery_lid`, BMS inside), dropped into the tray as one payload. Not
  structure — its lid screws are one of the two places a screw threads into plastic. The
  combs are clipped flush with the outer cells' tangent planes (no material outboard of any
  cell); `CELL_GAP` is the one number that costs height; the printed web between bores is
  `CELL_GAP − CH_FIT`. `batt_clear()` (lid to deck: 0.60 mm, a foam strip, never
  negative), `holder_clear()` and `module_clear()` exist because payloads are invisible to
  `interference()`.
- **The IMU sits on the deck's top face**, on the centreline inside the Orange Pi's
  standoff gap; `imu_clear()` measures against the Pi's *board*. The mount is `IMU_*` in
  section 3 via `imu_xyz()`, and both exporters emit the `imu` site from it.
  `rl/checks/imu_placement.py` is the argument for the position — run it whenever the
  mount or the gait moves (it runs on the mac). Moving the site is a re-baseline for `rl/`
  (retrain, not fine-tune).
- **The rear-panel openings are laid out inside the cradle flange's frame**, the XT60 above
  the cradle at z = +20 (the 9.64 mm band between `CRADLE_Z1` and `BODY_Z1`, so its z is
  not round). `panel_clear()` probes every opening along −x against the assembled body.

**Checks, and why each exists.** The recurring lesson: `isValid()`, `interference()` (static
body parts only) and `rom_scan` (moving parts only) are each blind to a class of defect.

- `build()` fails a part on `isValid()`, **`Volume() > 0`** and **solid count**
  (`PART_SOLIDS`, default 1; only `servo_gauge` is legitimately 2). A boolean on a
  degenerate contact comes back **inverted** with `isValid()` True (the chassis once
  shipped at −257 mm³ and everything downstream passed, 280 g light); an orphaned piece
  passes both. **Cut a through-path after everything that adds material near it.** Raising
  a `PART_SOLIDS` entry is a design claim — two of its three rows were permitting defects.
- **Fasteners get three separate questions: does the hole reach, does the head clear, can
  a driver get to it.** `foot_bolt_check()` (the foot bolt once opened inside the part and
  was specified 12 mm too short), `thrust_clear()`, `head_clear()`, `fork_access()` (a
  key-sized cylinder on each fork screw's real line, run **until it stands in open air** — a
  probe that ends inside a bore passes on a screw nobody can turn), `cradle_clear()`,
  `cradle_head_clear()`. `FOOT_BOLT_L` follows from `FOOT_CB_Z`, `FOOT_NUT_Z`, `M3_NUT_H`;
  move one, move the BOM line.
- **An access problem that keeps needing cleverer holes means the part boundary is
  wrong.** Two answers to the hip fork screws (four coaxial bores, then two 20° channels
  through the corner, eight holes) were deleted when the cradle became a bolted part and
  the screws ended up in open air. `README.md`, "Reaching the fork screws".
- **Payload gaps get their own probe**: `gps_clear()` (`OPI_BOX`), `imu_clear()`,
  `batt_clear()`, `holder_clear()`, `module_clear()`, `panel_clear()`, `lidar_fov_clear()`,
  `camera_clear()`. Every `!! …` line the build prints is a failure like `!! INTERFERENCE`.
- **OCC booleans on the shin lofts fail silently**: a tool crossing a lofted spline end cap
  returns the input unchanged, `isValid()` True, and every later boolean fails the same
  way. Hence the cavity slab stops 0.5 mm short and the profile interpolation is monotone.
  After any change to `shin_beam`, verify with `tools/section_check.py`, not `isValid()`.

## Verifying a change

Any edit to `mini_dog.py` — a constant as much as a part function — is unfinished until all
six steps have run in the same pass, so the printed, analysed and simulated robots never
disagree. A "cosmetic" parameter still moves the moment arms `fea.py` derives.

1. `.venv/bin/python mini_dog.py` — every part valid; ROM table (hip pitch collapsing to
   ±15° means the pitch axis lost its 30 mm drop below the roll axis); every clearance
   line green, no `!!`.
2. Mass and print bbox: parts must fit a 256 mm bed.
3. `.venv/bin/python render.py` — look at the three PNGs.
4. `.venv/bin/python fea.py --all` (`--selftest` first if `fea.py` changed; `--orient` if a
   `PRINT_ORIENT` entry changed). Run it **before** the change too. Judge on the
   **inter-layer** SF (second of the `SF xy / z` pair). A drop is a regression: fix it or
   report before/after explicitly.
5. `.venv/bin/python export_sim.py --check` — rebuilds `out/sim/`, stands both files up in
   MuJoCo, then on the terrain scene; renders the camera; scans one LiDAR frame;
   cross-checks total mass against `fea.robot_mass()`. `!! it fell over`, `!! the two
   files disagree` or a jumped link mass is a model regression.
6. Regenerate the ROS 2 description and run its regressions — the second exporter lives
   outside this tree and nothing here imports it:

   ```bash
   cd ../ros2 && ../3d/.venv/bin/python smalldog_description/scripts/generate_model.py
   ../3d/.venv/bin/python tools/standalone_sim.py --headless
   ../3d/.venv/bin/python tools/standalone_sim.py --headless --terrain
   ../3d/.venv/bin/python tools/standalone_sim.py --course
   ```

   No `colcon build` needed (`--symlink-install`). `--lidar` is a diagnostic; it changes no
   dynamics.
7. `rom_scan(..., step=2)` before committing to real joint limits; `export_sim.py
   --rom-step 2` re-scans.

### Current baseline (2026-09-12, 2.494 kg)

Inter-layer SF per load case. `stall` scales with `SERVO_STALL_NM` only, the three ground
columns with `fea.robot_mass()`. Read `--orient` for `thigh_A` and `cradle_front`.

| | stand4 | stand2 | land3g | stall | `--orient` |
|---|---|---|---|---|---|
| `hip_bracket_A` | 46.2 | 23.1 | 7.7 | 2.2 | 5.57 |
| `thigh_A` | 18.6 | 9.3 | 3.1 | 1.1 | 2.01 |
| `shin_A` | 59.2 | 29.6 | 9.9 | 4.5 | 12.65 |
| `cradle_front` | 4.0 | 2.0 | 0.7 | 2.7 | **1.33** |

`export_sim.py --check`: `4 feet down, upright +1.00`, base z 187 mm, camera axis
(+0.99 0 +0.10), ROM ±90 / ±90 / ±110 (±90 is the scan window, not a stop). Probes:
`imu clear +3.40`, `batt clear +0.60`, `clamp clear +1.43`, `head clear +0.65`,
`cradle bolts +5.15`, `fork access: all six arms`, `lidar fov 34.0 / 48.1 / 23.1`,
`camera view: out of frame`.

Step 6, same seeds: **flat trot 522.8 mm**; **terrain seeds 7…12: 415 ±44 mm, 0/6 down**;
**course seeds 7/8/9: 1986 / 1716 / 2149 mm, all upright**. (`SERVO_STALL_NM` 3.2 and
`MJ_DAMPING` 0.92 moved the joint ceiling 3.15 → 3.28 rad/s; the control — 4.50 / 1.37 on
the same mesh — reads 456.7, and the terrain was 403 ±34, 1/6 down: one distribution.)

### How to read step 6

- **The flat trot is deterministic to 0.1 mm and is the control**: unchanged means the
  model did not move. It is also **hypersensitive to total mass** (11 g anywhere:
  778 → 597 mm) and flat under torque changes, so never read a distance change as a
  geometry regression without a control beside it.
- **Read the terrain arm over seeds 7…12 or not at all** (±70 mm across seeds, chaotic in
  mass on one seed). Two means inside each other's spread are one distribution.
- **The course is a report, not a pass/fail**, read over seeds 7/8/9; the unchanged tree
  has read 5/7, 4/7 and 3/7 on the default seed.
- **Run a +Δm control**: the same mass parked in `ELECTRONICS_KG` on the unchanged
  geometry. 7 g of dead mass puts one terrain seed on its back and halves the course; the
  seed-8 fall in the baseline above reproduces at 0.79 g of dead mass to two millimetres,
  so it is the mass cliff, not the geometry. Without that column a mass change reads as
  the part breaking the walker.
- **Distances fell by a third twice (fitted actuator; then the gait limiting to the
  achievable `(forcerange − frictionloss)/damping` = 3.28 rad/s instead of the vendor
  no-load speed) and both were the model getting honest.** Do not recover them by putting
  `MJ_DAMPING` or the old rate limit back. The hand-tuned gait is due a re-tune.
- When `terrain.py` changes, re-baseline by running the unchanged model on the new course
  first. Seed sweeps need one heightfield file per seed (MuJoCo caches by file name).

## Notes

- FEA meshes in `out/fea/` are cached on the STEP content hash; stale entries are safe to
  delete.
- `README.md` marks several dimensions **verify** (Orange Pi hole pattern, `OPI_BOX`,
  connector bodies, GPS board, camera lens); keep the marker until measured.
- `tools/section_check.py` gives exact section moments along the shin in seconds — use it
  to converge a profile, then still run `fea.py`.
