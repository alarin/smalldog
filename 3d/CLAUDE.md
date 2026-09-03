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
  re-measure — do not adjust it to make a part fit.
- **The ST3215 case has no threaded side holes.** The whole sleeve+fork architecture
  exists because of that. Nothing may load the servo through a printed thread or a
  single-shear horn.
- **No metal parts, no machining, no external bearings** (628/685 etc.). FDM only, plus
  the stock aluminium hubs. This is a hard constraint from the handoff doc.
- Nothing threads into plastic. Every screw that is not going into the stock aluminium
  lands in a nut, and every one of those nuts sits in a `nut_slot()` — a channel of the
  nut's across-flats width, so its two walls stop the nut turning. When you add one,
  the `ang` argument is not cosmetic: it has to point at a face that is still reachable
  at the moment in the assembly order when that nut goes in, and that is what fixes the
  assembly order in `README.md` (the LiDAR pedestal has to be bolted to the deck before
  the deck goes on the tray, because its screw heads end up inside the tray). At the
  servo hubs the screw threads into the stock aluminium plate (both are tapped M2.5) —
  there is no room for a nut there, and a hex pocket would eat over half the fork arm
  right under the screw head.
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
  `foot bolt:` line; treat `!! FOOT BOLT` as a failure like `!! INTERFERENCE`. The
  length follows from `FOOT_CB_Z`, `FOOT_NUT_Z` and `M3_NUT_H` — if you move any of
  those, `FOOT_BOLT_L` and the BOM line in `README.md` move with them. Fixed 2026-08-31.
- **Nothing goes into 23 < r < 34 of a joint axis over the sleeve's length.** The distal
  fork's spine sweeps that annulus, and the hip bracket's inboard web already comes to
  r = 22.0. It is the binding constraint on the sleeve thrust clamp, and **screws count**:
  the lug corner sits at 21.6 and always cleared, but an M3×10 cap head reaches 24.2 and
  bound the first assembled leg over ±2…38° of travel. The fastener is specified headless
  for that reason — `THRUST_HEAD_*` — and `thrust_clear()` prints the margin on every run
  (`clamp clear:`), with `thrust_bolts()` in the static side of all three `rom_scan` calls.
  A check over the printed solids alone cannot see this: the part sticking out is hardware.
- **Do not slit the sleeve.** Its `-x` end wall is the tube's only crossing of y=0 — the
  cable window has already eaten the `+x` one — so a C-clamp slit there opens the whole
  sleeve-plus-link box section. This was tried and measured: `thigh_A` inter-layer SF at
  stall 1.1 → 0.6, deflection 2.65 → 12.2 mm, and tying the `+x` wall back across the
  window recovered none of it (0.5). The clamp that survives is the one that adds material
  and takes the play out in thrust (`THRUST_*`), not the one that cuts the ring.
- **The camera's slot at the nose is four measured walls, and three of them belong to
  other parts.** Floor `CAM_LEDGE` = 15.36 (the hip-roll cradles and their root gusset),
  ceiling the LiDAR pedestal's base disc at z = 29 (hence `LIDAR_BASE_FLAT`), back the
  chassis front face at x = 63, front the hip-roll fork's rear arm at
  `ROLL_X + FORK_Y0` = 68.1 — a disc that sweeps r <= 34 about the roll axis over the whole
  roll ROM. Only the lens may go past 68.1, and only because on the centreline it stays
  36.3 mm from either roll axis. Moving `LIDAR_BASE_R`, `LIDAR_X`, `SLEEVE_W`, `ARM_T` or
  `roll_module`'s rails moves one of those walls; `interference()` and `rom_scan` will say
  so, but the fix is in the camera block, not in a part function. A 90 x 15 mm board in a
  5 x 15 mm slot has ~1 mm everywhere: treat every one of these as load-bearing.
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
- **The servo's stall torque and no-load speed are read from here too**, by both
  exporters — `SERVO_STALL_NM` = 2.94 and `SERVO_NOLOAD_RADS` = 4.71. This was the
  same defect a third time and the worst-stated of the three: `export_sim.py` read
  them, `generate_model.py` kept rounded copies (`J_EFF = 3.0`, `J_VEL = 4.7`), so
  the two exporters were not duplicating a constant, they were emitting **robots with
  different servo strength** — and which sim you loaded decided how strong the servo
  was. Fixed 2026-08-31 with the re-baseline in the same commit; `rl/` loads the ROS 2
  model, so that was the one that mattered.
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
- **The IMU board's slot is 3.6 mm and interference() cannot see either wall.** Above it is
  the deck's underside at `BODY_Z1`; below it is the battery pack's top at
  `BODY_Z0 + BATT_H` = 21.4, and the pack is a payload, so `imu_clear()` exists for it the
  way `gps_clear()` exists for `OPI_BOX`. The board and its components are 2.8 mm of the
  3.6, which is why `IMU_STACK` says headerless and why `ref/imu/README.md` says solder to
  the pads. Moving `DECK_T`, `BATT_H`, the deck window or `IMU_TAB_T` moves one of those
  two walls; the build prints the remaining gap on every run, and it is allowed to be
  small but never negative.
- `build()` checks `shape.isValid()` per part; an `!! INVALID` line in the output is a
  failure, not a warning.
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
  sensor's cone, point rate, range and noise are in section 3's LiDAR block; `lidar.py`
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
   `chassis_bottom` / `chassis_top` / `lidar_mount` / `gps_mount` / `camera_mount` — and
   the camera module itself, which is not a printed part but is bolted to the same body —
   that
   share more than 1 mm³ of solid, and with `gps clear:` — the GPS mast against the Orange
   Pi's `OPI_BOX` envelope, which `interference()` cannot see because the Pi is a payload
   and not a part.
   That is a failure, not a warning — `isValid()` never sees it, and `rom_scan` only covers
   the parts that move. Keep it cheap: three static solids, no sweep.
2. Mass and print bbox in the same table: parts must fit a normal 256 mm bed.
3. `.venv/bin/python render.py` and actually look at the three PNGs.
4. **Strength, every time** — `.venv/bin/python fea.py --all` (covers `hip_bracket_A`,
   `thigh_A`, `shin_A` over all four load cases). Run it *before* the change too, or keep
   the previous run's output, so there is a baseline to compare against. Judge on the
   **inter-layer** SF (the second of the `SF xy / z` pair) — that is the one FDM parts
   actually fail at. Any part whose inter-layer SF drops is a regression: fix it or report
   the before/after numbers explicitly, do not just note that the parts are still valid.
   Run `--selftest` first if you touched `fea.py` itself, and `--orient` if you changed a
   part's `PRINT_ORIENT` entry.
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
   deterministic at the default seed — **currently cleared 4/7, corridor reach 2606 mm,
   upright at x = 2606 mm at 2.496 kg** (the control beside it, on the pre-2.94-torque
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
- **`bench_rig.py` is the one file here that is allowed to be outside the ritual**, and
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
