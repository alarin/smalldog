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
| `render.py` | offscreen VTK renders of `out/stl/*.stl`. |
| `tools/` | one-off measurement/diagnostic scripts, not part of the build (see `tools/README.md`). |
| `ref/` | vendor downloads: ST3215 STEP/PDF/wiki, Waveshare ROBOTIC DOG STEP. Read-only inputs. |
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
- Fasteners land in nut pockets on outer faces — nothing threads into plastic.
- **Masses and densities live once, in `mini_dog.py` section 4.** `fea.py`, `export_sim.py`
  and `../ros2/.../generate_model.py` all read them from there and keep no copies. They
  used to, and the servo mass silently diverged (55 g here, 60 g in the ROS 2 model — 60 g
  of robot). Anything the robot carries belongs in that block, including payload that has
  no printed part: the ground load cases in `fea.py` scale with the total.
- `build()` checks `shape.isValid()` per part; an `!! INVALID` line in the output is a
  failure, not a warning.
- **Every model change is re-checked for strength.** Any edit to `mini_dog.py` — a constant
  in the parameter block just as much as a part function — is unfinished until
  `.venv/bin/python fea.py --all` has been run and its safety factors compared against the
  numbers from before the change. A parameter you think is "cosmetic" still moves the load
  path: wall thickness, joint offsets and link lengths all change the moment arms `fea.py`
  derives from the joint frames. The same edit invalidates `out/sim/` — re-export it in the
  same pass (see below), so the printed, analysed and simulated robots never disagree.
- **The sim model is generated, never hand-tuned.** `out/sim/*.urdf|.xml` is output, like
  the STEPs. A link mass, a joint limit or an axis that is wrong in simulation is wrong in
  `mini_dog.py` — fix it there and re-export. Masses, inertias and limits come from the
  real solids and the ROM scan; only the MuJoCo actuator feel (`MJ_*`) and the collision
  primitives are estimates, and they are flagged as such in `export_sim.py`.
- The Waveshare `ROBOTIC DOG.step` supplies the **shape** of the shin (`SHIN_PROFILE`,
  measured by `tools/ref_ws_shin.py`) and nothing else. It is an aluminium-plate,
  single-shear-horn design: do not transfer its joint spacing or its absolute sections.

## Verifying a change

1. `.venv/bin/python mini_dog.py` — all parts valid, and check the ROM table it prints.
   Hip pitch collapsing to ±15° means the pitch axis lost its 30 mm drop below the roll axis.
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
   `!! it fell over`, `!! the two files disagree` or a link mass that jumped is a
   regression in the model, not in the exporter. It also cross-checks its own total
   against `fea.robot_mass()`.
6. **Regenerate the ROS 2 description, every time** — there are *two* exporters of this
   CAD, and this one lives outside the repo:

   ```bash
   cd ../ros2 && ../3d/.venv/bin/python smalldog_description/scripts/generate_model.py
   ../3d/.venv/bin/python tools/standalone_sim.py --headless
   ```

   The first rewrites `smalldog_description/{meshes,urdf,mujoco,robot_params.json}`; the
   second must end `RESULT: OK — stands and trots forward` with a travelled distance close
   to the previous run. No `colcon build` is needed — that workspace is built with
   `--symlink-install`, so the install space points at these files. `out/sim/` from step 5
   and the ROS 2 package are *not* the same model: different link decomposition (the foot
   is merged into the shin there) and different collision primitives. Both must be current.
7. `rom_scan(..., step=2)` before committing to real joint limits — the default 10° sweep
   is coarse. `export_sim.py` reads the limits out of `out/bom.json`, so re-run
   `mini_dog.py` before re-exporting, or pass `--rom-step 2` to re-scan them itself.

## Notes

- FEA meshes in `out/fea/` are cached on the STEP content hash; stale entries accumulate
  and are safe to delete (they just re-mesh).
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
