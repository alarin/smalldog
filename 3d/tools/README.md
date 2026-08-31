# tools/ — one-off measurement & diagnostic scripts

Throwaway scripts kept for provenance: they are how the numbers in the root `README.md`
were *measured* rather than guessed. None of them is imported by `mini_dog.py`,
`fea.py` or `render.py` — nothing here is part of the build.

All paths inside are relative to the repo root, so run them from there:

```bash
.venv/bin/python tools/measure_servo.py
```

| script | what it measures |
|---|---|
| `measure_servo.py` | ST3215 STEP: solid list, overall bbox |
| `measure2.py` | ST3215 STEP: per-solid volumes and bounding boxes |
| `measure3.py` | ST3215 STEP: hub plates (⌀19.2), their Y offsets, ⌀14 bolt circle |
| `measure4.py` | ST3215 STEP: checks for threaded holes in the case side walls (there are none) |
| `measure_dog.py` | Waveshare ROBOTIC DOG STEP: assembly bbox, solid inventory |
| `measure_dog2.py` | Waveshare ROBOTIC DOG STEP: joint axis positions |
| `diag.py` | sweeps hip pitch, reports which solids interfere at each angle |
| `diag2.py` | hip bracket ∩ chassis bottom: clash volume and location |
| `ref_ws_shin.py` | Waveshare DOG PRO lower leg out of `ref/ROBOTIC_DOG_-STEP` — **the** reference for `SHIN_PROFILE` |
| `ref_calf_profile.py` | same profile for Unitree A1/Go1/Go2/B2, MIT mini cheetah and Spot, off their own URDF meshes — a cross-check on `ref_ws_shin.py` |
| `orient_scan.py` | unsupported area, print height and bed contact per build direction, off `out/stl/*.stl` — the screen behind `PRINT_ORIENT`; settle it by slicing, and by `fea.py --all --orient` for the load-bearing parts |
| `section_check.py` | area second moments and bending stress along the shin, sliced off the real solid — the cheap half of `fea.py` |
| `slice_orca.py` | slices with the OrcaSlicer CLI and the presets set up in the GUI → `out/gcode/<name>.{gcode,3mf}`. Part names resolve against `out/stl/` then `out/bench/stl/` (`--stl-dir` overrides), so a plate may mix robot and `bench_rig.py` parts. Default run is the test leg (`hip_bracket_A`, `thigh_A`, `shin_A`) on `TOP Neptune4` / `0.2-0.8 Neptune 4` / `TOP НИТ petg черный (scaled)` |

The `diag*.py` scripts import `mini_dog` and are the manual version of `rom_scan()`;
use them when a ROM number in the build output looks wrong and you need to know
*which* part is doing the blocking.

`ref_ws_shin.py` and `section_check.py` are offline and read only what is in the repo.
`ref_calf_profile.py` downloads into `tools/_refcache/` (gitignore-able) and writes
`out/ref_calf_profile.png` with `--plot`. It needs network on first run only.

`slice_orca.py` is macOS-only (it reads `/Applications/OrcaSlicer.app` and
`~/Library/Application Support/OrcaSlicer/user/default`) and it flattens the GUI's user
presets before handing them to the CLI: `--load-settings` does *not* resolve a preset's
`inherits`, and a user preset is only a diff, so passing one straight through silently
slices with built-in defaults for everything the diff does not mention.
