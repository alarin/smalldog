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
| `section_check.py` | area second moments and bending stress along the shin, sliced off the real solid — the cheap half of `fea.py` |

The `diag*.py` scripts import `mini_dog` and are the manual version of `rom_scan()`;
use them when a ROM number in the build output looks wrong and you need to know
*which* part is doing the blocking.

`ref_ws_shin.py` and `section_check.py` are offline and read only what is in the repo.
`ref_calf_profile.py` downloads into `tools/_refcache/` (gitignore-able) and writes
`out/ref_calf_profile.png` with `--plot`. It needs network on first run only.
