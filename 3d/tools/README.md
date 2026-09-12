# tools/ — one-off measurement & diagnostic scripts

Kept for provenance: they are how the numbers in `../README.md` were *measured* rather
than guessed. None is imported by the build. All paths inside are relative to `3d/`:

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
| `ref_ws_shin.py` | Waveshare DOG PRO lower leg out of `ref/ROBOTIC_DOG_-STEP` — **the** reference for `SHIN_PROFILE` |
| `ref_calf_profile.py` | the same profile for Unitree A1/Go1/Go2/B2, MIT mini cheetah and Spot, off their own URDF meshes. Downloads into `tools/_refcache/` (ignored); `--plot` writes `out/ref_calf_profile.png` |
| `orient_scan.py` | unsupported area, print height and bed contact per build direction, off `out/stl/*.stl` — the screen behind `PRINT_ORIENT`; settle it by slicing, and by `fea.py --all --orient` for the load-bearing parts |
| `section_check.py` | area second moments and bending stress along the shin, sliced off the real solid — the cheap half of `fea.py` |
| `slice_orca.py` | slices with the OrcaSlicer CLI and the GUI's presets → `out/gcode/<name>.{gcode,3mf}`. Part names resolve against `out/stl/` then `out/bench/stl/` (`--stl-dir` overrides). `--upload` posts the gcode to the machine preset's `print_host` (Moonraker) and starts it, `--upload-only` just stores it; both refuse a printer that is already printing. macOS-only; it flattens a user preset's `inherits` chain first, because `--load-settings` does not and would silently slice with built-in defaults |
| `capture_pcd.cpp` | the L2 → one `.pcd`. Built and run **on the Pi** against `unilidar_sdk2`; kept so a capture in `ref/lidar/` is reproducible |
| `stream_pcd.cpp` | the L2 → a TCP stream, one message per frame, on the Pi; `pcview.py --stream` is the other end |
| `pcview.py` | **a point cloud viewer** over a capture, over the modelled L2 (`--sim`, ray-casting `lidar.py` off `out/sim/mini_dog.xml`), over the live sensor (`--stream <host>`), or all at once. `--stats` prints the cone, range percentiles and return fraction without a window |

## pcview.py

A viewer, not a driver: nothing here talks to a real L2. The sensor side is Unitree's SDK
on the Pi; what crosses is a file or a TCP stream. Units are metres (`--mm` for a cloud
exported in CAD units).

Its numbers are the README's LiDAR **verify** rows: the off-axis histogram is
`LIDAR_FOV`/`LIDAR_FOV_NEGA`, the range percentiles `LIDAR_R_MIN`/`LIDAR_R_MAX`, the
scatter of a wall at a known standoff `LIDAR_SIGMA`, points-per-frame `LIDAR_RATE`. Run
`--sim --stats` first so you know what the model claims before the real cloud disagrees.

```bash
# live: on the Pi, once
~/unilidar_sdk2/unitree_lidar_sdk/bin/stream_pcd 9910
# on the mac
.venv/bin/python tools/pcview.py --stream 10.0.1.47 --accum 36   # 3 s of history on screen

# a capture
.venv/bin/python tools/pcview.py cap.pcd --clip -0.4,1.9 --view top   # a floor plan
```

- `--accum N` shows a window of recent frames. One L2 frame (~5200 points) looks like
  nothing; the scan is non-repetitive, so 12 frames is one honest second of dwell. The
  reader drops frames rather than falling behind.
- `--level` (default on) stands the cloud upright using the L2's own accelerometer, which
  `capture_pcd.cpp` writes into the `.pcd` header and `stream_pcd.cpp` sends per frame.
  Without one it says so and leaves the cloud alone — the largest plane indoors is as
  likely the ceiling as the floor. `f` flips the up axis.
- `--clip LO,HI` keeps a height band; indoors the ceiling is ~45 % of the cloud in one bin.
- `--view iso|top|front|side` (`v` cycles); each view carries its own up vector, because a
  `top` view with a +Z up vector is degenerate and VTK then picks the roll itself.
- Height zero is the lowest *visible* surface: an upward-looking L2 on a desk cannot see
  the floor, and `--stats` says so.
