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
| `capture_pcd.cpp` | the L2 → one `.pcd`. Built and run **on the Pi** against `unilidar_sdk2`; kept here so a capture in `ref/lidar/` is reproducible |
| `stream_pcd.cpp` | the L2 → a TCP stream, one message per frame. Also runs on the Pi; `pcview.py --stream` is the other end |
| `pcview.py` | **a point cloud viewer** — an interactive VTK window over a `.pcd`/`.npy`/`.npz`/`.bin`/`.csv` capture, over the modelled L2 (`--sim`, which ray-casts `lidar.py` off `out/sim/mini_dog.xml`), over the **live** sensor (`--stream <host>`), or over any of those at once, which is how a real capture gets compared against what the model predicts. `--stats` prints the cone, the range percentiles and the return fraction without opening a window |
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

`pcview.py` is a **viewer, not a driver**: nothing in this repository talks to a real
Unitree L2. The sensor side is Unitree's own SDK, on whatever machine the L2 is plugged
into, and what crosses to here is a file — which suits the three-machine split, since the
git repository is the only thing that crosses anyway. Units are metres, not the CAD's
millimetres, because both the SDK and MuJoCo speak metres; `--mm` is there for the one
case where somebody exports a cloud in CAD units.

Its numbers are the ones the README's LiDAR **verify** rows want, and every one is a
property of the points rather than of anything in `mini_dog.py`: the off-axis histogram is
`LIDAR_FOV`/`LIDAR_FOV_NEGA`, the range percentiles are `LIDAR_R_MIN`/`LIDAR_R_MAX`, the
scatter of a wall at a known standoff is `LIDAR_SIGMA`, and points-per-frame is
`LIDAR_RATE`. Run it on the model first (`--sim --stats`) so you know what the model
claims before the real cloud arrives to disagree with it. Read the cone and range only in
the sensor frame — it says so instead of printing them off a world cloud.

### Watching the real sensor live

`stream_pcd` on the Pi serves one message per lidar frame on TCP 9910; `pcview --stream`
connects and redraws. The Pi is the server because it is the machine that stays put — the
mac dials out, so no inbound port has to be open on the laptop and neither end has to know
the viewer's address.

```bash
# on the Pi, once:
~/unilidar_sdk2/unitree_lidar_sdk/bin/stream_pcd 9910
# on the mac:
.venv/bin/python tools/pcview.py --stream 10.0.1.47
.venv/bin/python tools/pcview.py --stream 10.0.1.47 --accum 36   # 3 s of history on screen
```

`--accum` is not a cosmetic smoothing knob. One L2 frame is ~5200 points and looks like
nothing; the scan is non-repetitive, so showing a window of recent frames is the honest way
to make a live view legible — 12 frames is one second of real dwell, and moving the sensor
smears the picture by exactly as much as a real one-second exposure would. The reader runs
on its own thread and keeps only the newest window, so a slow render **drops** frames
rather than falling behind the sensor.

The caption counts both (`live 12f/42` = twelve frames on screen, forty-two received). If
the sender goes away the window keeps the last cloud and says so in the caption instead of
tearing itself down.

### Making a cloud readable

A raw lidar cloud is drawn in the **sensor's** frame, and the sensor is almost never
level — on this robot it is bolted at `LIDAR_TILT = 45°`. Unlevelled, every wall leans,
"height" means nothing, and the floor grid is a sheet of graph paper hanging across the
middle of the room. Three flags fix that, and they are the difference between the picture
being a rainbow blob and being a room:

```bash
.venv/bin/python tools/pcview.py cap.pcd                        # levels itself, colours by height
.venv/bin/python tools/pcview.py cap.pcd --clip -0.4,1.9        # drop the ceiling
.venv/bin/python tools/pcview.py cap.pcd --clip -0.4,1.9 --view top   # a floor plan
```

- **`--level`** (on by default) stands the cloud upright using the **L2's own
  accelerometer**, which `capture_pcd.cpp` writes into the `.pcd` header and
  `stream_pcd.cpp` sends with every frame. At rest an accelerometer points up, which is
  all that is needed. With no accelerometer in the capture it says so and leaves the cloud
  alone rather than guessing from the points — a room's largest plane is as likely to be a
  ceiling or a wall as a floor, and here it *is* the ceiling.
- **`--clip LO,HI`** keeps a height band above the floor. Indoors this matters more than
  anything else: in the reference capture the ceiling is **45 % of the whole cloud in one
  12 cm bin**, so from above you see a lid and nothing else.
- **`--view iso|top|front|side`** (`v` cycles). `top` on a clipped cloud is a floor plan.
  Each view carries **its own up vector**, which is not decoration: `top` looks straight
  down, so a +Z up vector is parallel to the view direction and VTK then picks the roll
  itself — the picture comes out arbitrarily rotated or mirrored, which reads as "upside
  down". Same defect that once made the first render an empty window with one dot in it;
  it survived longer here because a degenerate top view still draws *something*.
- **`f`** flips the up axis, for a capture whose accelerometer convention turns out to be
  the other way round. On the L2 it is not: verified against the room, the accelerometer
  points up, and the ceiling sits at +1.97 m along it.

Note what the height zero actually is. An upward-looking L2 reaches only 6° below its own
base plane, so on a desk **1.7 % of its points are below it** — it cannot see the floor,
and `ground` is the lowest *visible* surface, about desk height. `--stats` says so when it
happens rather than letting "height above the floor" stand unchallenged.

Colour is `height` by default on a levelled cloud, over a turbo ramp — VTK's default
blue→red hue sweep spends most of its length in cyan and green and reads as texture that
is not in the data. `ground` is a robust low percentile, not a plane fit, for the same
reason `--level` does not fit planes.
