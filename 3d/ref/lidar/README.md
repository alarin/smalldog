# ref/lidar — the Unitree L2, measured

Like `ref/camera/` and `ref/imu/`, this is a **read-only input**: what a real sensor did,
transcribed, so the constants in `mini_dog.py` can point at something other than a
catalogue. Unlike those two, it is not a set of dimensions off a drawing — it is a capture
off the unit that arrived, taken 2026-09-05.

| | |
|---|---|
| hardware | 2.2.1.1 |
| firmware | 2.8.11.1 |
| SDK | 2.0.9 (`unilidar_sdk2`) |
| dirty | 3.1 – 3.2 % (the sensor's own window-contamination figure) |
| mode | ENETUDP, the factory default |

## What was measured, and what it changed

| | was | measured | now |
|---|---|---|---|
| point rate | 21600 /s **verify** | **62341 /s** | `LIDAR_RATE = 62340` |
| frame rate | 10 Hz (a sim choice in `lidar.py`) | **12.0 Hz**, 5196 pts/frame | `LIDAR_FRAME_HZ = 12.0`, in `mini_dog.py` |
| cone | 96.0° (`LIDAR_FOV_NEGA`) | **96.4°** max off-axis | unchanged — confirmed |
| range | 0.05 … 30 m **verify** | 0.17 … 10.1 m *observed* | unchanged — still **verify** |
| range noise | 20 mm 1σ **verify** | not measured | unchanged — still **verify** |

The point rate is the one that mattered: it was **2.9× low**, and low is the direction that
flatters a model, so any "the cloud is sparse" conclusion drawn off this repo before
2026-09-05 was drawn off a number nobody had checked. Two captures, 5 s and 20 s, agree to
2 parts in 62000 — 62339 and 62341 /s — so it is the sensor's clock, not a sample.

**The range row is not a measurement and must not be read as one.** The capture was taken
in a room about 10 m across: 0.17 m is the nearest thing in it and 10.1 m the furthest
wall, so the numbers bound the room, not the sensor. `LIDAR_R_MIN`/`LIDAR_R_MAX` keep their
**verify** marks until someone puts a target at a known distance. Same for `LIDAR_SIGMA`:
it needs a flat wall at a measured standoff and a fit of the scatter about its plane, which
is ten minutes' work whenever the sensor is next on the bench.

## The pattern, read off the capture

The manual says the L2 is densest at the middle of its vertical FOV; `lidar.py` used to
model a Risley pair and said so. Neither is what the unit does. Counts per 4° band of
off-axis angle are **flat** — 2650 ± 60 from 0° to 88° — and flat per degree means the
density per steradian is **1/sin θ**: ~30× the rim on the axis, falling monotonically, no
rim peak. The last 8° (88 → 96) thin linearly to nothing (a desk edge may be in that).

| off-axis band | points | pts/sr |
|---|---|---|
| 0 – 4° | 2666 | **174185** |
| 20 – 24° | 2580 | 15704 |
| 44 – 48° | 2770 | 8780 |
| 72 – 76° | 2450 | 5812 |
| 84 – 88° | 2605 | 5954 |
| 88 – 92° | 1571 | 3582 |
| 92 – 96° | 364 | 832 |

The time order says why: θ runs 88 → 0 → 88 at a constant ~13° per 21 returns while the
azimuth holds and then flips by 180° — the beam **spins in a plane through the axis**, one
meridian line per revolution, and that plane **precesses**. Rates: 216 axis passes per
second (the θ spectrum peaks at 215.7 Hz) and the meridian azimuth drifts 1567°/s over the
20 s, residual 4° — `BEAM_HZ` 215.7, `PREC_HZ` 4.354 in `lidar.py`, `lidar_spin` in the
model. The ratio is not an integer, so successive precession turns interleave and the
pattern never closes. `lidar.py` and `mujoco_lidar.cpp` now emit exactly this; the
reproduced profile is within ±6 % of the table to 88°.

What it does to `LIDAR_TILT`: at 45° the dense axis points 45° above the horizon. See the
tilt block in `mini_dog.py` and `tools/lidar_tilt.py`.

`tools/pcview.py --stats` prints points **and** points-per-steradian — the count column on
its own argues the sensor is uniform, which is the opposite of what it does.

## The capture

`l2_room.pcd` — 20 s, 240 frames, an ordinary room, sensor sitting on a desk. Decimated
1:21 (1247077 → 59385 points) with a fixed **stride**, not a random sample: the L2's scan
is non-repetitive, so a stride preserves the pattern's shape and its time order where a
random draw would smear both. Fields are `x y z intensity`, metres, in the sensor frame
(+Z is the optical axis, the same convention as `lidar_link`).

```bash
.venv/bin/python tools/pcview.py ref/lidar/l2_room.pcd --stats   # the table above
.venv/bin/python tools/pcview.py ref/lidar/l2_room.pcd --sim -c file   # real vs modelled
```

**This file predates the accelerometer being recorded, so it will not level itself.** Pass
`--up 0.206,-0.177,10.11` (the reading from the capture taken minutes later in the same
place) or re-capture. Everything taken with the current `capture_pcd.cpp` carries its own
`# imu_accel_xyz` header and levels without being asked.

The room it was taken in: sensor on a desk pointing at the ceiling, which is 2.0 m above
it and accounts for **45 % of every capture** — worth knowing before reading anything into
the point distribution, and the reason `pcview --clip` exists.

## How to take another one

The L2 has no driver in this repository and is not going to get one here — `lidar.py` is a
*simulation* of it. The capture path is Unitree's SDK on the Orange Pi, and what crosses to
this tree is a file, which is what the three-machine split expects anyway.

```bash
# on the Pi (Ubuntu 24.04, aarch64):
git clone --depth 1 https://github.com/unitreerobotics/unilidar_sdk2
cd unilidar_sdk2/unitree_lidar_sdk && mkdir -p bin
g++ -std=c++17 -O2 -Iinclude examples/capture_pcd.cpp \
    lib/aarch64/libunilidar_sdk2.a -lpthread -o bin/capture_pcd
./bin/capture_pcd out.pcd 20            # seconds
```

`capture_pcd.cpp` is `tools/capture_pcd.cpp` here, kept so the capture is reproducible, and
`tools/stream_pcd.cpp` is its live sibling — same SDK, same input, but it serves each frame
over TCP so `pcview.py --stream <pi>` can watch the sensor in real time instead of a file.

**The network is the part that catches people out.** The L2 ships static on
`192.168.1.62/24` and streams UDP `6101 → 6201` at the host it is told about, which is
`192.168.1.2` by default. The Pi's LAN is on `wlan0` (10.0.1.x) and its ethernet port had
no address at all, so the sensor was ARPing into silence — link up, 5 packets/s, nothing
listening. One address fixes it:

```bash
sudo ip addr add 192.168.1.2/24 dev enP4p65s0     # gone on reboot; make it an NM profile to keep
```

Ethernet is **not** PoE here: the L2 takes 12 V 1 A on its own DC3.5-1.35 barrel. The USB
adapter's DC jack is the alternative for the serial path, not an addition — power one, not
both.
