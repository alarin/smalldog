# Camera module — Weinan WN-L2101.K203L (Sony IMX415)

Vendor input for [`../../mini_dog.py`](../../mini_dog.py)'s `CAM_*` block, like everything
else under `ref/`: **read-only**, and nothing here is generated.

> **The original drawing is missing.** These numbers were transcribed from the vendor's
> *Product Dimensions* sheet (a three-view: Top / Side / Bottom) that came with the
> listing, but the image itself was pasted into a chat and never landed on disk. If you
> still have it, drop it in beside this file as `WN-L2101.K203L-dimensions.png` and delete
> this paragraph — a transcription is not a drawing, and the two readings flagged below
> can only be settled against the original or against a caliper.

## What the module is

| | IMX415 — **fitted** | OV5693 — the alternative, not used |
|---|---|---|
| model | WN-L2101.K203L-4K 8MP IMX415 | WN-L2103.K208L |
| sensor | Sony IMX415, **1/2.8"**, 8 MP | OmniVision OV5693, 1/4", 5 MP |
| max mode | 4K | 2K |
| lens | FF, 90 / 120 / 135° offered — **90° fitted** | AF/FF, 90° |
| interface | USB 2.0 UVC, MJPEG / YUV2, + **stereo** digital mic on board | USB 2.0 UVC, MJPEG / YUV2 |
| brand / origin | Weinan Electronics, China | Weinan Electronics, China |
| certification | CE, FCC, RoHS | CE (Eurotest), FCC, RoHS |

Chosen for the **sensor area**, not the megapixels: 1/2.8" is ~3× the area of the OV5693's
1/4", and a walking robot needs a short exposure or a face smears. The mic is a free
extra, and it is a **pair** rather than one capsule — which makes it a possible bearing
sensor and not only an input; see *The mic* below for what that does and does not buy. Why 90° rather than 120/135: a wider lens shrinks pixels-on-face by 1.4–1.8× and
adds barrel distortion at the edges, which is where the 5-point warp before an embedding
is most sensitive — and this robot has an L2 for the wide view.

## Dimensions, as transcribed

All ±0.2 unless noted. **The board is a 90 × 15 mm strip with the lens near one end** —
this, not the pixel count, is what shaped `camera_mount`.

| dim | value | what it is | consumed as |
|---|---|---|---|
| length | **90.0** | overall PCB, bottom view | `CAM_BOARD[0]` |
| width | **15.0** | overall PCB, all three views | `CAM_BOARD[1]` |
| thickness | **1.6** | PCB, side view | `CAM_BOARD[2]` |
| lens holder | **Ø14.0** | M12 holder OD, top + side view | `CAM_LENS_D` |
| lens stand-off | **16.2** | holder face to PCB front face, side view | `CAM_LENS_H` |
| axis position | **70.59** | along the board, from the connector end — *see reading note* | `CAM_LENS_U` |
| ear pitch | **18.0** | the two Ø2.2 holes straddling the lens, on the width centreline | `CAM_EAR_P` |
| mounting holes | **Ø2.2** | M2 clearance; two at the lens, two at the far end, one at the connector end | `CAM_EAR_D` |
| connector | **5.2** tall × **4.17** deep | on the BACK face, at the connector end | `CAM_CONN` |
| connector pitch | 1.25 | the tail connector's pins | — |
| header footprint | 3.8 × 4.79 | a small header, twice (near each end, front face) | — |
| far-end holes | 2.6 / 2.1 / 2.6 | across the 15 mm width, positioning the two end holes | — |
| near-end hole | 8.1 | from the bottom edge, ordinate from a 0 datum at the hole | — |

### Two readings that are not certain

1. **`CAM_LENS_U = 70.59`.** In the top view this dimension runs from the connector end of
   the board up to the lens block, with a nested **15.2** on the block itself. It is read
   here as *connector end → optical axis*. If 70.59 is instead the connector end → the
   *far edge* of the 15.2 block, the axis is at 70.59 − 7.6 = **62.99** and the module sits
   7.6 mm off where `mini_dog.py` puts it. That would not move the lens (it is on the
   robot's centreline either way) but it moves the 70.6 mm tail, and the tail is the part
   with 1 mm of clearance. **Check this with a caliper before printing.** The value used
   is the conservative one: 70.59 makes the tail *longer*, so if the true figure is 62.99
   every clearance in `camera_mount` improves and only `CAM_END` wants shortening.
2. **`CAM_FOV_D = 90`.** The catalogue states one angle. It is read here as the
   **diagonal**, which is the usual convention for M12 lens kits, and `camera_fov()`
   derives 82 × 52° from it and the 16:9 sensor. If the vendor meant the horizontal, every
   field-of-view number in the build output and in `README.md` changes.

### Not on the drawing at all

`CAM_OPT` (entrance pupil, up the axis from the PCB face — 12 mm assumed), `CAM_RATE`
(frames/s achievable at 4K over USB 2.0 MJPEG — 15 assumed) and `CAMERA_KG` (12 g
assumed; the vendor gives no mass). All three are marked `**verify**` in `mini_dog.py`.
Weigh the module and read the real UVC mode list off the device — `v4l2-ctl --list-formats-ext`
on the Orange Pi — before trusting the last two.

## The mic

Two capsules, not one — and the three-view does not show the mic at all, so **the one number
that matters, the spacing between them, is not transcribed here**. It decides what the second
channel is worth:

| spacing | what the second channel buys |
|---|---|
| ≥ 50 mm | real two-mic work: delay-and-sum, a GCC-PHAT bearing, a null steered at a servo. At 60 mm the largest inter-channel delay is 175 µs — 2.8 samples at 16 kHz, 8.4 at 48 kHz. |
| ~10 mm | 0.5 samples at 16 kHz. No usable bearing; coherence-based noise suppression only. |

The geometry is in its favour if they are spread along the strip: the board's long axis lies
along the robot's **y** (`CAM_TAIL = -1`), so a pair separated along the board is separated
*laterally*, and lateral is the useful axis — it gives azimuth to whoever spoke, which is
what a walking robot can act on. Elevation it would not get, and does not need.

Two checks on the real module, in this order, before anything is designed around it:

```bash
arecord -l                                              # does it enumerate as 2-channel at all
arecord -D plughw:1,0 -c 2 -f S16_LE -r 48000 two.wav   # and are the two channels different?
```

Plenty of modules sold as stereo duplicate one capsule into both channels, and that is a
thirty-second test. Then caliper the spacing, the same way `CAM_LENS_U` wants a caliper.

## How it is used

The module is not modelled from a vendor STEP (there isn't one): `mini_dog.camera_module()`
builds a board + holder + connector from the table above, purely so that `interference()`,
`rom_scan()` and the assembly can see the thing that is actually bolted on. The printed
part that holds it is `camera_mount`; the sensor half — MJCF `<camera>` and the two URDF
frames — is [`../../camera.py`](../../camera.py). The packaging argument, and the four
measured walls that leave exactly one place for a 90 × 15 mm strip, are in `mini_dog.py`'s
camera parameter block and in the main [README](../../README.md).
