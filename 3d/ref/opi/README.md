# Orange Pi 5 Pro — vendor drawing and the board in hand

Vendor input for [`../../mini_dog.py`](../../mini_dog.py)'s `OPI_*` block, like everything
else under `ref/`: **read-only**, nothing here is generated.

`OPi_5_Pro_V1_2_1-dxf.pdf` is the vendor's two-page art film (page 1 bottom view, page 2
top view with the dimensions). Board serial in hand: 2024082600100403, 16 GB.

## Transcribed from the drawing

| what | value | in `mini_dog.py` |
|---|---|---|
| PCB outline | 89.1 × 56 × 1.6 | `OPI_BOX` is the outline *with* the connector row |
| mounting holes | 4 × M2.5 (H1–H4), **58 × 49** | `OPI_HOLES` |
| hole column, from the micro-SD end | **7.6** | `OPI_HOLE_EDGE` |
| hole rows, from the long edges | 3.5 (symmetric: 49 + 2 × 3.5 = 56) | pattern centred in y |
| two more holes (H5, H6) | on the 58-column line, 28.3 below H4 and 33.4 right of H1 — not used | — |
| M.2 M-key, underside | socket at the micro-SD end, **2280 outline** with its standoff at the ports end: a module spans the whole underside, ~22 wide, 4 mm off the centreline | the 8 mm `OPI_UNDER`, over the whole PCB |
| 40-pin header | top side, along one long edge | inside the 16 mm `OPI_ABOVE` |
| USB-A stack, RJ45, HDMI | top side, at the −x (ports) end, standing ~3 mm proud of the PCB edge | the box's extra 3 mm in x |

## Measured off the board, and where it disagrees

| what | measured | drawing | used |
|---|---|---|---|
| outline with connectors | 92 × 57.3 | 89.1 × 56 | measured — the box is a keep-out |
| hole to hole | 57.5 × 48.5 | 58 × 49 | **drawing** — a caliper on two ⌀2.7 holes reads short |
| PCB to the top of the tallest top-side part | 16 | — | measured (`OPI_ABOVE`); a cooler over the SoC may exceed it — **verify** |
| PCB to the bottom of the underside stack | 8 | — | measured (`OPI_UNDER`): the M.2 socket with a module in it |
| mass | 82 g board, 18 g cooler, 100 g case with screws — 200 g | 62 g on the product page | measured; `ELECTRONICS_KG` = 200 + a 50 g harness allowance |
