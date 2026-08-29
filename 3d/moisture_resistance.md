# Making the dog moisture-resistant — a plan, not a change

Nothing in this document is built yet. It is the survey that has to come before the
first `SEAL_*` constant: what the model is open at today, what the purchased parts
put a ceiling on, and what each stage costs in mass, heat and re-verification.

Read [README.md](README.md) for the spec and [CLAUDE.md](CLAUDE.md) for the change
workflow; every stage below ends in that same workflow, and the mass note in §5 is
the reason it cannot be skipped.

## 1. The ceiling is the purchased parts, not the CAD

| part | why it caps the rating |
|---|---|
| 12 × ST3215 | 25T output shaft with no lip seal, cable exit with no gland, case halves with no gasket, a gearbox that breathes. No edit to `mini_dog.py` changes any of that. |
| Unitree L2 | the manual forbids anything in front of the optical window — *"Even installing a transparent glass plate on the optical window will affect the performance"* — so the sensor cannot be covered and is protected to **its own** IP rating. **verify** against the L2 manual; that figure is not in this repository. |
| FDM PETG | four walls is not watertight across the layer boundaries, whatever the geometry does. |

So the target is **IPX4–IPX5** — rain, wet snow, spray off the feet, a watering can —
stated in the README as *do not immerse, do not jet-wash*. IP67 is not reachable on this
architecture and should not be claimed.

## 2. What is open today

The chassis is a ventilated box, not an enclosure.

| opening | code | area |
|---|---|---|
| 6 side windows 22 × 18 (x = −34/0/34, z = −4…14, both sides) | [`mini_dog.py:775`](mini_dog.py#L775) | ~2380 mm² |
| rear bus window 32 × 20 | [`mini_dog.py:778`](mini_dog.py#L778) | 640 mm² |
| deck cut-out 32 × 68 and the 2 × 52 slot | [`mini_dog.py:828`](mini_dog.py#L828) | ~2280 mm² |
| ⌀22 LiDAR cable core (`LIDAR_CORE_R`) | [`mini_dog.py:827`](mini_dog.py#L827) | 380 mm² |
| XT60 / XT30 / JST-XH pockets — the gap around each connector body; the 0.8 mm lip locates, it does not seal | [`mini_dog.py:779`](mini_dog.py#L779) | perimeter |
| deck ↔ tray joint: a flat face on a **2.8 mm** rim, 8 × M3 over a 436 mm perimeter → ~55 mm pitch | `DECK_SCREWS`, [`mini_dog.py:161`](mini_dog.py#L161) | whole perimeter |
| every `nut_slot()` channel — open to the outside by definition, or the nut cannot go in | throughout | 8 + 4 + 4 + 2 + 24 |
| per joint: 15 × 12 cable window + 2 × obround 20 × 12 cooling windows | [`mini_dog.py:638`](mini_dog.py#L638), [`mini_dog.py:650`](mini_dog.py#L650) | 12 × 3 |

The body alone is about **5700 mm²** of open section.

## 3. Stage A — seal the electronics bay

Cheapest in mass and in risk, and it covers everything that actually dies from water.

1. **A `SEAL_*` block in section 3**, shaped like `PANEL_*`: every dimension there, no
   literals in the part functions.
2. **A flange instead of a rim.** Widen the tray's top rim from 2.8 to ~8 mm and cut a
   groove for ⌀2 mm cord — or for a printed TPU 95A gasket, since TPU is already in the
   BOM for the feet and that adds no purchased part. An elastomer breaks none of the
   no-metal / no-machining / no-bearings rule.
3. **Screw pitch.** Even cord compression wants 30–40 mm; the deck is at 55. That is
   12–14 screws instead of 8 → new bosses, new `nut_slot()`s, and **each new slot's `ang`
   re-fixes the assembly order** in `README.md`, per the invariant in `CLAUDE.md`.
4. **Closing the side windows is a thermal decision, not a geometric one.** The Orange Pi
   5 Pro and the BMS (`ELECTRONICS_KG` = 0.25) end up in a sealed box with conduction as
   the only path out: a thermal pad from the Pi to the deck, deck as the radiator. Nothing
   in this repository models heat — this has to be measured on hardware, and it is the
   stage's real risk.
5. **A membrane vent (ePTFE, Gore-type, M5/M12) is mandatory, not optional.** Three
   reasons at once: pressure equalisation across warm/cold cycles (otherwise the gasket
   is pumped and the box inhales water), condensation, and a path for gas if the 3S2P pack
   fails. A hermetic box around six 21700s is the wrong answer.
6. **Connectors.** XT60 and XT30 carry no IP rating. Either recess them behind a tethered
   cap — the rear panel is already tight, the XT60 sits below the window because the strip
   beside it is 16.2 mm against the 16.5 an XT60 needs — or swap them for sealed shells.
   The balance lead wants a gland or potting.
7. **The servo bus has to leave the box** — twelve servos are outside it. A printed gland
   with an elastomer bush, and a drip loop below the entry.
8. **Camera.** The board already sits in a closed slot; seal the M12 holder against a
   printed bulkhead with an O-ring so the optics are outside and the electronics inside.
   A flat window in front of the lens is allowed (unlike the LiDAR) but adds flare and
   fogging — sealing the holder is better.
9. **GPS.** A 2 mm PETG radome (transparent at 1.575 GHz) over the receiver, or pot the
   board. The patch keeps its own tape.
10. **Every new cover or shield goes through the checks that already exist**:
    `interference()`, `lidar_fov_clear()` (`z + (x − LIDAR_X)·tan 45° < LIDAR_SEAT_Z`;
    the deck lip has 4 mm of margin), `camera_clear()`, `gps_clear()`. Nothing may go in
    front of the L2.

## 4. Stage B — legs and joints: drainage, not sealing

- The sleeve is open by construction: the cable window plus two lightening windows that
  are also the servo's cooling. Closing them cooks the servo; leaving them pours water
  onto its case. The compromise is to turn them downward or shield them with a labyrinth
  lip, and **nothing may enter r < 23 mm of the joint axis** — the distal fork's spine
  sweeps that annulus.
- **Drain holes at the low point of every hollow part**, the `shin` cavity above all
  (⌀2–3 mm). Careful: a boolean tool that crosses the loft's end caps makes OCC return
  its own input silently — check with `tools/section_check.py`, never with `isValid()`.
- The TPU foot presses onto the spigot and is bolted from below; it will fill. One drain
  hole in the sole.
- Bellows over the twelve joints: TPU, but they eat ROM and add mass, so a full
  `rom_scan(step=2)` follows. Not recommended for V1.
- Modifying the servo itself — shaft lip seal, sealant on the case joint, potting the
  cable exit — is outside this repository, but it has to be written into `README.md`,
  because the claimed rating depends on it.

## 5. Stage C — printing

- 5–6 walls or a 0.6 nozzle, raised flow, ironed top surfaces; or an epoxy wash inside the
  tray, which is the cheapest thing that works.
- **This moves `PRINT_FILL`.** Re-slice and re-measure — `out/gcode/summary.json` is the
  slicer side of that table — or the masses in `fea.py`, `export_sim.py` and the ROS 2
  generator all drift together.
- ASA over PETG buys UV, not moisture.

## 6. What it costs, and what must be re-run

The risk here is mass, not strength. From `CLAUDE.md`: 2.448 kg travels 778 mm in the 5 s
flat trot and **+11 g anywhere** drops it to 597 mm — the walker sits near a bifurcation.
A flange, a gasket, covers, four to six more screws and thicker walls are easily +50…100 g,
so every measurement wants the **unchanged** model re-run beside it.

The workflow for any of the stages above, in full:

1. `mini_dog.py` — ROM table, `body clear`, `lidar clear`, `gps clear`, `camera_clear()`.
2. `fea.py --all` — mandatory as soon as anything on the leg moves; judge on the
   inter-layer SF.
3. `export_sim.py --check`.
4. `cd ../ros2 && ../3d/.venv/bin/python smalldog_description/scripts/generate_model.py`,
   then the three `standalone_sim.py` runs, with a control run of the unchanged model.
5. Re-slice, update `PRINT_FILL`, and `tools/orient_scan.py` for any new part.
6. Update the fastener table and the **assembly order** in `README.md`.

## 7. Recommended V1.5

Stage A in full — flange, gasket, membrane vent, connectors, camera, GPS — plus drainage
and shields from stage B. The servo stays the weak link, and that goes into *Known
limitations* explicitly, together with *do not immerse*. Bellows and IP67 are a separate
project, and one that ends in a different servo rather than in different CAD.
