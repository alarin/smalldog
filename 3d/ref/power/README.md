# Power modules — the converters, as measured

Vendor/caliper input for [`../../mini_dog.py`](../../mini_dog.py), like everything else
under `ref/`: **read-only**, and nothing here is generated. The rails these parts serve,
and why there are two of them, are in [`../../../POWER.md`](../../../POWER.md).

Neither module has an envelope in `mini_dog.py` yet. `POWER.md`'s "What it costs the CAD"
is the rule: the moment one goes inside the body it becomes a modelled keep-out like
`OPI_BOX`, and that edit goes through the full ladder in `../../CLAUDE.md` (rebuild →
`fea.py --all` → `export_sim.py --check` → regenerate `../../../ros2`).

## Rail 1 — 12 → 5 V buck, for the Orange Pi

**XL4015E1, 5 A, constant-voltage + constant-current, red PCB.** 4–38 V in → 1.25–36 V out,
adjustable on two trimpots. Non-synchronous (Schottky), TO-263 on a bare copper pour, no
heatsink fitted.

### The envelope

| dim | value | note |
|---|---|---|
| length | **51.3** | |
| width | **26.0** | |
| height | **15.6** | the **toroidal inductor**, which is the tallest point |
| mass | **18.0 g** | vendor figure; inside the 15–25 g `ELECTRONICS_KG` allowance |
| mounting holes | **⌀2.3 × 4**, one per corner | measured |
| hole pitch | **46.5** (length) × **21.5** (width) | measured, centre to centre |
| underside | **1.5** | solder and components below the PCB — **inside** the 15.6 |
| above the PCB | **12.5** | derived: 15.6 − 1.5 − 1.6 of board |

**Nothing overhangs** — the two screw terminals are inside the 51.3 × 26 outline, so the
board outline *is* the swept envelope in plan. Measured/confirmed against the part, not
the listing.

### Why this part satisfies POWER.md

Conditions 1 and 2 of "one converter is enough" are met by the part number: **5 A** rather
than 3 A, and a **4 V** input floor against a 3S pack that bottoms at 9.9 V before any
transient. Condition 3 — wiring it as a star at the fused P+ with bulk capacitance — is
still on the builder. The module's own input cap is a 220 µF 35 V electrolytic, which is
the right order of magnitude but not low-ESR; `POWER.md`'s bulk-capacitance line still
applies.

### What the hole pattern implies

**⌀2.3 is an M2 clearance hole, and this repo has no M2.** The fastener block at
`mini_dog.py:65-78` carries `M25_*` and `M3_*` only, so mounting this board on its own
holes means adding an `M2_CLR` / `M2_TAP` pair — and both would be **UNVERIFIED** in
exactly the sense `../../CLAUDE.md` describes for the M2.5/M3 pair: a standard band minus
an FDM allowance, not a measurement. Print a coupon first. An M2 formed thread in FDM is
the marginal end of that rule even so.

The board is off the torque path — a converter bracket holds nothing but itself — so
threading into plastic *is* legal here. But a formed thread in FDM is a one-assembly
thread, and this is a board that may come out for service, so the honest options are an M2
nut, a heat-set insert, or **no fasteners at all**: a printed cradle with a retaining lip
is a fair answer to four ⌀2.3 holes.

Derived, and load-bearing for whatever holds it:

| | |
|---|---|
| hole centre → end edge | **2.40** (1.25 mm of board past the hole wall) |
| hole centre → side edge | **2.25** (1.10 mm) |
| a boss with 1 × D wall round an `M2_TAP` hole | r = 3.0, so it **overhangs the board** 0.75 mm each side and 0.60 mm each end |
| footprint at the standoff plane | **52.5 × 27.5**, not 51.3 × 26 — the bay is sized by the bosses, not the PCB |

### The bay is 15.6 mm: it is adjusted OUTSIDE the robot

Both trimpots and both terminal screws are driven from the top face, so a converter that
is adjusted in place needs a screwdriver path above it and the bay grows by the length of
a driver. **It is not adjusted in place.** Like the battery, it is a MODULE: trimmed to
5 V under load on the bench, CC pot wound to the top, both terminal blocks already wired,
and only then does it go in. Re-adjusting means taking it out.

Two consequences, and both are as load-bearing as a dimension:

- **It must come out without dismantling the robot**, which is the same conclusion the
  fastener trade reaches from the other side. Four M2 screws that cannot be reached past
  the deck would be the worst of both worlds; a cradle with a retaining lip is the answer
  to both questions at once.
- **The wires go into the terminal blocks before the board goes into the bay.** This is an
  assembly-order rule of exactly the kind `../../CLAUDE.md` records for `nut_slot()`'s
  `ang` argument — afterwards, the terminal screws face a wall.

### What is still to measure

| | why it matters |
|---|---|
| wire bend room off each short end | IN and OUT terminals face outward along ±length; the feed comes from the fused P+ node inside the tray |

**15.6 is the total, mounting plane to the top of the coil** — the 1.5 mm of solder and
components underneath is inside it, not on top of it. So a bay 15.6 mm deep holds the
part, *provided* it stands the board off by at least 1.5 mm: the underside is populated,
and it may not sit on a flat floor.

### The 1.5 mm standoff is too short for a thread, and that decides the fastener

`M2_TAP` wants 2 × D = **4.0 mm** of engaged length against a standoff that only has to be
**1.5 mm** tall. Three ways out, and they are not equivalent:

| | costs |
|---|---|
| let the thread continue into what the standoff stands on | needs ≥ 2.5 mm of material below the boss; impossible on a thin deck |
| grow the standoff to 4 mm | the bay becomes **18.1 mm** deep, not 15.6 |
| a nut on the far side, an insert, or **no fastener at all** | a printed cradle with a retaining lip and a 1.5 mm relief under the board owes nothing to M2-in-FDM |

The last one is the cheapest answer to four ⌀2.3 holes in a board that is off the torque
path, and it keeps the bay at 15.6.

### Two things that are not dimensions, and matter more

1. **The 5 V is set by a trimpot, not fixed.** A knocked pot puts an arbitrary voltage on
   the Pi. Set it, verify under load, then lock it — and mount the board so the pots are
   not reachable by accident once the deck is on.
2. **The CC pot is a trap.** These ship with the current limit at an arbitrary setting; if
   it is low it folds back under the Pi's inrush and looks exactly like a brownout. Set it
   to the top before setting the voltage.
3. **5 A is the heatsinked number.** Non-synchronous at ~85 % into 5 V × 4 A is ~3.5 W in a
   TO-263 with no heatsink. Derate, or fit one — and `POWER.md` already forbids putting it
   against the cells. A heatsink changes the 15.6.

## Rail 2 — 12 V boost / buck-boost, for the L2

Does not exist yet, and may not need to: it is conditional on the L2's input tolerance,
which is the open **verify** in `POWER.md`. Nothing to measure until that is settled.
