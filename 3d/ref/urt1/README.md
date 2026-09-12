# Feetech URT-1 — the servo bus adapter, as measured

Vendor/caliper input for [`../../mini_dog.py`](../../mini_dog.py): **read-only**, nothing
here is generated.

The URT-1 is the robot's **only** link from the Orange Pi to the ST3215 bus. The handoff
brief's two-tier plan — an ESP32 doing low-level servo control under the Pi — was never
built: there is no ESP32 firmware anywhere in this repository, `robot/` is `pyserial` on
the Pi, and `PLAN.md` step 9 moves the 50 Hz loop *onto* the Pi. So this board is the
whole of the bay, and the ESP32 half of "the ESP32/URT-1 bay" should go.

## Measured

| dim | value |
|---|---|
| length | **56.0** |
| width | **36.6** |
| height | **14.5** over a mode switch; **13.0** over the connectors once the switch is cut |
| mounting holes | **⌀3.0 × 4** |
| hole pitch | **48.5** (length) × **27.5** (width) |
| hole centre → edge | 3.75 (ends), 4.55 (sides) |
| mass | **verify** — not yet weighed |

⌀3.0 is a snug M3 clearance; `M3_CLR` in this repo is ⌀3.4, so the board's own holes are
tighter than the repo's standard hole and an M3 is a close fit, not a slip fit.

## It does not fit the bay that is named after it

Measured on the current solid, not estimated:

| | |
|---|---|
| tray inner rear face | x = **−60.20** |
| battery module rear face | x = **−46.15** |
| **rear strip** | **14.05 mm** |
| connector-panel pads intrude to | x = **−55.00** (`PANEL_T` = 8.0 from the outer face) |
| **strip clear of the pads** | **8.85 mm** — this is the "8.9 mm" the `BATT_X` comment cites |
| the `ESP_X` rib splits the strip | 7.70 mm / 3.35 mm |
| URT-1 needs | **14.5 mm** + a fit ≈ 15.5 |

Deleting the ESP32 rib does not rescue it: the undivided strip is 14.05 against a board
whose *thinnest* dimension is 14.5, and clear of the connector pads it is 8.85. `BATT_X`
has only 4.3 mm of forward travel before the module hits the front of the tray, which
would take the clear strip to 13.15 — still short, and with nothing left at the front.

Cutting the mode switch takes it to **13.0**, which is inside the undivided 14.05 strip
but still outside the 8.85 that is clear of the connector pads. And height is not the only
wall: standing in the strip the board must be 56 across y and 36.6 in z, and 36.6 raised
clear of a connector pad's top at z = −8.75 reaches z = 27.85 against a deck at 25 (the pad has since moved to z = +20; the arithmetic is stale).

## There is no home for it anywhere on this robot

| space | why not |
|---|---|
| rear strip | 8.85 mm clear of the pads, against 13.0 |
| side strips beside the module | 9.1 mm each |
| over the battery lid | 1.4 mm |
| deck, aft | `OPI_BOX` starts at x = −72, i.e. it already overhangs the deck's own rear edge at −63; the GPS mast is at x = −52 |
| deck, forward | x = 28…63 is 35 mm, less than the board's 36.6 — and the LiDAR pedestal spans x = 16…68 through all of it |
| deck, outboard | 15 mm strips |

## The conclusion: the URT-1 is a BENCH tool, not a robot part

The board is 36.6 × 56 because it is a configuration and development adapter — USB, servo
headers, a mode switch — and the robot needs exactly one function out of it: a USB ↔ TTL
half-duplex bridge. `../../README.md:687` already uses it that way ("set servo IDs 1…12
over the bus **before** assembly"), and `../../../POWER.md` already forbids the other
thing it looks capable of ("never through the URT-1" — a stalled ST3215 draws 2.7 A and a
debug adapter is not a power distribution board).

**Recommendation: keep the URT-1 on the bench, and on the robot drive the bus from the
Orange Pi's own UART pins — no USB, and no board in the bay at all.**

This is not just a packaging dodge. `../../../robot/README.md`'s 50 Hz budget already says
where the tick is lost: *"The wire is never the constraint. The host is: USB frame
scheduling, the adapter's latency timer (FTDI ships at 16 ms, which alone eats the
tick)."* A native UART has no frame scheduling and no latency timer, so this deletes the
dominant jitter source rather than packaging around it. `bus.py` already tolerates the
half-duplex echo (`discard_echo` auto-detects it), so the driver needs no change.

### What it still needs, and what must be verified first

The Pi's UART has separate TX and RX; the servo bus is one wire. Something has to do
half-duplex. The lowest-part-count answer is the classic passive one — RX straight on the
bus line, TX through a diode or series resistor so it can pull the line without fighting
the servo — which is three components inline and effectively zero volume.

Four **verify** items, none of them more than a few minutes, and the first one can damage
the Pi if skipped:

| | how to settle it |
|---|---|
| **the level in the servo → Pi direction** | the servo accepts 3.3 V in (spec: high 2–5 V, low 0–0.45 V), so driving it is fine. What it *outputs* is the risk: if it drives to 5 V it over-drives the Pi's 3.3 V GPIO. Scope or meter the line while a servo replies, **before** connecting it to the Pi |
| **1 Mbit on the Pi's UART** | a 24 MHz reference wants a divisor of 1.5 for 1 Mbps — it needs a fractional divider. Confirm the port actually achieves 1 000 000 baud and not something near it, or the framing errors are silent |
| **which header UART is free** | UART0 is usually the debug console; the rest need a device-tree overlay |
| **direction timing** | the passive circuit needs none. A tri-state buffer would, and toggling it from Python at 1 Mbit is not viable — that path only works with the driver's RS485 auto-direction (`TIOCSRS485`), which is itself a verify on this SoC |

### What the CAD gets

No board in the rear bay. The 8.85 mm strip becomes a cable route and, at most, a seat for
a three-component inline circuit. The `ESP_X` divider rib goes, and `BATT_X` = 5.0 is
freed from the 8.9 mm constraint it was chosen to satisfy.

## Prior art: people do drive these servos straight off an MCU UART

**[`sepastian/ESP32_ST3215`](https://github.com/sepastian/ESP32_ST3215)** — "ESP32
controlling ST3215, *not* using the official controller board." The whole circuit is **one
10 kΩ resistor** between TX (GPIO17) and RX (GPIO16), with both tied to the servo's data
line, common ground, 1 Mbps. The Dynamixel world has done the same thing for fifteen years
on the same bus topology.

**It is a one-servo bench circuit, and it does not carry to this robot.** The author
addresses neither multiple servos, nor echo, nor reliability, and there are two separate
reasons it breaks here:

1. **Contention.** The Dynamixel community's verdict on the resistor trick is explicit: if
   the UART's TX drives the line high while a servo answers by pulling it low, the servo's
   driver is fighting the TX pin. The resistor limits the current; it does not stop the
   fight.
2. **Slew rate, which is the one that actually kills it.** 10 kΩ into the bus capacitance:
   one servo on a short lead is ~50–100 pF, so RC = 0.5–1 µs against a **1 µs bit** at
   1 Mbps — marginal on a breadboard. This robot is **12 servos on a 4-branch signal
   star** with a real harness: several hundred pF plus stub reflections. 10 kΩ will not
   work, and lowering the resistor to fix the slew rate makes the contention worse.

The community's answer is a **tri-state buffer** (74HC125 / 74HC241) with direction
control — which is precisely what is inside the URT-1, and why Feetech ships one at all.

### The part that resolves it: Waveshare Bus Servo Adapter (A)

[Product page](https://www.waveshare.com/bus-servo-adapter-a.htm) ·
[wiki](https://www.waveshare.com/wiki/Bus_Servo_Adapter_(A)). It does the half-duplex in
hardware and takes **UART directly** — the page names Raspberry Pi Zero, ESP32, Arduino
and STM32 — as well as USB. That is the native-UART path (no USB stack, no latency timer)
*with* hardware direction control, on a board far smaller than the URT-1.

#### As specified by Waveshare

| | |
|---|---|
| footprint | **42.0 × 33.0** |
| height | **still not published.** Tallest is the barrel jack, then the green screw terminal, then the A/B jumper; USB-C and the servo headers are low. Caliper it on arrival |
| mounting holes | **⌀2.5**, spacing **37.0 × 28.0** |
| interface | UART header (**TX / RX / GND**) *or* USB-C, selected by an **A/B jumper**: A = UART ~ SERVO, B = USB ~ SERVO |
| servo ports | **two**, 3-pin, labelled **D V G** |
| power in | DC **9–12.6 V**, on a 5.5 × 2.1 barrel **and** a parallel green 2-pin screw terminal (**DC+ / DC−**) |

#### What the board drawing settles

- **The UART path is a first-class mode, not a hack.** There is a dedicated TX/RX/GND
  header and an A/B jumper that hands the servo bus to it. Set **A** and the USB-C is out
  of the picture entirely — which is the whole point of the exercise.
- **The signal-only wiring is physically easy.** The servo ports are `D V G` with V in the
  middle, so a 3-pin housing with the middle contact omitted gives data and ground with V+
  unconnected. No cutting of traces.
- **The barrel jack is optional, and it is the tallest thing on the board.** The green
  DC+/DC− screw terminal sits in parallel with it. For a permanent install the screw
  terminal is the better input anyway — a barrel jack works loose under vibration — and
  **desoldering the barrel is then a real height saving**, the same move as cutting the
  URT-1's mode switch. **verify** that the green terminal is a parallel *input* and not a
  pass-through output before relying on this.
- **The A/B jumper is set-once**, like the buck converter's trimpots, so it is configured
  on the bench before the board goes in. Its shunts stand proud; a solder bridge instead
  saves a little more height.

M2.5 is already native to this repo — `M25_CLR`, `M25_TAP`, `M25_NUT_AF` — and is what the
Orange Pi's own standoffs use, so this board mounts with the constants that already exist.
No new fastener pair, unlike the buck converter's M2.

#### Its power path cannot carry this robot, and that is not a detail

"DC 9–12.6 V (ST series servo)" on a 5.5 × 2.1 barrel means the **servo rail is routed
through the board**. A 5.5 × 2.1 barrel is good for roughly 5 A. This robot draws the 24 A
transients `../../../POWER.md` is written around, and twelve stalled ST3215s would be
~32 A. `../../../robot/README.md:715` already states the rule for the URT-1 and it applies
here verbatim: *a debug adapter is not a power distribution board.*

**Use it for signal only.** The servos are fed from the fused P+ star as POWER.md
already specifies; the adapter's servo connectors get **data and ground only, V+ left
unconnected**, which needs a custom harness because a stock servo lead carries all three.
Its barrel then supplies nothing but the board's own logic current, which is what a
5.5 × 2.1 can actually do.

#### Where it goes, checked against the solid

Standing in the rear strip at **y ∈ [−23, +19], z ∈ [−8.75, +24.25]** it clears
everything (stale — the panel now has one pad, the XT30 at y ±9, z 13.7…25): the XT60 pad tops out at z = −8.75, the XT30 pad is at y = 19…31, the rear
deck bosses are at y ≥ 35.2, and the full 14.05 mm of strip thickness is available there
because no pad reaches that band. It leaves **0.75 mm to the deck**, and it is asymmetric
in y, which costs nothing.

Two things that still need the real CAD pass rather than arithmetic: that 0.75 mm is
nothing, and the "bus window" the `PANEL_*` comments place at |y| = 16 is referenced but
its geometry was not located — it may cross this band.

#### Still to verify

**Ordered 2026-09-09; these are the checks for when it lands.**

- the board's **height** — the number the whole placement above depends on
- the **UART logic level** against the Pi's 3.3 V, in both directions
- **mass**, for `ELECTRONICS_KG`
- whether the servo connectors' **V+ pins can be left unconnected** without upsetting the
  board — the signal-only wiring above depends on it
- that its bus driver is a real **tri-state buffer** and not the resistor trick in a box:
  scope the data line's edges with several servos on the harness, which is the case the
  ESP32 prior art never covers

Until it arrives nothing in the CAD should move. The `ESP_X` rib removal, the `BATT_X`
change that the 8.9 mm strip no longer constrains, and the ESP32 references listed in the
git history all want the height first, and they are one pass.
