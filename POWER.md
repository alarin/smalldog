# POWER.md — the power tree, and the two things it is short of

Written 2026-09-09, out of the question "is a 12 → 5 V converter enough to run the Pi and
the IMU". The short answer is at the bottom of *The 5 V rail*; the rest is why.

**Nothing here has been measured on a complete robot**, because on 2026-09-09 the robot is
1.55 kg on a bench supply and the battery, the Orange Pi, the IMU and every converter below
are still missing ([`PLAN.md`](PLAN.md)). What *is* measured is marked as such and cited.
Everything else is vendor or catalogue and carries **verify**, in the sense
[`3d/CLAUDE.md`](3d/CLAUDE.md) uses it: a number to settle before it is trusted, not a
number to quietly round.

**This file is the electrical side only.** Mass, keep-outs and geometry live once, in
[`3d/mini_dog.py`](3d/mini_dog.py) — section 4 for the mass block, the `PANEL_*` block for
the rear connector panel. Where a figure appears both here and there, the CAD is right and
this file is stale.

## The tree

```
  3S2P pack, 6 x 21700, 12.6 -> 9.9 V, 0.42 kg          BATTERY_KG, mini_dog.py:638
    |
    +-- fused P+, INSIDE the tray: the module's lead is the master disconnect  30 A MIDI
    |          (bench supply goes on the same node, deck off)
    +-- XT30   charge only, the one external connector, own branch fuse      7.5 A ATO
    +-- JST-XH 3S balance lead, stays in the case: balance-charge with the module out
    |
    +--> raw pack ------> 12 x ST3215: one 18 AWG pair PER LEG from the node, spliced
    |                     at the hip; the bus leads carry data only   UNREGULATED, and deliberately so
    |
    +--> buck 12->5 V ---> Orange Pi 5 Pro
    |                        +-- 3.3 V header pin --> BMI088 IMU        (SPI, 2 x CS)
    |                        +-- USB --------------> IMX415 camera
    |                        +-- ------------------> GY-NEO6MV2 GPS     **verify** rail
    |
    +--> 12 V, regulated -> Unitree L2                    see *The LiDAR wants its own rail*
```

One connector leaves the tray. `PANEL_AT` puts the **XT30 above the cradle** at z = +20
on the centreline (`3d/README.md`, *Payload bays*); the other openings are cable windows
(a bus window and a low window in each end wall). There is no XT60 - the load lead never leaves the tray - and no balance
pass-through. The charge shell is an XT30 so the wrong plug does not fit.

## The BMS is same-port, and there are two fuses

Settled 2026-09-09, when the board arrived. It is a YH2204A-class 3S with a mode switch:
**same-port** bridges C- and P- into one node and is rated 50 A each way, **split-port**
keeps them separate for 60 A. Both are the vendor's own wiring diagrams - bridging those
two terminals is a documented mode, not a bodge.

**It is wired same-port, and the reason is the fused P+ node's second job.** The rating is
noise: peak draw is ~35 A and the fuse below opens at 30, so neither 50 nor 60 is
reachable. What decides it is that the node is a *bench supply input* as well as the load
output (the tree above). A supply sitting above pack voltage there pushes current
into P-. In split-port the charge FETs are not in that path and the discharge FET's body
diode conducts in exactly that direction - so it charges the pack with the overcharge
cutoff completely out of the loop. Same-port puts both FET banks in series, so that same
current is protected. It costs ~2.5 W more dissipation in the board at 35 A, in a closed
tray beside the BMS bay (`mini_dog.py:1085`) - **verify** it against a real thermal soak.

For the record, per use case: split-port is marginally better for running on battery (half
the FET loss) and for charging while running (independent limits); same-port is decisively
better for the bench supply. Only the last of those can destroy a pack.

**Two fuses, because the charge branch is a sixth of the main lead.** Same-port is about the
BMS's *negative* terminals, not about the robot having one connector - the load lead and
the XT30 stay separate leads whose negatives merely meet at the shared node, so a fuse in
the XT30's *positive* branch carries charge current only.

```
pack B+ --[30 A MIDI]--+----------------- P+ node   load / bench supply (inside)
                       +--[7.5 A ATO]---- XT30 +   charger

pack B- -- BMS B- =[chg FETs]=[dis FETs]= C-/P- bridged --+-- P- node
                                                          +-- XT30 -
```

Without the second fuse the XT30's 5 A lead (`3d/README.md:270`) sits behind 30 A of
protection: a resistive fault holds 25 A in it indefinitely and nothing upstream ever
opens, because a 30 A fuse *carries* 30 A - it needs roughly 2x for seconds. The BMS is no
help either, its OCP is 50 A. And never put a fuse in the shared part - the common
negative, or the trunk before the branch point - that is the 30 A fuse's job and it is the
only one that sees both currents.

### Ordered 2026-09-09

| part | for | RUB |
|---|---|---|
| Derzhatel Midival 30-80 A, OEM 0300360 | main fuse holder | 337 |
| ELF Midi 30 A 32 V, 2 sht | the 30 A main fuse, + spare | 240 |
| REXANT vlagozashchitnyy na provode | charge-branch holder | 139 |
| TESLA ATO 5-40 A, 12 sht + shchiptsy | supplies the 7.5 A, + spares | 243 |
| Provod 12 AWG silicone, 3 m red + 3 m black | main lead, 3.4 mm2 | 1455 |
| NKI 6.0-6 ring terminals, M6, 10 sht | onto the Midival studs | 126 |

2540 RUB total. **MIDI, not mini-ANL**, and that is the one non-obvious choice: mini-ANL is
a car-audio format whose stocked range starts around 80 A, so a 30 A one is hard to buy at
all. MIDI at 30 A is a commodity, and a 30 A fuse in a 30-80 A holder runs nothing at its
limit - unlike a 30 A blade in a 30 A-max holder, which is exactly at its own.

**Three things to check when it arrives**, none of them settled:

| | why it is open |
|---|---|
| the Midival's stud size | M6 rings were bought on a guess. They drop onto an M5 or M4 stud with a washer, so only an M8 defeats them |
| NKI 6.0 crimped onto 12 AWG | the barrel is 4-6 mm2, the conductor is 3.3. Fold the stripped end back to fill it - do not crimp it loose, this is the joint the whole pack's fault current crosses |
| the mass of holder + fuse + leads | it lands in `ELECTRONICS_KG`, and `3d/CLAUDE.md` has 11 g moving the flat trot 778 -> 597 mm. Weigh it, do not estimate it |


## The servos are not regulated, and that is a decision

Everything from the P+ node to the twelve ST3215s is raw pack, sagging 12.6 → 9.9 V over the
discharge. Do not "fix" this:

- **Supply voltage is a gait variable.** The bench measured position-loop stiffness at
  **28.8 N·m/rad at 12.1 V and 24.5 at 10.1 V** — linear in supply
  ([`ST3215_STS3215_measured_parameters.md`](ST3215_STS3215_measured_parameters.md), and
  `PLAN.md`'s note on the battery). The robot gets ~25 % softer as it discharges. That
  belongs in domain randomisation, not in a regulator.
- **The identification set was captured at three supply voltages on purpose**
  (`robot/README.md`, *The three voltages are not thoroughness*). A regulated bus would
  make that work describe a robot nobody is building.
- And the current is absurd: see the table below.

## What the bus actually draws

Every number in this table is measured, on one robot at **1.55 kg with no battery, no Pi
and no payload**. `robot/README.md`, *What the robot weighs today is not what the limits are
for*, is explicit that *nothing* measured under load at that mass transfers to 2.499 kg — not the current, not the sag. Read these as the right
order of magnitude and the wrong absolute.

| | | source |
|---|---|---|
| whole robot standing, 1.55 kg | **0.4 A at 12 V** | `robot/README.md:405`, at the bench supply |
| ... of which the twelve motors | ~0.07 A | same |
| ... so quiescent, per servo | **~27 mA** | same — `PRESENT_CURRENT` does not see the servo's own electronics |
| trot at 2.55 kg | **0.86 A peaks** | `robot/README.md:469` — off register 69, which is *supply* current; that peak was one servo, see the next rows |
| **untethered profile trot, on the pack** (Pi, IMU, no plate; 0.11 m/s, period 1.35 s) | **sum of twelve: p50 0.25, p95 0.63, peak 1.1 A**; one servo peaks at 0.66 A (`fl_pitch`) | `bench/pack_sag.py` on `bench/data/trot_pack_full_{a,b}.npz`, two runs, 2026-09-14 |
| ... standing, same run | **0.09 A** on the bus, 12.1 V | same — the bench supply's 0.4 A was the adapter and the servo electronics, which register 69 does not see |
| ... sag at the servo | **0.35–0.40 V** (12.10 → 11.70 V) | same; `R_eff` **310–330 mΩ** from bus-min V against summed A (r² 0.7, the draw only spans 1 A) — pack plus the P+ run plus the servo's own input, measured where the servo sees it |
| one servo stalled | **2.7 A** | `robot/README.md:716` |
| `PROTECTION_CURRENT` as configured | 310 counts ≈ **2.0 A** | register dump in the parameters doc; the 6.5 mA LSB is documented for register 69, **verify** that it is the same for this one |

So the standing draw is nothing and the transient is everything: twelve servos at their
protection limit is ~24 A, and a four-leg push-off is a large step change into the pack's
internal resistance. **That dip is what the converter has to ride through**, and it lands
in the middle of the 50 Hz loop.

**Measured, at a full pack:** the analytic trot draws about a tenth of what the
paragraph above budgets for — the sum never passed 1.1 A and the servo-side sag was 0.4 V.
At this gait the pack is not what limits anything; the same recording says the servos
run **20–25° behind the goal at p95** on every pitch and knee (peak 31°), because the
trot is scheduled at 3.09 of 3.28 rad/s available and the loaded servo does not deliver
that. The transient case — a push-off that recruits several servos at once — has not
happened yet because no gait on this robot pushes off. The servos are the meter, each
reporting its supply volt and current in the feedback the loop reads anyway:

```bash
python runtime/walk.py --port /dev/ttyACM0 --profile --log bench/data/trot_pack_full.npz
python bench/pack_sag.py bench/data/trot_pack_full.npz   # sag, R_eff, draw by gait phase
```

Still to do: the same run near-empty (the ~25 % softer robot), and any gait that
actually pushes off. The 5 V budget below is still sized off a component list.

## The 5 V rail

| load | draw | confidence |
|---|---|---|
| Orange Pi 5 Pro | 5 V, **~1.8 A at full CPU load** (1.1 A into the buck at 9 V, 8 cores busy, 2 min, no throttle); vendor says 4 A class | measured on the buck, bench supply reading back |
| IMX415 camera, USB UVC | ≤ 0.5 A | **verify** — bus-powered, vendor gives no figure |
| GY-NEO6MV2 GPS + active patch | ~50 mA | **verify** — a bazaar part, like every other number on it (`mini_dog.py:651`) |
| BMI088 IMU | ~5 mA, and **not on this rail** — see below | |

**So: one 12 → 5 V converter is enough for the Pi and the IMU, on three conditions.**

1. **Size it at 5 A, not 3 A.** ≥25 W out is ~2.5 A in at 12 V at 90 %. A cheap module
   badged 5 A is nearer 3 A continuous with no airflow, inside a closed printed body, next
   to the cells. Derate it.
2. **Buy the input range, not the label.** A part specified around a 12 V nominal input can
   drop out at the bottom of a 3S discharge, which is 9.9 V before any transient. Want a
   wide-input buck that still regulates in the 7–8 V region, or a buck-boost.
3. **Wire it for the transient**, per the next section. This is the condition that actually
   gets skipped, and its symptom is a Pi that reboots when the robot pushes off.

The 5 Pro takes a dumb 5 V source: it boots and runs full-load from the XL4015 with no
PD in the loop (the kernel exposes no `typec` port at all). Measured on the buck: OUT
held at the same voltage at 9.0 V in as at 12 V in, so the pack floor is inside the
converter's headroom. It is trimmed to **5.0 V under load** — the module as shipped read
4.86 under load, 0.1 V above USB's floor.

## The node, and how the joints are made

**The P+ and P− nodes are two WAGO 221-415 lever nuts** (5 × 0.14–4 mm² fine-stranded,
32 A; 29.9 × 18.6 × 8.3 mm — vendor sheet, **verify** on the part), hot-glued in the
rear strip standing on the upper cradle-screw seats, one each side of the centreline,
levers inward (`NODE_*` in `mini_dog.py`; `node_clear()` checks the box every build).
Five positions each: the pack lead in, four legs out. That is all the strip holds — it is
17.6 mm deep and 14.8 behind the seats, and a 221 needs 18.6 along the wire, so a second
pair cannot stand there. So the charge branch (7.5 A ATO → XT30) and the buck's feed take
**ring terminals on the MIDI holder's load stud** (the NKI 6.0s), and the adapter's logic
takes the buck's own IN terminals. Nothing solders onto the 12 AWG.

Why not a solder splice or perfboard for the node: a 12 AWG + 4 × 18 AWG joint needs more
heat than a hand iron delivers, solder wicks up stranded wire and every lead then breaks
at the stiff edge, and nothing is serviceable; perfboard pads carry nothing like 30 A and
the holes do not take 12 AWG. A drone PDB does the job for ~1000 RUB and adds only what
the stud already gives.

Per joint:

| joint | how |
|---|---|
| node | 221-415, one conductor per clamp, strip 11 mm |
| charge branch, buck feed | crimped rings on the MIDI holder's load stud |
| leg splice at the hip (18 AWG → tails to `*2`, `*3`; `*1` gets its own tail in the tray) | lineman splice, soldered, adhesive-lined heatshrink, a tie ≤ 20 mm either side |
| servo plugs | 5264 crimp housings, or the crimped ends of stock Feetech leads; **D only** on the jumpers between servos — ground comes back on the 18 AWG |
| buck, adapter, ATO holder | the boards' screw terminals, ferrules |
| bus star, D + G from the adapter's two ports | one Y-lead per port |

Crimp what flexes; solder only what is tied down on both sides; no joint inside a moving
span. The 470–1000 µF 25 V low-ESR sits on the buck's IN terminals, 12 V side — that is
the nearest screw to the node it can reach. Near the Pi the useful capacitor is the buck's
OUT one; a 12 V bulk cap there is on the wrong rail.

## Wiring, which is where the brownout is won or lost

- **The leg pairs are the fix for the daisy chain.** As first wired, `*2` and `*3` drew
  V+/GND through `*1`'s 5264 connector (~2.5 A a contact, **verify**): three stalled
  servos is 8 A through it. One 18 AWG pair per leg from the node, out with the `*2` lead
  across the roll arm (`3d/README.md`, *Cables*), spliced at the hip bracket. Check before
  rework: log `PRESENT_POSITION`'s voltage byte on `*1` vs `*3` in a loaded trot.
- **Star at the pack, not at the far end of the bus.** Take the converter's feed at the
  fused P+ node, not off a servo bus branch: every milliohm of the servo run turns those
  24 A transients into volts at the converter's input.
- **Bulk capacitance on both sides of the buck** — a few hundred µF low-ESR in, some out.
  The input cap is what carries the Pi through the push-off dip.
- **Single-point ground.** Servo return and the Pi's ground meet once, at the pack. The
  IMU's ground is the Pi's, because it is powered from the Pi.
- **Never through the URT-1.** `robot/README.md:716` — a stalled ST3215 draws 2.7 A and a
  debug adapter is not a power distribution board. The same sentence applies to powering
  anything else off it.

## The IMU is not a rail

The BMI088 hangs off the **Pi's 3.3 V pin** at a few mA. It needs no converter of its own
and it must not be fed 5 V: the part is 3.3 V, and `3d/ref/imu/README.md` has it going in
headerless with six wires soldered straight to the pads — VCC, GND, SDI, SCK and **two**
chip selects, because the accelerometer and the gyroscope are two devices on one SPI bus.
Being on the Pi's own 3.3 V is also what gives the bus a shared ground, which is what SPI
wants.

(If the breakout in hand carries its own regulator and tolerates 5 V, that is still not a
reason to use it: the logic is 3.3 V either way, and the Pi's GPIO is 3.3 V. **verify**
against the real board when it arrives.)

The slot it lives in is **3.6 mm** from the pack's top to the deck's underside and the
board spends 2.8 of it. Nothing else — no converter, no connector, no header — goes there.

## The LiDAR wants its own rail

`3d/ref/lidar/README.md:116`: ethernet here is **not** PoE — the L2 takes **12 V on a
DC3.5-1.35 barrel**, and the L2 manual's own specification line is 12 V 10 W
(`mini_dog.py:640`).

A 3S pack is 12 V for the first few minutes of a discharge and 9.9 V at the end of it. So
raw pack into the L2 is a sensor that browns out at low state of charge — and low state of
charge correlates with having been walking, which is exactly when the point cloud matters.
**verify** the L2's input tolerance against the manual; if it is not comfortable down to
~9.5 V, a second converter (a ~15 W boost or buck-boost to a regulated 12 V) is not
optional. This is the part of the answer that "one 12 → 5 V converter" does not cover.

## What it costs the CAD

`ELECTRONICS_KG = 0.200 + 0.032` is the Orange Pi in its case plus the harness and bus
adapter (`mini_dog.py`, section 4); the node's two lever nuts and the leg pairs are inside
it. **The buck has its keep-out**: `BUCK_KG` = 18 g at `buck_com()`, on the Pi's case top
under the GPS platform (`3d/README.md`, *Payload bays*), taken out of that allowance so
the total did not move. The bus adapter and the fuse holder still have none, and
`3d/CLAUDE.md`'s rule holds for them: anything the robot carries lives once, in
`mini_dog.py`, as a modelled envelope like `OPI_BOX`, through the usual ladder (rebuild →
FEA → `export_sim.py --check` → regenerate the ROS 2 description) in the same pass.

Two placement constraints that held: not in the deck slot, and a buck under load is a heat
source, so not against the cells — the case top is on the far side of the Pi from them.

## Where this sits in the plan

- **Step 7, the IMU** (`PLAN.md`): needs none of this. The 50 Hz loop closes from the mac
  over CDC, and "no battery, so no need" for the Pi is still the right call.
- **Steps 8–10, pack → Pi → LiDAR**: need all of it, in that order. The converters are
  bought against this file's **verify** list, and every one of them is a five-minute check
  against a manual or a meter.

## The verify list, in one place

| | how to settle it |
|---|---|
| L2 input voltage tolerance | the L2 manual; it decides whether rail #2 exists |
| camera and GPS draw on the 5 V rail | meter, once the Pi is up |
| `PROTECTION_CURRENT` LSB | the Feetech register table, against a clamp meter on a stalled servo |
| pack sag at the terminals during a trot, at 2.499 kg | the one measurement that sizes everything above |
| the Midival holder's stud size, against the M6 rings | a caliper, when it arrives |
| BMS dissipation at 35 A in same-port, in a closed tray | a thermal soak on the bench, not the 4 mOhm datasheet line |
| mass of the fuse holders, fuses and leads | a scale, into `ELECTRONICS_KG` |
