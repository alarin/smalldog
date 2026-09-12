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
    +--> raw pack ------> 12 x ST3215 on one bus          UNREGULATED, and deliberately so
    |
    +--> buck 12->5 V ---> Orange Pi 5 Pro
    |                        +-- 3.3 V header pin --> BMI088 IMU        (SPI, 2 x CS)
    |                        +-- USB --------------> IMX415 camera
    |                        +-- ------------------> GY-NEO6MV2 GPS     **verify** rail
    |
    +--> 12 V, regulated -> Unitree L2                    see *The LiDAR wants its own rail*
```

One connector leaves the tray. `PANEL_AT` puts the **XT30 above the cradle** at z = +20
on the centreline (`3d/README.md`, *Payload bays*); the bus window is the only other
opening. There is no XT60 - the load lead never leaves the tray - and no balance
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
| trot at 2.55 kg | **0.86 A peaks** | `robot/README.md:469` — off register 69, which is *supply* current; **verify** whether the logged peak is one servo or the sum |
| one servo stalled | **2.7 A** | `robot/README.md:716` |
| `PROTECTION_CURRENT` as configured | 310 counts ≈ **2.0 A** | register dump in the parameters doc; the 6.5 mA LSB is documented for register 69, **verify** that it is the same for this one |

So the standing draw is nothing and the transient is everything: twelve servos at their
protection limit is ~24 A, and a four-leg push-off is a large step change into the pack's
internal resistance. **That dip is what the converter has to ride through**, and it lands
in the middle of the 50 Hz loop.

**Not measured, and it is the number this file most wants:** the whole-robot supply
current and the pack-terminal sag during a trot at the design mass. It needs the pack, and
it is one afternoon at the bench with a supply that reads back. Until then the 5 V budget
below is sized off a component list, not off the robot.

## The 5 V rail

| load | draw | confidence |
|---|---|---|
| Orange Pi 5 Pro | 5 V, **4 A class** (20 W) | **verify** — vendor figure for the Type-C port |
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

Also **verify** how the 5 Pro's Type-C port takes power — whether it accepts a dumb 5 V
source or expects a PD negotiation it will not get from a buck module. If in doubt, feed
the 5 V and GND header pins instead and accept that this bypasses the board's own input
protection.

## Wiring, which is where the brownout is won or lost

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

`ELECTRONICS_KG = 0.25` is "Orange Pi 5 Pro / BMS / wiring" (`mini_dog.py:639`). A 5 A buck
module at 15–25 g fits inside that allowance without moving the mass budget — but **there
is no keep-out for it anywhere in the model**, and `3d/CLAUDE.md`'s rule is that anything
the robot carries lives once, in `mini_dog.py`. If a converter goes inside the body it
becomes a modelled envelope there, like `OPI_BOX` is for the Pi, and it goes through the
usual ladder (rebuild → FEA → `export_sim.py --check` → regenerate the ROS 2 description)
in the same pass.

Two placement constraints that are already decided: not in the 3.6 mm deck slot (the IMU
has it), and a buck under load is a heat source, so not against the cells.

## Where this sits in the plan

- **Step 7, the IMU** (`PLAN.md`): needs none of this. The 50 Hz loop closes from the mac
  over CDC, and "no battery, so no need" for the Pi is still the right call.
- **Steps 8–10, pack → Pi → LiDAR**: need all of it, in that order. The converters are
  bought against this file's **verify** list, and every one of them is a five-minute check
  against a manual or a meter.

## The verify list, in one place

| | how to settle it |
|---|---|
| Orange Pi 5 Pro input: 5 V/4 A, and dumb-5 V vs PD | vendor spec, then a bench supply and a meter |
| L2 input voltage tolerance | the L2 manual; it decides whether rail #2 exists |
| camera and GPS draw on the 5 V rail | meter, once the Pi is up |
| `PROTECTION_CURRENT` LSB | the Feetech register table, against a clamp meter on a stalled servo |
| pack sag at the terminals during a trot, at 2.499 kg | the one measurement that sizes everything above |
| the Midival holder's stud size, against the M6 rings | a caliper, when it arrives |
| BMS dissipation at 35 A in same-port, in a closed tray | a thermal soak on the bench, not the 4 mOhm datasheet line |
| mass of the fuse holders, fuses and leads | a scale, into `ELECTRONICS_KG` |
