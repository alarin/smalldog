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
    +-- XT60   master disconnect / bench supply, on the pack's fused P+
    +-- XT30   charge only, deliberately the smaller shell
    +-- JST-XH 3S balance lead, passed through outside the tray
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

The three ways out of the tray are the ones the CAD already cut: `PANEL_AT` puts the XT60
under the bus window and the XT30 beside it, and the two shells differ so a charger
physically cannot be plugged into the bus (`mini_dog.py:250`).

## The servos are not regulated, and that is a decision

Everything from the XT60 to the twelve ST3215s is raw pack, sagging 12.6 → 9.9 V over the
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
  XT60 / fused P+, not off a servo bus branch: every milliohm of the servo run turns those
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
