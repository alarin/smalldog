# Mini Robot Dog — Codex Handoff

## Goal
Build a low-cost mini quadruped mainly as a development platform for:
- gait / IK / locomotion basics
- ROS 2 integration
- 3D LiDAR SLAM
- autonomous navigation

Priority is to get a working prototype quickly, not to make final production mechanics.

## Current mechanical direction
- Use the **Waveshare ST3215 12-DOF robotic dog STEP model** as the starting point rather than redraw V1 from scratch.
- Image/file naming around the model includes **“WAVEGO PRO BETA v3”**, but the useful source is the ST3215 robotic-dog CAD from the Waveshare ST3215 resources page.
- First prototype should be **fully 3D printed structurally**.
- User explicitly does **not** want aluminium/frezing for V1.
- If something flexes or breaks, reinforce/redesign only that part later.
- Keep the original geometry/kinematics initially, then iterate.

### Uploaded CAD files
- `ROBOTIC DOG.step`
- `ST3215.step`

The ST3215 servo CAD should be used when making/adjusting servo pockets and joint geometry.

## Servos
Chosen actuator:
- **Feetech / compatible STS3215 / ST-3215**
- 12 V version
- 30 kg·cm stall torque
- ~10 kg·cm rated torque
- 0.222 s / 60°
- 55 g
- 12-bit magnetic encoder
- 25T output spline
- TTL half-duplex serial bus
- nominal 12 V, allowed about 4–14 V
- stall current about 2.7 A
- rated current about 0.9 A
- gearbox about 1:345

Planned quantity:
- **12 servos**, 3 DOF per leg.

Important limitation:
- These servos are **not meaningfully backdrivable** because of the high reduction.
- That is acceptable because this prototype is mainly for SLAM/navigation/IK/gait, not force-control research.

## Joint support / “second hub” conclusion
Important latest conclusion:

STS3215 has:
- one real driven 25T output side
- one opposite **passive plastic support pin / auxiliary hub side**

The servo is supplied with two aluminium hub plates:
- driven hub on the 25T output
- passive hub on the opposite plastic support

For **V1**, use the servo exactly this way.

### Do NOT add external bearings yet
We previously discussed 685 / 625 / 628 bearings, but after inspecting the servo geometry/photo:
- **do not design 628 or 685 support bearings into V1**
- use the stock passive hub first
- only add a real external bearing later if the passive plastic side develops too much flex/wear/play

User has many 628 bearings, but current decision is **not to use them unless testing shows a need**.

## Structural design philosophy
For V1:
- print body, hips, thighs, shins
- use supplied aluminium servo horns/hubs
- use normal metal screws/bolts
- keep cable routing accessible
- no belts
- knee servo mounted directly at knee
- servo body should stay on the proximal link (e.g. knee servo on thigh, not shin)

If replacing thin aluminium-looking parts from the reference model with prints:
- make them boxy/ribbed rather than printing a thin plate copy
- ~2.5–3 mm shell walls as a starting point
- local reinforcement around servo pockets / screw bosses

Materials considered:
- ASA: good
- PA / PA-CF: good
- PETG: acceptable for prototype
- PLA not preferred for outdoor/march cold use

## Power
Battery decision to remember:
- **6 × new Molicel 21700 cells**
- configuration: **3S2P**
- nominal pack voltage ~10.8–11.1 V
- full voltage **12.6 V**
- this can feed STS3215 servo bus directly

Preferred cell reference:
- Molicel P42A-class high-drain cells

Why 3S2P:
- more runtime
- less voltage sag
- enough current reserve for 12 servos + computer + LiDAR

## BMS
Target:
- compact cheap **3S Li-ion/NMC BMS**
- around **40–50 A**
- balancing required
- NTC desirable but not mandatory
- no need for expensive Smart/Bluetooth/CAN BMS for V1

A cheap 3S 50 A board was considered acceptable as a prototype BMS.

Important:
- choose **3S Li-ion 3.7 V / 4.2 V per cell**
- NOT LiFePO4
- not 4S
- not BMS that starts at 8S

Recommended system fuse:
- roughly **30–35 A** after battery/BMS

## Servo communications
Chosen interface concept:
- ESP32 → Feetech bus adapter → all 12 servos on one TTL half-duplex bus

Adapter shown/considered:
- **Feetech URT-1**

All servos:
- share power/GND/data bus
- each gets unique ID
- use Sync Write for coordinated joint commands

## Compute / ROS 2
Architecture direction:
- **ESP32**: low-level servo control
- **Orange Pi 5 Pro**: ROS 2 / SLAM / navigation / higher-level control

The Orange Pi is the main onboard computer candidate.

> **Superseded for V1**, and left here because this file is the brief as it was written.
> The runtime drives the servo bus from the Orange Pi directly through the URT-1; there is
> no ESP32 in V1. What that buys, what it costs and what has to hold for it to be safe is
> in [`../robot/README.md`](../robot/README.md). The rear bay is still sized for an ESP32,
> so this is a deferral and not a rejection.

## LiDAR / SLAM
SLAM is a major project goal.

User wants a **3D LiDAR from the beginning**, ideally something that can later be moved to the big robot.

Options discussed:

### Unitree 4D LiDAR L2
Main practical candidate:
- ~230 g
- 360° horizontal coverage
- large vertical FOV
- built-in IMU
- ROS 2 / Point-LIO ecosystem
- easier path to working 3D LiDAR-inertial SLAM
- more expensive than STM ToF chip solutions

### ST VL53L9CX
Interesting alternative:
- tiny flash 3D ToF sensor
- limited FOV per sensor
- would likely need multiple sensors for broad coverage
- much cheaper per sensor than Unitree L2
- much more custom software / synchronization / point-cloud fusion work
- not yet the easy “drop-in 3D SLAM” choice

Current practical bias:
- Unitree L2 for easiest real 3D SLAM
- VL53L9CX is interesting for later distributed sensing

## Outdoor / cold
Prototype may be tested around mid-March in Moscow region.

Main considerations:
- servos should be okay around typical March sub-zero temperatures
- battery should be kept inside body if possible
- do not charge Li-ion cells below 0 °C
- protect electronics/servo connectors from wet snow
- beware condensation when bringing cold robot indoors

## CAD / workflow
User may continue in Codex and CAD tools.

Relevant previous discussion:
- STEP import into Onshape may lose feature history / import poorly
- Parasolid `.x_t` is often better if conversion is needed
- Fusion source is preferable if Waveshare provides it
- Inventor does not directly treat Fusion `.f3d` as native without going through Fusion/export

For immediate work:
1. open/import `ROBOTIC DOG.step`
2. use `ST3215.step` as the exact servo reference
3. preserve reference-model joint center distances for V1
4. redesign parts only where needed for printing/manufacturability
5. initially keep stock passive servo hub support
6. avoid adding unnecessary bearings/aluminium/machining

## Near-term design tasks for Codex/CAD
- Identify every structural part in the Waveshare STEP.
- Separate likely “plate-like” parts from printed housings.
- Convert plate-like geometry into printable ribbed/boxed parts while preserving joint centers.
- Verify servo clearances using `ST3215.step`.
- Verify cable exit clearances for both servo connectors.
- Check collision-free range of hip roll, hip pitch, knee.
- Design mounting volume for:
  - 3S2P 6×21700 battery
  - compact 3S 50 A BMS
  - ESP32 / URT-1
  - Orange Pi 5 Pro
  - future 3D LiDAR mount
- Keep battery low and central.
- Keep LiDAR mount on top, near body center.
- Avoid machining-dependent features in V1.

## Key user preference for this prototype
**Fast, fully printed, iterative prototype. Avoid aluminium machining/frezing unless a real weakness appears in testing.**
