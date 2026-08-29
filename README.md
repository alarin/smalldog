# SmallDog

A 12-DOF printed quadruped ("mini robot dog") built around 12 × Feetech/Waveshare
**ST3215** bus servos — parametric CAD, FEA and a ROS 2 / MuJoCo simulation of the
same robot.

| | |
|---|---|
| [`3d/`](3d/) | **the source of truth.** Pure code-CAD (CadQuery) — every dimension, mass and joint limit lives in [`3d/mini_dog.py`](3d/mini_dog.py). Also linear-static FEA ([`fea.py`](3d/fea.py)), the URDF/MJCF exporter ([`export_sim.py`](3d/export_sim.py)) and the servo test stands ([`servo_bench.py`](3d/servo_bench.py), spec in [`3d/BENCH.md`](3d/BENCH.md)). |
| [`ros2/`](ros2/) | ROS 2 workspace: description, `ros2_control` wiring, trot gait, keyboard teleop, MuJoCo standing in for the hardware. |

Everything structural is FDM-printed (PETG/ASA): no aluminium, no machining, no
external bearings — only the stock servos and their aluminium hubs.

## The two are one model

`ros2/smalldog_description` is **generated**, not authored:

```bash
cd ros2 && ../3d/.venv/bin/python smalldog_description/scripts/generate_model.py
```

Masses, inertias, link lengths and joint limits are read out of `3d/mini_dog.py` and
its ROM scan. A link mass or a joint limit that is wrong in simulation is wrong in the
CAD — fix it there and re-export. See [`3d/CLAUDE.md`](3d/CLAUDE.md) for the full
change workflow (rebuild → FEA → re-export sim → regenerate the ROS 2 description).

## Getting started

- Printed parts and the servo interface: [`3d/README.md`](3d/README.md)
- Simulation, gait and teleop: [`ros2/README.md`](ros2/README.md)
- Measuring the servo, so the gait can stop guessing: [`3d/BENCH.md`](3d/BENCH.md)

## Where the numbers are still guesses

Every *structural* dimension in this repo is measured. Every *actuator* number is not:
stall torque, no-load speed and the four MuJoCo `MJ_*` constants are vendor figures or
estimates, and the joint's real compliance and backlash are nowhere in the model at all.
That is the ceiling on both the simulation and on any attempt to replace the analytic leg
IK with a search over joint-space motions — a search can only be as good as the actuator
model it filters against. [`3d/BENCH.md`](3d/BENCH.md) is the CAD and the BOM for three
printed stands that measure them, and a map from each measurement to the constant it
replaces.

## Not in this repository

Generated output is ignored — `3d/out/` (STEP/STL/BOM/renders/FEA meshes) and the
colcon `build/`, `install/`, `log/` trees. Re-run the scripts above to produce them.

`3d/ref/` holds vendor downloads (Feetech/Waveshare ST3215 STEP + drawings, Waveshare
ROBOTIC DOG STEP) used as measurement inputs only; they remain the property of their
respective vendors.
