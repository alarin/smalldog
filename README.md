# SmallDog

A 12-DOF printed quadruped ("mini robot dog") built around 12 × Feetech/Waveshare
**ST3215** bus servos — parametric CAD, FEA and a ROS 2 / MuJoCo simulation of the
same robot.

| | |
|---|---|
| [`3d/`](3d/) | **the source of truth.** Pure code-CAD (CadQuery) — every dimension, mass and joint limit lives in [`3d/mini_dog.py`](3d/mini_dog.py). Also linear-static FEA ([`fea.py`](3d/fea.py)), the URDF/MJCF exporter ([`export_sim.py`](3d/export_sim.py)) and the servo identification bench ([`servo_bench.py`](3d/servo_bench.py), spec in [`3d/BENCH.md`](3d/BENCH.md)). |
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
- Talking to the hardware, and identifying the servo: [`robot/README.md`](robot/README.md)
- The bench that identification runs on, as printed parts: [`3d/BENCH.md`](3d/BENCH.md)

## Where the numbers are still guesses

Every *structural* dimension in this repo is measured. Every *actuator* number is not:
`rl/params/st3215.json` is committed as vendor priors and says so — `"fitted": false` —
and the four MuJoCo `MJ_*` constants in `3d/export_sim.py` are flagged as estimates. That
is the ceiling on the simulation and on anything trained in it.

[`robot/bench`](robot/README.md) is the pipeline that replaces them: drive one servo
through trajectories designed around the degeneracies, then fit. It needs hardware to run
on, and [`3d/BENCH.md`](3d/BENCH.md) is that hardware — a printed portal that hangs one
ST3215 as a pendulum in the robot's own sleeve and fork, so what gets identified is this
robot's joint rather than a servo in a vice.

## Not in this repository

Generated output is ignored — `3d/out/` (STEP/STL/BOM/renders/FEA meshes) and the
colcon `build/`, `install/`, `log/` trees. Re-run the scripts above to produce them.

`3d/ref/` holds vendor downloads (Feetech/Waveshare ST3215 STEP + drawings, Waveshare
ROBOTIC DOG STEP) used as measurement inputs only; they remain the property of their
respective vendors.
