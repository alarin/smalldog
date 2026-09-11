# SmallDog

A 12-DOF printed quadruped ("mini robot dog") built around 12 × Feetech/Waveshare
**ST3215** bus servos — parametric CAD, FEA, a ROS 2 / MuJoCo simulation, an RL policy and
the hardware runtime, all of one robot.

| | |
|---|---|
| [`3d/`](3d/) | **the source of truth.** Pure code-CAD (CadQuery): every dimension, mass and joint limit lives in [`3d/mini_dog.py`](3d/mini_dog.py). Also linear-static FEA ([`fea.py`](3d/fea.py)) and the URDF/MJCF exporter ([`export_sim.py`](3d/export_sim.py)). |
| [`ros2/`](ros2/) | ROS 2 workspace: generated description, `ros2_control` wiring, trot gait, keyboard teleop, MuJoCo standing in for the hardware. |
| [`rl/`](rl/) | the RL walking policy: MJX/Brax PPO against the same generated model. |
| [`robot/`](robot/) | the hardware side: ST3215 bus driver, the identification bench, the 50 Hz runtime. |

Everything structural is FDM-printed (PETG/ASA): no aluminium, no machining, no external
bearings — only the stock servos and their aluminium hubs.

## One model, three consumers

`ros2/smalldog_description` is **generated**, not authored:

```bash
cd ros2 && ../3d/.venv/bin/python smalldog_description/scripts/generate_model.py
```

Masses, inertias, link lengths, joint limits, the actuator constants and the sensor
parameters are all read out of `3d/mini_dog.py`. A number that is wrong in simulation is
wrong in the CAD — fix it there and re-export. [`3d/CLAUDE.md`](3d/CLAUDE.md) has the
change workflow (rebuild → FEA → re-export sim → regenerate the ROS 2 description).

## Machines

| machine | does | runbook |
|---|---|---|
| a mac | CAD, FEA, both sim exporters, the ROS 2 workspace | [`3d/CLAUDE.md`](3d/CLAUDE.md) |
| Windows/WSL2, RTX 3070 | RL training, the pure-Python sim regressions | [`WSL.md`](WSL.md) |
| Orange Pi 5 Pro | the runtime on the robot | [`robot/README.md`](robot/README.md) |

No shared filesystem between them: the git repository is the only thing that crosses.

## Where to read

- Printed parts, servo interface, assembly: [`3d/README.md`](3d/README.md)
- Simulation, gait, teleop, regressions: [`ros2/README.md`](ros2/README.md)
- Bench, bring-up, the measured servo: [`robot/README.md`](robot/README.md) and the public
  reference [`ST3215_STS3215_measured_parameters.md`](ST3215_STS3215_measured_parameters.md)
- Power tree and its **verify** list: [`POWER.md`](POWER.md)
- What to build next: [`PLAN.md`](PLAN.md)

## Not in this repository

Generated output is ignored — `3d/out/` (STEP/STL/BOM/renders/FEA meshes), `rl/runs/`
(checkpoints) and the colcon `build/`, `install/`, `log/` trees. Re-run the scripts to
produce them.

`3d/ref/` holds vendor downloads (ST3215 STEP + drawings, Waveshare ROBOTIC DOG STEP) and
transcribed measurements of bought parts, used as inputs only; they remain the property of
their respective vendors.
