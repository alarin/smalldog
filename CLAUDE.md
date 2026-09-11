# CLAUDE.md

A 12-DOF printed quadruped on 12 × Feetech ST3215 servos. `README.md` is the human-facing
overview and the machine map; this file only routes.

## The trees

| tree | what it is | read first |
|---|---|---|
| `3d/` | **the source of truth** — code-CAD, FEA, the URDF/MJCF exporters | `3d/CLAUDE.md` |
| `ros2/` | ROS 2 workspace + MuJoCo. `smalldog_description` is **generated** from `3d/`, never authored | `ros2/README.md` |
| `rl/` | the RL walking policy — the third consumer of the same CAD | `rl/CLAUDE.md` |
| `robot/` | the hardware side: bus driver, bench, the 50 Hz runtime | `robot/README.md` |

`3d/CLAUDE.md`'s rule about what is source and what is output holds across the whole
repository: anything under `ros2/smalldog_description/` or `3d/out/` is output — fix it in
`3d/mini_dog.py` and regenerate.

## Machines

Three machines, no shared filesystem, no scp: **the git repository is the only thing that
crosses**, so a result that has to reach another machine gets committed.

| machine | does | runbook |
|---|---|---|
| a mac | CAD, FEA, both sim exporters, the ROS 2 workspace | `3d/CLAUDE.md` |
| Windows/WSL2, RTX 3070 (8 GB) | RL training, the pure-Python sim regressions | `WSL.md` |
| Orange Pi 5 Pro | `robot/runtime` on the robot | `robot/README.md` |

Work out which one you are on before running anything; `WSL.md` has the check.

A fourth box (RTX 5070 Ti, 16 GB VRAM, 35 GB RAM) exists with no checkout and no runbook.
It matters because 8 GB is the reason `rl/` has never launched the box-terrain run. Before
planning anything around it: confirm an `sm_120` jax actually executes a jit on that GPU
(CUDA 12.8+; a mismatch is a silent CPU fallback, not an import error), and measure the
VRAM of the 2048-env model with boxes. It gets its own runbook when it gets a checkout.

## Conventions for these files

- **Dated history lives in `git log`, not here.** A doc carries the current rule, the
  current number and a one-line reason. A superseded number is deleted, not annotated.
- A number marked **verify** is unmeasured. Keep the marker until it has met hardware.
- When a bench number changes it goes in two places: the tree that uses it, and
  `ST3215_STS3215_measured_parameters.md`, the public reference.
- Commit only when asked. The mac and the WSL2 box share `main`: `git pull --rebase` first.
