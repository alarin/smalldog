# CLAUDE.md

A 12-DOF printed quadruped on 12 × Feetech ST3215 servos. `README.md` is the
human-facing overview; this file only routes.

## The trees, and which rules apply

| tree | what it is | read first |
|---|---|---|
| `3d/` | **the source of truth** — code-CAD, FEA, the URDF/MJCF exporters | `3d/CLAUDE.md` |
| `ros2/` | ROS 2 workspace + MuJoCo. Its `smalldog_description` is **generated** from `3d/`, never authored | `ros2/README.md`, and `3d/CLAUDE.md` step 6 |
| `rl/` | the RL walking policy — the third consumer of the same CAD | `rl/CLAUDE.md` |
| `robot/` | the hardware side: bus driver, bench, the 50 Hz runtime | `robot/README.md` |

`3d/CLAUDE.md`'s rules about what is source and what is output hold across the whole
repository, not just inside `3d/`. Anything under `ros2/smalldog_description/` or
`3d/out/` is output: fix it in `3d/mini_dog.py` and regenerate.

## Three machines, one repository

The CAD lives on a mac, training on a Windows/WSL2 box with an RTX 3070, the runtime on
the robot's Orange Pi. There is no shared filesystem between them and no scp — **the git
repository is the only thing that crosses**, so a result that has to reach another
machine gets committed.

Work out which machine you are on before running anything. On the mac, read
[`MAC.md`](MAC.md) first — it is the queue of CAD changes that have not yet been through
CadQuery, and two of them move printed geometry. On the WSL2 box, read
[`WSL.md`](WSL.md) first: it says which of the per-tree verification steps that machine
can actually run, which environment they run in, and which it cannot build at all.
