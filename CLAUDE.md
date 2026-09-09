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

## Three machines, one repository (and a fourth, unconfigured)

The CAD lives on a mac, training on a Windows/WSL2 box with an RTX 3070, the runtime on
the robot's Orange Pi. There is no shared filesystem between them and no scp — **the git
repository is the only thing that crosses**, so a result that has to reach another
machine gets committed.

Work out which machine you are on before running anything. On the mac, read
[`MAC.md`](MAC.md) first — it is the queue of CAD changes that have not yet been through
CadQuery, and two of them move printed geometry. On the WSL2 box, read
[`WSL.md`](WSL.md) first: it says which of the per-tree verification steps that machine
can actually run, which environment they run in, and which it cannot build at all.

**A fourth machine exists but is not set up: a separate box, 35 GB RAM, RTX 5070 Ti,
16 GB VRAM.** No checkout and no session on it yet, so nothing in this repository has
ever run there and no runbook describes it. It is recorded here because of what it would
unblock rather than as a machine to reach for: the 3070's 8 GB is the stated reason
`rl/` has never launched the box-terrain training run it made possible in `cf5f236` —
adding collision pairs to a 2048-env model on that card is the shape of an OOM that has
taken it down before — and 16 GB removes that constraint.

Two things to settle there before planning anything around it, in this order:

- **The 5070 Ti is Blackwell, compute capability 12.0**, needing CUDA 12.8+ and a jax
  built for `sm_120`. The 3070 is Ampere (`sm_86`), so the working install on the WSL2
  box is no evidence at all that the same wheels run here. Confirm `jax.devices()` sees
  the GPU *and* that a trivial jit actually executes — a mismatch shows up as a silent
  fallback to CPU or a PTX error at first compile, not at import, so an install that
  looks fine can be running on the CPU at a hundredth of the speed.
- **Measure the VRAM for the 2048-env model with boxes.** That number is the one `rl/`
  said was missing, and it is the whole reason this machine is interesting.

Neither the env count nor the `--boxes` sizing carries over from the 3070: `BOX_PATCH_M`
scatters ±4 m, which works out to 0.22 boxes met per episode, so the curriculum needs
sizing before a terrain run means anything on any card. If this box gets a checkout, it
needs its own runbook file alongside `MAC.md` and `WSL.md` — do not let it inherit the
3070's assumptions by default.
