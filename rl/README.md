# SmallDog RL

The third consumer of the CAD, after [`3d/`](../3d) and [`ros2/`](../ros2): an
RL walking policy trained in MuJoCo and run on the robot at 50 Hz, replacing the
analytic IK and the hand-written trot in `ros2/smalldog_walker`.

This tree reads the **generated** description in `ros2/smalldog_description`
(MJCF, meshes, `robot_params.json`) and never writes to it. `3d/CLAUDE.md`'s rule
holds here too: the simulated robot is output of the CAD, and a mass, a limit or
an axis that is wrong in simulation is wrong in `3d/mini_dog.py`. Model changes
that exist only for training — a fitted actuator, backlash, per-environment
randomisation — are applied programmatically through `mujoco.MjSpec` at load
time (see `checks/imu_placement.py` for the mechanism), so nothing hand-tuned
ever lands in a committed XML.

No CadQuery is needed here. That is deliberate: the CAD lives on the mac,
training happens on a CUDA machine, and the repository is the only thing that
crosses between them.

## Environments

```bash
uv sync --extra cpu      # mac: same code on CPU JAX, for smoke tests
uv sync --extra cuda     # WSL2 + RTX 3070: training
uv sync --extra fit      # scipy + matplotlib, for the bench fit
```

The training box is an **RTX 3070 — Ampere, `sm_86`, 8 GB**, under WSL2. Ampere
is covered by every `jax[cuda12]` wheel, so there is no CUDA-version tightrope
here; check the card is actually visible before blaming anything else:

```bash
nvidia-smi                                        # inside WSL
python -c "import jax; print(jax.devices())"      # must list a CudaDevice
```

**8 GB of VRAM is the binding constraint, not the FLOPs.** In MJX, memory goes
as `num_envs` × the model's contact capacity: budget 1024–2048 environments, not
the 4096–8192 the published quadruped configs assume. PPO learns fine at that
width, it just costs wall-clock. Two things bite specifically at 8 GB:

- JAX preallocates 75% of VRAM on the first device call. Set
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` (or `XLA_PYTHON_CLIENT_PREALLOCATE=false`),
  or the first OOM will be unreadable.
- Windows spends VRAM on the desktop and the browser — 0.5–1.5 GB gone before
  training starts, which is a tenth of the card. Train with the browser closed
  and hardware acceleration off in it.

On WSL2 the NVIDIA driver belongs on the **Windows** side; installing a Linux
driver inside WSL overwrites the passthrough and `nvidia-smi` stops seeing the
card. Keep the checkout inside the WSL filesystem (`~/smalldog`), not under
`/mnt/c` — file operations there are an order of magnitude slower, which is felt
on mesh loads and checkpoint writes.

## What MJX can collide, and what it cannot

Measured on the WSL2 box against the locked resolution (mujoco 3.12.0, jax
0.11.1), not assumed:

| scene | `mjx.put_model` |
|---|---|
| `mujoco/scene.xml` (flat) | ok |
| `mujoco/scene_terrain.xml` | `NotImplementedError: (mjGEOM_CYLINDER, mjGEOM_BOX) collisions not implemented` |

**The heightfield is not the problem — it is supported.** The exception names a
geom *pair*, not a geom, and isolating it (load the scene through `MjSpec`, set
`contype = conaffinity = 0` on the two course logs, recompile, `put_model` → ok)
puts the blame on `type="cylinder"`: the two logs of the obstacle course against
the robot's 11 collision boxes. Everything else in that scene, `type="hfield"`
included, goes to the GPU fine.

The robot itself is MJX-clean by construction, and this is worth not breaking:
every mesh in the generated `robot.xml` is `contype="0" conaffinity="0"` — visual
only — and the whole collision set is 4 capsules (shins), 4 spheres (feet) and 11
boxes. The LiDAR and camera cylinders are visual, so they cost nothing.

So there are three surfaces, and the split is deliberate:

- **flat + procedural boxes**, generated in `model.py` — the default training
  surface, because it randomises per environment, which a baked heightfield
  cannot.
- **`scene_terrain.xml`** — available as a second training surface. Fixed
  geometry, but it is the same ground `ros2/tools/standalone_sim.py --terrain`
  reports on, which makes the two comparable.
- **the obstacle course** — vanilla MuJoCo only, in `eval.py`'s sim-to-sim pass.
  The logs are exactly what MJX will not take.

Re-run the check if the lock moves; `put_model` is the arbiter. And never "fix" a
raise by editing the scene — `scene_terrain.xml` and the course are output of
`3d/terrain.py`.

## Watching it learn

Training writes `runs/<name>/ckpt/<step>/` at every eval, and `replay.py` reads
one of those directories the same way it reads a finished run. So the progress
video is two commands:

```bash
python train_ppo.py                                  # 12 evals, 12 checkpoints
python replay.py runs/<name>/ckpt/*/ --robots 96     # a row per checkpoint
```

Front row is step 0 face-planting, back row is the policy that finished, and a
regression between two checkpoints is visible in one frame instead of being a
wobble in a reward curve.

The checkpoints have to be written *during* the run. When `train()` returns there
is one set of weights left and the progression is gone — `--no-checkpoints` exists
but buys nothing worth the loss. Twelve of them cost ~9 MB, in `runs/`, ignored.

Rendering goes through `jaxenv.configure_gl()`, which is the difference between
the card and a software rasteriser on the WSL2 box; `replay.py` prints
`GL_RENDERER` before it starts spending the clock. `../WSL.md`, "Rendering,
viewers", has the measurement and the reason host RAM is the part that matters.

## Checks

Both need only `mujoco` and `numpy`, so they run on any of the three machines.

```bash
python checks/check_model.py                  # audit the MJCF, exit != 0 on failures
python checks/check_model.py --terrain        # ... on the heightfield scene
python checks/check_model.py --view           # ... plus the passive viewer
python checks/check_model.py --shot out.png   # ... plus one offscreen frame
python checks/imu_placement.py                # does the IMU mounting point matter?
```

`--view` opens MuJoCo's passive viewer; on macOS that needs `mjpython`, exactly
as `ros2/tools/view.sh` documents for the standalone sim. `--shot` needs a GL
backend (`MUJOCO_GL=egl` with a GPU, `osmesa` for software) and says so plainly
rather than dying if there is none.

`actuator.py` is the servo model and is the one file the training environment and
the bench fit share: `robot/bench/fit_bam.py` fits its parameters, `rl/model.py`
will install them into MuJoCo, and neither keeps a second copy of the law. It has
its own consistency check:

```bash
python -c "import actuator; actuator._selftest()"
```

`check_model.py` is not a second copy of `export_sim.py --check`, which asks
whether the model is still the robot. This asks whether the model is fit to
train against — which turns on things a standing test cannot see: that the
actuator's dominant inertia is a guess, that the contact parameters the solver
applies are not the ones the model asked for, and that the three joint-limit
ladders mean three different things.

## Layout

| path | state |
|---|---|
| `checks/check_model.py` | model audit — step 2 |
| `checks/imu_placement.py` | how much the IMU mounting point costs — step 2 |
| `actuator.py` | ST3215 voltage law, back-EMF, friction, backlash — step 3 |
| `params/st3215.json` | actuator parameters. **Currently the vendor priors, not a fit** — `actuator.load()` says so out loud. Overwritten by `robot/bench/fit_bam.py` |
| `params/bus_timing.json` | written by `robot/bench/bus_probe.py`; the measured command delay step 4 randomises around |
| `params/domain_rand.json` | randomisation ranges, each marked measured or guessed — step 4 |
| `model.py` | training model = generated MJCF + MjSpec edits — step 4 |
| `env/` | observations, actions, rewards, commands, randomisation — step 4 |
| `jaxenv.py` | the env vars this box needs, each set before the library that reads it |
| `train_ppo.py` | MJX + Brax PPO, a checkpoint per eval — step 5 |
| `eval.py` | deterministic rollouts, metrics, sim-to-sim in vanilla MuJoCo — step 5 |
| `replay.py` | a herd of checkpoints replayed kinematically, to mp4 — step 5 |
| `export_onnx.py` | policy + observation normaliser in one graph — step 6 |
| `runs/` | ignored |
