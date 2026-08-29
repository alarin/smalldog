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
uv sync --extra cuda     # WSL2 + RTX 5070 Ti: training
uv sync --extra fit      # scipy + matplotlib, for the bench fit
```

`jax[cuda12]` must resolve to a jaxlib built against **CUDA 12.8 or newer** —
Blackwell is `sm_120`, and an older wheel installs cleanly and then dies at the
first kernel launch with *no kernel image is available for execution on the
device*. Check with `python -c "import jax; print(jax.devices())"` before
blaming anything else.

On WSL2 the NVIDIA driver belongs on the **Windows** side; installing a Linux
driver inside WSL overwrites the passthrough and `nvidia-smi` stops seeing the
card. Keep the checkout inside the WSL filesystem (`~/smalldog`), not under
`/mnt/c` — file operations there are an order of magnitude slower, which is felt
on mesh loads and checkpoint writes.

## Checks

Both need only `mujoco` and `numpy`, so they run on any of the three machines.

```bash
python checks/check_model.py                  # audit the MJCF, exit != 0 on failures
python checks/check_model.py --terrain        # ... on the heightfield scene
python checks/check_model.py --view           # ... plus the passive viewer
python checks/check_model.py --shot out.png   # ... plus one offscreen frame
python checks/imu_placement.py                # does the IMU mounting point matter?
python checks/wheels.py                       # wheels, and what ratio they need
python checks/wheels.py --gear 6 --diameters 0.06,0.08,0.10
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
| `checks/wheels.py` | wheels: free, ratchet, and driven over gear ratio and diameter |
| `actuator.py` | ST3215 voltage law, back-EMF, friction, backlash — step 3 |
| `params/st3215.json` | actuator parameters. **Currently the vendor priors, not a fit** — `actuator.load()` says so out loud. Overwritten by `robot/bench/fit_bam.py` |
| `params/bus_timing.json` | written by `robot/bench/bus_probe.py`; the measured command delay step 4 randomises around |
| `params/domain_rand.json` | randomisation ranges, each marked measured or guessed — step 4 |
| `model.py` | training model = generated MJCF + MjSpec edits — step 4 |
| `env/` | observations, actions, rewards, commands, randomisation — step 4 |
| `train_ppo.py` | MJX + Brax PPO — step 5 |
| `eval.py` | deterministic rollouts, metrics, sim-to-sim in vanilla MuJoCo — step 5 |
| `export_onnx.py` | policy + observation normaliser in one graph — step 6 |
| `runs/` | ignored |
