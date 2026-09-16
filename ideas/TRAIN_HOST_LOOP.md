# Train against the host loop — the spec for step 4 of `FAST_SERVOS.md`

For the session that runs it: a WSL2 box (`WSL.md` / `GPU.md`), `rl/` conventions in
`rl/CLAUDE.md`. Read this whole file before touching `actuator.py`.

## Goal

A policy that walks on the robot under `robot/runtime/mode2.py` better than s4 does today:
the same ~0.2 m/s at cmd 0.2, a start-up under 0.3 s instead of 0.6, pitch held within
±5° under way, and a stand at cmd 0 that survives a bad hand-over (a −16° pitch at
torque-on tipped s4 forward). Speed is not the goal — s4 already walks at the sim's speed.

## What is true on the robot now (2026-09-16)

The servos run in MODE 2 (open-loop PWM) and the Pi closes the position loop
(`robot/runtime/mode2.py`, `Mode2Runtime`). What a joint sees, in order:

1. The policy's target arrives every 20 ms (50 Hz), the same as before.
2. The host loop runs one **sub-tick** per bus round trip: read all twelve (SyncRead),
   compute, write all twelve duties (SyncWrite). **165 Hz measured on the Pi** with
   twelve servos — 6.1 ms per sub-tick, set by the CH340 adapter's 2 ms per transaction.
   A new target is picked up by the next sub-tick: 0–6 ms of latency, plus whatever the
   50 Hz tick's own scheduling adds (the existing `bus.delay_s_abs` draw covers that).
3. The law, per joint, in the joint frame, then the servo's sign:

       duty = (11.3 / U_pack) · (kp · (target − q) − kd · ω + kff · ω_target)

   **For a policy the gains are the model's: kp 5220, kd 0, kff 0** — `policy.py`'s
   defaults. 5220 duty per rad is `actuator.Params.kp` = 5.22 in duty-fraction units.
   The `11.3 / U_pack` factor keeps the stiffness in volts constant as the pack drains
   (`mode2.VOLT_REF`); the sim's `U = duty · U_bat` then makes it voltage-independent
   until the duty clips. `q` is the encoder count (0.0015 rad steps), `ω` the servo's
   own speed register. **No dead zone, no punch, no goal profile** — those were the
   firmware's; they are gone. (The trot uses 9000 / 150 / 200; a policy must not: its
   targets step, and kff turned a 0.1 rad step into a 1000-duty kick and a dive.)
4. The duty is clipped to ±1000, then **folded on the current register**: full duty at
   or below 1.4 A, linearly down to 0.25 of it at 2.0 A (per servo, on that servo's
   reading), and the same fold on the **sum over twelve** between 7 and 10 A. The
   register is the supply current, `|duty| · |i_motor|` (`registers.py`,
   `CURRENT_LSB_A`), read one sub-tick late. The fold exists because the firmware still
   cuts torque at 2 A for 2 s in MODE 2 and a cut servo is dead until torque is cycled.
5. The duty holds until the next sub-tick.

Everything below the duty — the bridge, the pack sag, the motor, the gearbox friction,
the backlash — is unchanged and already in `actuator.py`.

Measured with the runtime, for the model to reproduce:

| | firmware loop (model today) | host loop, trot gains 9000/150/200 |
|---|---|---|
| ±15° sine, gain at 1 / 2 / 3 / 5 Hz | 0.74 / 0.25 / – / 0.07 | 0.99 / 0.98 / 0.96 / 0.63 (mac, 1.9 kHz) |
| 0.1 rad step, peak speed | 0.8 rad/s | 2.9 rad/s (≈ 80 rad/s²) |
| standing error | — | 0.05° free, 0.5–1.3° on the robot |

For the **model gains 5220/0/0 there is no bench table yet** — the sine test
(`python robot/bench/pwm_loop.py --id 1 --kp 5220 --kd 0 --kff 0 --runs BC`) on a free
servo takes two minutes on the Pi and is the number `chirp_gain.py` should be checked
against. Ask for it before the run starts; if it cannot be had, the 9000/150/200 table
above is the check and the gains are a parameter.

## What changes in `rl/`

### `actuator.py` — a host-loop mode on `Params`, the one copy of the law

- `host_loop: bool = False`, `host_hz: float = 165.0`, `host_volt_ref: float = 11.3`,
  `fold_i_soft: 1.4`, `fold_i_hard: 2.0`, `fold_floor: 0.25`, `fold_i_sum: 10.0`
  (register amps; the sum fold's soft point is `fold_i_sum · soft / hard`).
- `duty()` in host mode: no dead zone, no punch; `u = kp·e − kd·w` on the quantised
  error, times `host_volt_ref / u_bat`, clipped to ±`duty_max`.
- `profile_goal()` in host mode is the identity: `goal = target`. Do not set
  `goal_acc` to 1e6 as a trick — the flag is explicit so `check_model.py` can see it.
- The fold: a function `fold(p, i_reg_per_joint)` returning the per-joint duty multiplier
  from steps 4 above. `bus_torque()` applies it to the duty **from the previous
  sub-tick's current** — carry `i_reg` in the scan state with `goal`; that is the one
  sub-tick of lag the real fold has.
- Register in the jax dataclass's `data_fields` / `meta_fields` as the existing ones are.
  `_selftest()` gets: host mode with the fitted params and kp 5.22 reproduces a 0.1 rad
  step at ≥ 2.5 rad/s peak, and ±15° at 2 Hz at ≥ 0.9 gain (with 9.0 / 0.15 / 0.2 if the
  5220/0/0 bench number is not in yet).

### `env/walk.py` — the sub-tick

The physics steps at 1 ms and the actuator torque is recomputed every step. In host
mode the **duty is recomputed every `1000 / host_hz` ≈ 6 physics steps and held**
between; the target the loop sees is the delayed action, as now. The motor torque from
that held duty is still evaluated every physics step (back-EMF moves with ω). Carry a
sub-tick counter in `info` next to `goal`. The bus delay draw (`params/domain_rand.json`
`bus.delay_s_abs`) becomes **0.002–0.008 s** in host mode: one sub-tick of pickup latency
on top of the tick's own; keep the fractional-tick arithmetic in `model.delay_ticks`.

### `params/`

- `st3215.json` stays the fit; `host_loop` defaults false there. Training passes
  `--host-loop` (new flag on `train_ppo.py`, like `--goal-acc`) which flips the flag
  and the delay range; `run.json` records it.
- `domain_rand.json`: `kp` keeps its ±15 % spread (it is now the Pi's number, exact —
  but the encoder, the volt scaling and the bus make it not exact at the joint; keep the
  spread). Add nothing else. Do not narrow anything.

### `checks/check_model.py`

Runs with `--host-loop` and asserts: the profile is off (a 0.5 rad step's peak speed is
not `sqrt(8·0.5)`), the fold multiplier moves across the current range, the delay draw
is non-zero on some fraction of draws. Exit 0 both ways before training.

### `eval.py`

`--host-loop` too; it is the honest number. And the **acceptance test for the model
change itself, before any training**:

- s4 (`--init-from` its run, or the ONNX through `replay.py`) in the **host-loop model**
  must **walk at cmd 0.2 at 0.15–0.25 m/s** and stand at cmd 0 — that is what it does
  on the robot (`robot/bench/data/mode2_s4_vx02_{a,b}.npz`: 20 cm in 1.5 s, the gait
  starting 0.6 s after the command).
- s4 in the **profiled model** (today's) must still rock in place (`rl/CLAUDE.md`,
  re-baselines: 1.35 m → 0.22 m at cmd 0.4).

If the first fails, the model is wrong, not the policy; do not train until it passes.

## The run

- Box: either. Flat, no boxes, `num_envs` 2048 on the 5070 Ti (1024 on the 3070).
- `--init-from` s4's run directory in `rl/runs/` on the training box (gitignored; `ls
  rl/runs | grep s4`). The ONNX in git (`git show ca93dca:rl/policy/`) is the policy
  only — `--init-from` needs (normaliser, policy, value). If the run directory is gone,
  train from scratch with the same recipe and expect ~2× the steps.
- Reward: the 2026-09-15 one (stride-averaged tracking, `tracking_sigma` 0.05), unchanged.
  Do not re-tune the reward in this run; it is the actuator that changed.
- `--vx 0 0.4`, 30 M steps, 12 evals, checkpoints on. Seed 0 and one more.
- Smoke on the mac first (`--smoke`), `check_model.py --host-loop` exit 0.

## Acceptance

Sim (`eval.py --host-loop`, deterministic): ≥ s4's distance at cmd 0.2 and 0.4; body
pitch std under way ≤ s4's; stands at cmd 0 from a 15° pitch perturbation.

Robot (`python3 runtime/policy.py ../rl/policy --port /dev/ttyACM0 --vx 0.2 --seconds
1.5 --ramp 4 --stand-before 2 --log ...`, MODE 2 and the model gains are the defaults):
the bench's 40 cm twice without a hand; then on the floor,
`ros2/tools/straight_test.py` with the L2: ≥ 0.8 m in 5 s at cmd 0.2, heading within
5°. Export with `export_onnx.py` into `rl/policy/` and commit the ONNX with the numbers
in the message, the way `b276193` did.

## What we do not do

- Change `gait.py` or the reward. One variable at a time; the actuator is the variable.
- Train with kff, or with the trot's gains. The policy's loop is 5220 / 0 / 0.
- Model the loop at 250 Hz or 1 kHz because the mac did 1.9 kHz. The robot is 165 Hz until
  step 5 of `FAST_SERVOS.md` exists, and then this number changes in one place.
- Fit the fold's thresholds. They are the runtime's constants; copy them.

## Status (2026-09-16)

The model change is in: `actuator.Params.host_loop`, `actuator.servo_step` (one per-physics-step
function both `env/walk.py` and `eval.py` call), `--host-loop` on `train_ppo.py`, `eval.py`,
`checks/check_model.py`; `model.HOST_DELAY_S` is the 2–8 ms band.

- `_selftest`: on a free servo at 9000/150 the model passes 0.97 / 0.96 / 0.92 / 0.51 of the
  ±15° sine at 1 / 2 / 3 / 5 Hz (bench 0.99 / 0.98 / 0.96 / 0.63). The bench's 2.9 rad/s
  step is **kff's**: the feed-forward turned the 0.1 rad step into a saturated duty for the
  tick, and the model at a saturated duty gives 3.2 rad/s; the P loop alone gives 1.8 (kp 9000)
  and 1.7 (kp 5220). So the ≥ 2.5 check runs with the duty saturated.
- Acceptance (mac, 8 rollouts × 4 s): s4 in the host-loop model walks at cmd 0.2 — 0.475 m in
  4 s in the MJX battery, 603 mm in 4 s in vanilla MuJoCo (≈ 0.15 m/s; the robot does about
  0.2 under way) — and stands at cmd 0. In the profiled model s4 still rocks: 12 mm at 0.2,
  187 mm at 0.4. The model is a little slow against the robot, not on the wrong side of it.
- The fold is written as the runtime has it: the per-joint fold is a **clip** on the duty
  (`duty_max · f_j`), the bus fold a multiplier after it. The register current the fold reads
  is the held duty's supply current at the sub-tick, `|duty| · |i_motor|`.
- 165 Hz is not an integer number of 1 ms steps: a phase accumulator recomputes every 6th or
  7th step, and the phase is drawn per episode so the 50 Hz tick and the loop are not locked.

## Questions for review

