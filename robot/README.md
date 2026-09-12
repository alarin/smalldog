# SmallDog — the hardware side

Everything that talks to the robot. No ROS, no JAX: `pyserial` and `numpy` are the
dependency list, because this tree runs on the Orange Pi 5 Pro inside a 20 ms tick, and
because `robot/bench` had to run before there was a robot.

| path | what it is |
|---|---|
| `feetech/registers.py` | the SMS/STS control table, transcribed once |
| `feetech/bus.py` | half-duplex driver: read/write, SyncWrite, SyncRead, timing |
| `feetech/loopback.py` | a servo made of bytes, so the protocol is tested every run |
| `bench/bus_probe.py` | what a transaction really costs on this machine → `rl/params/bus_timing.json` |
| `bench/sweep.py` | drives one servo through the identification trajectories → `bench/data/` |
| `bench/fit_bam.py` | those csv files → `rl/params/st3215.json` |
| `bench/hysteresis.py` | the two differential ladders (`holdbi`, `speed`) that measure friction without the fit |
| `bench/torque_hold.py`, `bench/torque_limit.py` | one steady push against a scale on `3d/torque_rig.py`; the volatile `TORQUE_LIMIT` cap |
| `bench/noload_speed.py` | the no-load speed and the duty plateau |
| `bench/lift_test.py`, `bench/*.json` | is foot contact visible in the servo load? the recordings |
| `bench/runlog.py` | the csv format, defined once and used from both ends |
| `runtime/calib.py`, `calib.json` | which servo is which joint, its zero, its sign — a measurement of one robot, in git |
| `runtime/safety.py` | the limits, and the one place that decides to cut torque |
| `runtime/loop.py` | the 50 Hz tick, controller-agnostic |
| `runtime/walk.py` | the CLI that runs the trot on the robot |

Nothing here needs hardware to be exercised:

```bash
python -m feetech.bus --selftest      # packets, checksums, sign-magnitude, SyncWrite
python bench/bus_probe.py --dry-run   # the timing harness
python bench/sweep.py --dry-run --traj all
python bench/fit_bam.py --selftest    # generate a known servo, then find it again
python bench/torque_limit.py --selftest
python runtime/calib.py --selftest    # ids, centres, signs, the clamp, the round trip
python runtime/safety.py --selftest   # every limit trips, and only when it should
python runtime/loop.py --selftest     # 2 s of the real loop against a loopback bus
python runtime/walk.py --dry-run --profile   # ... and the whole trot on top of it
```

The measured servo parameters are collected for outsiders in
[`../ST3215_STS3215_measured_parameters.md`](../ST3215_STS3215_measured_parameters.md);
every number there came from one unit on this bench.

## The 50 Hz budget

At 1 Mbit a byte is 10 µs: SyncWrite of 12 goals 440 µs, SyncRead request 200, twelve
15-byte replies 2520 — **3.2 ms in a 20 ms tick**. One contiguous 15-byte read from
`PRESENT_POSITION` covers position, speed, load, voltage, temperature and current, which
is why `FEEDBACK_LEN` is 15. `SYNC_READ` is not in every firmware; the driver falls back to
sequential reads and reports which path it took.

The wire is never the constraint; the host is: USB frame scheduling, the adapter's latency
timer (FTDI ships at **16 ms** — `/sys/bus/usb-serial/devices/ttyUSB0/latency_timer`, set
it to 1), and the servo's Return Delay. `bus_probe.py` measures the real number and it is an
input to the training randomisation. **Measured, twelve servos, mac + CDC adapter:** full
tick p50 ~3.4 ms, p99 ~4.0 ms. Do not read the bus *scan*'s mean as this number; a scan
spends its time timing out on ids that do not exist.

Two things the bus does, understood and not:

- **Slot 9 of 12 gets a corrupt checksum on a quarter of SyncReads** (`0x22` arrives as
  `0xe2`, payload intact): a talker releasing the half-duplex line a bit-time early, since
  bits 6–7 are the last on the wire. Rotating the id list moves the fault with the slot.
  `Bus.sync_read` drops that frame and re-reads **only that id** (0.47 ms) — a re-read,
  never a repair: masking the bits off would be believing a frame because it is corrupt.
  `--profile` prints `N checksum (M re-read)`; M rising with N is this defect behaving.
- **A rare ~24 ms transport stall**, 0–3 missed deadlines per 500 ticks, almost certainly
  host-side on the mac. The number to trust is the one measured **on the Orange Pi**.

## The runtime, in order

`loop.Runtime` takes `source(dt, feedback) -> 12 joint angles` in `robot_params.json`'s
joint order and does the bus, the timing and the safety layer around it. Today the source
is `smalldog_walker`'s analytic trot, imported from `ros2/` (pure Python); tomorrow the ONNX
policy out of `rl/`. Neither gets its own idea of a soft limit.

### Bring-up

1. **Program the ids.** `python runtime/calib.py --ids` prints the map: **`<leg><joint>`**,
   leg counted round from the front right, joint counted down the leg — 11-13, 21-23,
   31-33, 41-43. Not `robot_params.json`'s `fl, fr, rl, rr` order; `calib.py`'s `LEG_DIGIT`
   is the one place the two are tied. No id lands in 1..12, so a factory-default servo
   answers as nobody; the ids run to 43, so a scan has to go past 43 (`feetech/bus.py
   --scan` does). Set them over the URT-1 **before assembly**.
2. **`python runtime/walk.py --preflight`** — pings all twelve, reads the control
   registers, checks the host including `latency_timer`. Nothing moves.
3. **Centre the servos, then `python runtime/calib.py --capture`.** Torque off, robot at
   mechanical zero (legs straight down). **Capturing alone is not enough**: as built, six
   joints sat within 330 counts of the 0/4095 encoder wrap and could not reach their soft
   limits (`Servo.to_counts` clamps silently, and wrapping in software sends the leg the
   long way round). Use the servo's own middle-position calibration — write **128 to
   `TORQUE_ENABLE`** with the joint held at zero; it computes `OFFSET` itself. Park
   `GOAL_POSITION` on the present position first, verify by reading back (the write gets
   no reply), then `--capture` records the residual. The soft clamp is ±1.45 rad of roll
   now (it was 0.78), so a joint near the wrap is twice as easy to hit — re-run `--capture`
   before trusting an old `calib.json`.
4. **`python runtime/calib.py --sign all`** moves each joint 8.6° and asks a human which way
   it went — the bus cannot tell which way the fork went on. Prompts are in the robot's
   frame (+X forward, +Y left, +Z up; positive roll swings the foot left, positive pitch
   backward, positive knee folds the shin back), **and bring-up happens belly-up**, where
   left and right reverse. Do not settle roll by eye: command one axis on all four legs at
   once and compare the legs against each other, marking a side first. This robot came out
   **roll +1,+1,−1,−1 (front/rear), pitch and knee +1,−1,+1,−1 (left/right)**.
5. **`--stand` before `--profile` before the keyboard.** Standing is the first real
   question: whether twelve ST3215 hold 2.5 kg at the commanded height without cooking.

### Foot contact from the servo load

`smalldog_walker/contact.py` gets contact from the knee servo's load minus what the same
leg reads at the same phase hanging free; the loop already SyncReads load every tick, so
it is nearly free. `walk.py --baseline` records the free-air curve, `--contact` uses it,
`--contact-threshold` is **50**.

- **The signal is real and it is latched, not noisy.** Present Load holds dead flat while
  the robot hangs and steps only when what is under the foot changes. Two independent 30 s
  free-air recordings correlate at r = 1.000 with a ~2-unit noise floor on a ~700-unit
  range. Loaded against free, per leg per phase bin, at 2.55 kg: stance medians 65–104
  units, swing medians ~0–10 — 32 to 53 σ of separation.
- **At 1.55 kg (no battery, Pi or payload) it is not there on every leg**: two knees moved
  nothing when set down. That is stiction — below the friction the motor does nothing
  whether the foot is loaded or not, so the duty carries no information. Measure at design
  mass; the plate's position matters (front duty 0.68 against the rear's 0.40 with it
  forward).
- **It had to be measured at a slow gait** (0.02 m/s, period 1.35 s): at 0.20 m/s the trot
  demanded 7.55 rad/s against the limiter, the commanded path was clipped, and the robot
  dragged with the residual peaking in the phase called *swing*. Do not carry 50 back to a
  gait with no clean swing phase to subtract against.
- **The sign is per leg** (knee hubs are mirrored left/right): `ServoContact` takes a
  `{leg: ±1}` dict, and `--contact-sign auto` fills it from `calib.json`'s knee signs, which
  came out identical to the map derived from the load residuals.
- **Baselines**: `runtime/contact_baseline_slow.json` (the measurement gait) and
  `contact_baseline_walk.json` (recorded at 0.14 m/s / 1.20 s). The rate ceiling has since
  moved the fitted gait to 0.11 / 1.50, so `mismatch()` flags the walk baseline on every run
  — re-record it hanging at the new point, or run `--as-commanded --period 1.2 --speed
  0.14`. A period under ~1.2 s at 50 Hz skips phase bins outright (`Baseline.coverage()`
  says which cure: longer period or fewer bins).
- **The threshold of 50 is provisional**: closing it needs a *ground* run at the walking
  gait, which the 40 cm bench cannot give (2.5 gait cycles against 60 bins).
- **Hang it the right way up.** The curve is gravity plus inertia plus friction, and only
  gravity cares which way up the robot is.

What the contact says about this robot: the front feet carry for a duty of 0.68 against the
commanded 0.50 and the rear 0.40–0.43 — the plate sitting forward, so the commanded phase
is not ground truth for contact. The feet themselves are the wrong shape for the floor (a
true R = 13 hemisphere, 2.5 mm Hertz patch at 1.3 MPa, against a sim that assumes μ = 1.2);
PLAN.md step 5 is the measurement that decides whether that matters.

### The guard

`safety.Limits`: 65 °C, 2.0 A held 0.3 s, 9.5 V, 0.70 rad of tracking error.

- **Every current limit is in supply amps, and the motor's is higher.** `PRESENT_CURRENT`
  reports the supply current behind the bridge, `d²·U/R`, while the motor carries `d·U/R`;
  2.0 A of supply is d ≈ 0.85 and 2.35 A through the motor. At low duty the register is
  quadratically insensitive — a joint pulling 1 A at 36 % duty reports 0.36 A. These trips
  catch a stalled servo; they are not a thermal limit on a joint working hard at partial
  duty. Standing at 1.55 kg the robot draws 0.4 A while the twelve registers sum to 0.07:
  ~27 mA of quiescent per servo the register does not see.
- **`PRESENT_TEMPERATURE` is unusable while the motor drives**: PWM noise reaches the ADC
  and a 30 °C servo returns isolated bytes up to 150. Every spike is a single sample, so it
  is median-filtered over nine and held 0.5 s; `--profile` counts the discards.
  `Guard.check_at_rest` still refuses on one hot reading before torque comes on.
- **Tracking is 0.70 rad, from the loaded run, not the estimate.** The trot at 2.55 kg peaks
  at 0.61 rad / 35°, reproducible, on a robot doing what it was told (0.35 tripped it after
  one second). `--track-rad` exists because a limit nobody can set is a limit nobody can
  measure. Position itself is trustworthy while driving (0 impossible jumps in 321
  samples). Both selftest fixtures derive from `Limits.q_err_rad`, so raising it cannot
  green a guard that no longer guards.

**What the robot weighs today is not what the limits are for**: 1.55 kg on the table
against 2.49 kg designed, and all of the missing 940 g is body mass. Nothing measured under
load at 1.55 kg transfers — not current, sag, tracking error or the contact threshold — which
is why the bench runs at 2.55 kg with a 1 kg plate.

### The gait is fitted to the servo

`walk.py` refuses to command a trot these servos cannot fly. At 0.20 m/s the trot demanded
7.55 rad/s; the commanded foot path was clipped before a servo saw it and the robot dragged
at 0.067 m/s. `feasible_gait()` finds the shortest period whose demand fits under the rate
ceiling (`joint_rate_ceiling_rad_s` = 3.15 from `robot_params.json`, × 0.95) with the
stride under `max_step`, and caps the speed if none does; `--as-commanded` restores the
old behaviour and names the defect. `walk.py --dry-run --profile`:

| commanded | best period | stride | demand | |
|---|---|---|---|---|
| 0.08 | 1.20 | 24 mm | 2.91 | comfortable |
| 0.10 | 1.30 | 32 mm | 2.99 | |
| **0.11** | **1.50** | **41 mm** | **2.98** | **what anything faster is capped to** |
| 0.12 and up | — | — | — | infeasible at any period |

Ground runs at the previous fit (0.14 / 1.20 s) measured **350–450 mm in 3 s = 0.12–0.15 m/s
against 0.067** — the cap is a faster robot than the drag was. Nothing has measured
0.11 / 1.50 on the ground yet. `feasible_turn()` fits the turn the same way: the teleop's
1.2 rad/s is capped to 0.65; `clamp_profile()` caps the scripted demo's own velocities.

The sim does not have this bug — MuJoCo's feet grip at μ ≈ 1.2 — so `gait.py` is untouched;
fixing it there would move every tuned number in `ros2/` to cure something only the
hardware suffers. The trot stays inside the soft limits on its own (knee 95.3° against
99.1° at 0.45 m/s); the runtime's clamp stays for a source that is *not* the trot.

## The bench, in order

**Build it first.** `3d/bench_rig.py` imports the same `mini_dog.sleeve()` and hub pattern
the robot uses, so the bench holds the servo the way the robot does:

```bash
cd 3d && .venv/bin/python bench_rig.py       # -> out/bench/{step,stl} + the numbers
.venv/bin/python tools/slice_orca.py --machine "Qidi Q2 0.4 nozzle - Copy" \
    --process "0.20mm Standard @Qidi Q2 - Copy" --filament "QIDI НИТ petg черный" \
    --name bench_rig --walls 4 --infill 30% \
    servo_gauge bench_stand bench_arm_s bench_arm_l
```

| part | g | why |
|---|---|---|
| `servo_gauge` | 11.8 | **print this alone first and stop.** Half a sleeve with a hub arm: check a real ST3215 drops into the bore and the four M3 land on the hub before spending five hours on a stand built from the same numbers |
| `bench_stand` | 154 | sleeve + thrust clamp + column; axis **190 mm** above the base so the ⌀140 disc clears |
| `bench_arm_s` | 5.07 | light arm, reach 45 mm — the direct `J_m` measurement |
| `bench_arm_l` | 8.17 | heavy arm, reach 90 mm — the holds and the second free swing |
| `bench_bushing` × 2 | 4.27 ea. | centre the ⌀29 bore on the M6 tip bolt; part of the tip mass |

Arm and bushing masses are weighed off the real print and are in `bench_rig.py`
(`BENCH_FILL`); the slicer was optimistic by 13 % on the arms and 46 % on the bushing.

Non-printed, and the bench does not work without them:

- a supply whose voltage you can **set** (12.6 / 11.1 / 9.9 V);
- **two weighed tip masses**, ~250 g short arm (≥ 227 g or the release is unreadable
  against `tau_c`), 350–450 g long arm (≤ 530 g or `J_load` passes the 35 % share the fit
  warns at). **Compact, one rigid body, outboard of the arm's face, gained along the bolt
  and never in the swing plane**: `J_load` is `m·r²` plus the mass's own moment, and a flat
  disc in the swing plane is over the line on shape alone; a rattling stack brings its own
  friction and damping, which the free swing's decay envelope will charge to the servo.
  Nine 40 × 40 × 3 mm steel squares (339 g) or 79 mm of 50 × 50 × 3 box section (350 g)
  both work; a 500 g calibration disc does not;
- **a clamp — mandatory.** The heavy arm out horizontal is 0.49× the stand's own
  overturning margin and `bench_rig.py` prints `!! IT WILL GO OVER unclamped`;
- 4 × M3 × 6 into the driven hub, 2 × M3 × 10 set screws + nuts for the thrust clamp.

Bolt the arm on so it **hangs straight down** at the servo's centre position: `fit_bam.py`
regresses against `m·g·r·sin(q)`, so q = 0 has to be the hanging position. `bench_rig.py`
prints the exact `sweep.py` line for each arm; weigh the printed arm and correct
`--arm-inertia` if it disagrees.

1. **`--check`** — preflight; prints every control register, nothing moves.
2. **`--traj rock`** — torque off, rock the horn against the play, watch Present Position.
   **Answered: the encoder is AFTER the gearbox** — it moves. The robot can observe its own
   play. Re-run after any gearbox or servo swap.
3. **`--traj freeswing`, light short arm** — the one direct measurement of the reflected
   rotor inertia (73× the knee link's own). `J_m` = `J_total − J_arm`, so the smallest mass
   at the shortest radius that still overcomes friction, **and two different arms**, or a
   heavier J with less friction fits the same fall.
4. **`--traj hold`, heavy long arm, at three voltages** — two linear regressions at
   steady state give the torque constant and the loop gain.
5. **`--traj step` / `triangle` / `reversal` / `chirp`**, same arm, same three voltages.
6. **`--traj holdbi` and `--traj speed`, heavy long arm, three voltages** — the two
   differential ladders `hysteresis.py` reads by subtracting one run from another ("Two
   things the fit cannot find").

The three voltages are not thoroughness: at one voltage back-EMF damping and viscous
friction enter every equation as the same coefficient of ω. **Every run records the servo's
control registers** and the fit refuses to merge runs whose registers disagree — a fit is
only valid for the P/D/I, dead zone, punch and acceleration the servo had at the time.

Power the servo from the supply directly, never through the URT-1 or USB (a stalled ST3215
draws 2.7 A). `sweep.py` aborts on temperature, current and travel and always leaves torque
disabled; the servo's own protection registers are read and never written.

## How the fit works, and what it can and cannot find

`fit_bam.py` runs analytic passes first and a trajectory refinement last, because the
analytic passes are where the identification happens:

| pass | gives | from |
|---|---|---|
| **hold ladder, both ways** | **`tau_c`, `mu_load`** | the up/down difference at each angle — a measurement, not a fit |
| free swing | `J_m`, `tau_c` | release acceleration, or the period if it oscillates |
| holds | `k_u·R`, `kp/R`, dead-zone intercept | two linear regressions at ω = 0 |
| saturation | `1/R`, `k_e/R` | one regression over the duty-pinned samples |
| reversal | dead zone | the smallest command that moves the joint |
| refinement | everything, a little | Powell over a diverse subset — **opt-in, `--refine`** |

- **The fit does not run in the model's own coordinates**: the instruments only ever see
  `k_u·R`, `kp/R`, `k_e/R` and `1/R`, so fitting R and k_u separately lets the optimiser
  walk along `k_u·R = const`. On synthetic data R recovered to 0.1 % while k_u sat 34 % off.
- **The refinement is Powell, not `least_squares`, and off by default**: the residual is
  not smooth (dead zone, punch, backlash, saturation are switches), finite differences read
  an optimality of ~10⁹ and `trf` converges after a step of zero. Powell still came back
  3.5× worse than the analytic seeds on synthetic data, so the guard discards it; `--refine`
  can decline but not damage. With the full data set it diverges (`kp` at its bound, `k_e`
  11.3, an implied no-load speed of 1.06 rad/s against a measured 3.86) — a fit reaching for
  torque the electrical model cannot supply, and adding friction did not change its
  character.
- **`--selftest`** generates data from a *known* servo and finds it again: R 3.7 %, k_e 2.2 %,
  k_u 6.5 %, kp 14.8 %, J_m 5.2 %, tau_c 24.5 %. `b_v`, `deadband`, `punch`, `theta_bl` are
  not recovered because the 8 s self-test truncation stops before the trajectories reach the
  commands they need; run them in full on hardware. `mu_load` is the opposite case: measured
  cleanly on hardware and *not* recoverable from the model's own output, because the smooth
  law has no stick — the self-test is indicting the model, not the experiment.
- **A run no pass consumes changes no parameter.** The analytic passes select on
  `freeswing`, `hold` and `holdbi`; everything else reaches the fit only under `--refine`.
  `check_identifiable` names every trajectory in the directory no pass reads. Do not read an
  unchanged fit as agreement.
- **`--seconds` truncates, and a run whose measurement is a difference between its halves
  is halved, not shortened.** `holdbi` is 36 s against a 25 s default and once silently lost
  two thirds of its descending rungs. `Run.WHOLE_RUN` names those.

`rl/params/st3215.json` is the current fit: 44 runs at three voltages. Read its `source`
string before quoting it.

### The load-dependent friction term

`actuator.Params.mu_load` — friction per N·m crossing the gearbox — measured by
`fit_bam.seed_from_holdbi` off the bidirectional ladder, running **first** among the
analytic passes because the free swing subtracts `tau_c` and had been using half the real
value:

| | before | after |
|---|---|---|
| `tau_c` | 0.084 | **0.184** |
| `J_m` | 0.0240 | **0.0165** |
| `mu_load` | — | **0.286** |
| position RMS | 8.96° | 8.84° |
| current RMS | 1.25 A | 1.01 A |

The ladder and the free-swing fit are independent routes that had disagreed by 2× and now
land on 0.184 and 0.186. `J_m` moving is a consequence.

### Static friction

`simulate()` uses Karnopp: below `v_eps`, friction opposes the net applied torque up to
`tau_c + mu_load·|tau_t|` and the shaft is held; a hold approached from two sides settles at
two duties. The magnitude is uncalibrated (0.72 V half-difference on the model against the
real 0.29). On the training path `friction()`/`motor_torque()` cannot do this — MuJoCo owns
the load there — so stick comes from MuJoCo's own `frictionloss` = `tau_c`
(`rl/CLAUDE.md`). A refit against the Karnopp model moved `J_m` +20 % and nothing else,
with one RMS better and one worse; it was **not** adopted.

### The stall torque, in newton-metres

**4.50 N·m at 12 V**, on `3d/torque_rig.py` — a printed C-frame with a 170 mm arm pressing an
M6 anvil onto a 2 kg coffee scale. `mini_dog.py`'s `SERVO_STALL_NM` is this number.

| `TORQUE_LIMIT` | d·U | scale, cold | τ |
|---|---|---|---|
| 200 | 2.41 V | 0.408 kg | 0.680 N·m |
| 350 | 4.13 V | 0.780 kg | 1.300 N·m |
| 450 | 5.27 V | 1.100 kg | 1.834 N·m |

k_u = **0.400 N·m/V**, friction intercept **−0.30 N·m** (a third independent read on
`tau_c`), residuals ±0.05. The register routes (7.4 N·m, two channels agreeing to 0.3 %)
and the datasheet (2.94) were both wrong, in opposite directions; there was never one scale
error to find. `k_t` and `R` stay unresolved because `PRESENT_CURRENT` is unusable below
~0.2 A.

- **Do not read a scale off `sweep.py --traj stall`.** Its 0.8 s bursts are right for the
  electrical fit and wrong for a scale, which reads the arm's inertia and its own filter
  ringing on top (920 g in bursts against 780 held — a constant ~130 g offset that corrupts
  the slope). `bench/torque_hold.py` is one steady 8 s push with a per-second profile.
- **The torque decays 4–7 % in 8 s** as the winding warms and R rises; the cold value is the
  one a leg uses in a transient and the one recorded. `rl/actuator.py` has no thermal term.
- **Fixture faults produce convincing wrong answers.** A kitchen scale bridging the 33 mm
  jaw rocks; on the table the reaction goes out through a bench clamp, and one clamp let the
  stand lean its own weight onto the scale (+70 g every rung). The encoder cannot see any
  of this. Next time: scale in the throat on a rigid plate wide enough for its feet.
- **`TORQUE_LIMIT` is volatile** (SRAM, reloaded from `MAX_TORQUE` on power-up).
  `bench/torque_limit.py` writes it, reads it back, and refuses while torque is enabled.
  Cap before every session and after every supply change.
- **Still open**: all three rungs are 20–45 % duty, so 4.50 is a 2.2× extrapolation, and the
  no-load plateau below may say the loop never drives past ~75 % (PLAN.md 3c). One voltage
  only so far.

### The no-load speed, and the plateau

**3.86 rad/s at 12.1 V** with `bench/noload_speed.py`, hub free, no arm; the vendor's 4.71
was 22 % high. `SERVO_NOLOAD_RADS` is 3.86.

| `TORQUE_LIMIT` | d·U | ω from position | ω from `PRESENT_SPEED` |
|---|---|---|---|
| 1000 | 12.0 V | **3.864 rad/s** | 3.835 |
| 800 | 9.6 | 3.851 | 3.835 |
| 600 | 7.2 | 3.093 | 2.953 |
| 400 | 4.8 | 2.077 | 1.994 |

The register's speed LSB is verified as a side effect (1–5 %). **The finding is the
plateau**: 800 and 1000 read the same, the register sits at 2500 counts/s for both, and the
rungs below are linear through the origin at 0.43 rad/s/V (k_e = 2.32 V·s/rad). **It is the
position loop's profile, not the bridge**: `noload_speed.py --pwm` (MODE 2, open-loop duty)
on the second unit reads 2.02 / 3.06 / 4.10 / 5.02 rad/s at duty 400 / 600 / 800 / 1000 —
linear through the origin, k_e = 2.39, no plateau — while position mode on the same unit
stops at 3.88 with the same flat 2500. So the 4.50 stall stands as far as this test can
say; it shows the motor reaches full duty in MODE 2, not that the position loop applies it
at zero speed — the torque rig at `TORQUE_LIMIT` 800 and 1000 is that check. `rl/actuator.py`
still frees to 5.9; what it needs is a rate cap on the goal. Both tools leave `TORQUE_LIMIT`
at their last rung; re-cap before the arm goes back on.

### Two things the fit cannot find, and the ladders that measure them

1.066 kg on the 90 mm arm at 8 / 10 / 12 V, `sweep.py --traj holdbi` and `--traj speed`,
read by `hysteresis.py`. The law is `tau_c·sign(w) + b_v·w`, both independent of load, so a
friction that grows with the torque carried has nowhere to go except `k_u`, and `b_v` sits
pinned on its bound however much data it is given. Both ladders are **differential**: a
difference between two runs of the same trajectory.

| | 8 V | 10 V | 12 V | must it move with V? |
|---|---|---|---|---|
| position-loop stiffness, N·m/rad | 28.1 | 34.4 | 40.9 | **yes**, 3.44 per volt |
| effective torque constant, N·m/V | 0.636 | 0.611 | 0.596 | no |
| friction at a hold, unloaded, N·m | 0.192 | 0.181 | 0.186 | no |
| …per N·m of load carried | 0.286 | 0.295 | 0.251 | no |
| kinetic Coulomb at zero load, N·m | 0.161 | 0.160 | 0.168 | no |
| total speed-proportional, N·m·s/rad | 1.403 | 1.364 | 1.334 | no |

The four rows holding still across a 1.5× supply range is the check that they are physical.

1. **Friction grows with load: +0.28 N·m per N·m carried, on a 0.19 N·m floor.**
2. **The stiffness was 40 % low and the error hid inside the friction**: a one-way ladder
   bills the friction band to elasticity. 40.9 N·m/rad at 12 V, not 28.8.
3. **`b_v` is not resolvable on this bench.** The speed ladder pins the *total*
   speed-proportional torque to 1.37 N·m·s/rad within 5 %, but that is `b_v + k_t·k_e`,
   back-EMF alone is 1.31, and neither term depends on supply, so the voltage sweep cannot
   split them. Use the total; it is what a simulator needs (`MJ_DAMPING`).

One loose end: at a commanded 2.0 rad/s the servo reaches 1.8 and no more, at 0.74 rad of
error, without the duty pinning. `D_COEF` is 32 and `rl/actuator.py` models `kd = 0`; an
internal output clamp would look the same, and this data does not separate them.

### Unit to unit

The same two ladders on three more ST3215s, heavy arm, 12 V only, in `bench/data-servo2/`,
`data-servo11/`, `data-servo21/` (`hysteresis.py --data <dir>`). Unit 2 was a loose servo
14 counts from the wrap and was recentred with the middle-position command (`OFFSET`
−1918); 11 and 21 came off the robot and ran with `--centre` at their hanging position so
`calib.json` still holds.

| | unit 1 | unit 2 | #11 | #21 | spread |
|---|---|---|---|---|---|
| position-loop stiffness, N·m/rad | 40.8 | 39.7 | 39.2 | 42.8 | ±4.5 % |
| effective torque constant, N·m/V | 0.596 | 0.578 | 0.574 | 0.620 | ±4 % |
| friction at a hold, unloaded, N·m | 0.186 | 0.117 | 0.144 | 0.181 | **0.64–1.01×** |
| …per N·m of load carried | 0.251 | 0.380 | 0.328 | 0.335 | **0.88–1.33×** |
| kinetic Coulomb at zero load, N·m | 0.168 | 0.124 | 0.153 | 0.192 | 0.67–1.04× |
| total speed-proportional, N·m·s/rad | 1.334 | 1.326 | 1.276 | 1.373 | ±4 % |
| no-load speed, position mode, rad/s | 3.86 | 3.88 | — | — | |

The electrical rows and the profile cap are the same servo to 5 %; the gearbox is not, and
the unit with the lowest floor has the highest load slope — two numbers, not one grease
number. Each unit's static and kinetic floors agree with each other, so the spread is the
gearbox, not the read. `rl/params/domain_rand.json` carries it: `tau_c` ×0.55–1.20,
`mu_load` ×0.75–1.55, `kp` ×0.85–1.15 (the observed spread with a margin for the eight
unmeasured), all `measured` now.
