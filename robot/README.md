# SmallDog — the hardware side

Everything that talks to the robot. No ROS, no JAX: `pyserial` and `numpy` are
the dependency list, because this tree has to run on the Orange Pi 5 Pro inside
a 20 ms control tick, and because `robot/bench` has to run before there is a
robot at all.

| path | what it is |
|---|---|
| `feetech/registers.py` | the SMS/STS control table, transcribed once |
| `feetech/bus.py` | half-duplex driver: read/write, SyncWrite, SyncRead, timing |
| `feetech/loopback.py` | a servo made of bytes, so the protocol is tested every run |
| `bench/bus_probe.py` | what a transaction really costs on this machine |
| `bench/sweep.py` | drives one servo through the identification trajectories |
| `bench/fit_bam.py` | those csv files → `rl/params/st3215.json` |
| `bench/runlog.py` | the csv format, defined once and used from both ends |
| `runtime/calib.py` | which servo is which joint, where its zero is, which way it turns |
| `runtime/safety.py` | the limits, and the one place that decides to cut torque |
| `runtime/loop.py` | the 50 Hz tick, controller-agnostic — step 7 |
| `runtime/walk.py` | the CLI that runs the trot on the robot |

Nothing here needs hardware to be exercised:

```bash
python -m feetech.bus --selftest      # packets, checksums, sign-magnitude, SyncWrite
python bench/bus_probe.py --dry-run   # the timing harness
python bench/sweep.py --dry-run --traj all
python bench/fit_bam.py --selftest    # generate a known servo, then find it again
python runtime/calib.py --selftest    # ids, centres, signs, the clamp, the round trip
python runtime/safety.py --selftest   # every limit trips, and only when it should
python runtime/loop.py --selftest     # 2 s of the real loop against a loopback bus
python runtime/walk.py --dry-run --profile   # ... and the whole trot on top of it
```

## The 50 Hz budget

At 1 Mbit a byte is 10 µs, and the whole 12-servo exchange is arithmetic:

| | bytes | µs |
|---|---|---|
| SyncWrite, 12 goal positions | 44 | 440 |
| SyncRead request, 12 servos | 20 | 200 |
| SyncRead replies, 15 B of feedback each | 21 × 12 | 2520 |
| | | **3.2 ms in a 20 ms tick** |

One contiguous 15-byte read from `PRESENT_POSITION` covers position, speed, load,
voltage, temperature and current — six registers, one round trip, two bytes of
padding wasted at 67..68. That is why `FEEDBACK_LEN` is 15 and not smaller.

The wire is never the constraint. The host is: USB frame scheduling, the
adapter's latency timer (FTDI ships at **16 ms**, which alone eats the tick —
`/sys/bus/usb-serial/devices/ttyUSB0/latency_timer`, set it to 1), and the
servo's own Return Delay register. `bus_probe.py` measures the real number, and
that number is an input to the training randomisation, not a diagnostic.

`SYNC_READ` is not in every firmware; the driver falls back to sequential reads
and reports which path it took, so the timing is never quietly measured on a
protocol the runtime will not use.

## The runtime, in order

`runtime/` is the loop that lives in the budget above. It is deliberately ignorant
of what is driving it: `loop.Runtime` takes

```python
source(dt, feedback) -> 12 joint angles, in robot_params.json's joint order
```

and does the bus, the timing and the safety layer around it. Today that source is
`smalldog_walker`'s analytic trot, imported — not copied — from `ros2/`, which is
pure Python and pulls in no part of ROS. Tomorrow it is the ONNX policy out of
`rl/`. Neither gets its own idea of what a soft limit is.

**Nothing here has run against a servo yet.** Every number below is either
arithmetic or a default chosen to be conservative; the `--selftest`s prove the
loop's own bookkeeping, not the robot.

### Bring-up, the order it has to happen in

1. **Program the ids.** `python runtime/calib.py --ids` prints the map the rest of
   the tree assumes — `fl_roll` = 1 through `rr_knee` = 12, in `robot_params.json`
   order. Set them with the Feetech tool over the URT-1 **before assembly**
   (`3d/README.md`, "Assembly order", step 2): once a servo is inside a sleeve
   inside a leg, it is still on the bus, but it is a bad time to discover two of
   them answer to 4.

2. **`python runtime/walk.py --preflight`.** Pings all twelve, reads the control
   registers, and checks the host — including `latency_timer`, which ships at 16 ms
   on FTDI and would eat most of the 20 ms tick on its own. Nothing moves.

3. **`python runtime/calib.py --capture`.** Torque off; hold the robot at the
   model's mechanical zero — legs straight down — and read. This is the only thing
   that ties the servo's counts to the model's radians, and it is a property of
   *this* assembly: the hub bolts on in any of four positions.

4. **`python runtime/calib.py --sign all`.** Moves each joint 8.6° and asks a human
   which way it went, because the bus cannot answer it — the servo reports its own
   counts happily whichever way round the fork went on. A wrong sign on one knee is
   a leg that drives itself into the floor the moment torque arrives, so it is
   asked rather than assumed. The prompts are written against the model's
   convention (+X forward, +Y left, +Z up, identical axes on all four legs): a
   positive **roll** swings the foot to the robot's left, a positive **pitch**
   swings it backward, a positive **knee** folds the shin back and shortens the leg.

5. **`--stand` before `--profile` before the keyboard.** Standing is the first real
   question and it is not "does it walk": it is whether twelve ST3215 hold 2.5 kg at
   the commanded height without cooking. Watch the peaks the run prints.

`calib.json` is a measurement of one physical robot and belongs in git — the Pi and
the mac have no other way to agree about which servo is `fl_knee`.

### What the loop does not do yet

- **No IMU**, so `TrotGait.feedback()` is not called and this is the blind
  open-loop trot. That is a supported mode, not a degradation hack: the gait falls
  back to it on its own if the sensors stop. Measured cost, from `ros2/README.md`'s
  own terrain sweep: 1.8° of heading drift over 1.2 m on flat ground, against 1.65 m
  of sideways travel on relief. Flat floor, blind, is fine; rough ground is not.
- **Foot contact is available without the IMU and is nearly free here.**
  `smalldog_walker/contact.py` gets contact from the knee servo's load minus what
  the same leg reads at the same phase in free air, and its one stated objection was
  cost — four extra round trips per tick. This loop retires that: it already
  SyncReads all fifteen feedback bytes from all twelve servos every tick, load
  included. `walk.py --baseline` records the free-air curve with the robot hanging,
  `--contact` uses it. The threshold is in Feetech's Present Load units and has to
  be re-found on hardware, so both are opt-in.
- **The servo model is still the vendor's.** `rl/params/st3215.json` says
  `fitted: false` out loud. That does not block the analytic trot — the trot
  commands positions and the servo's own loop decides the current — but it blocks
  the RL policy, which was trained against the datasheet servo and says so in its
  own run notes.
- **The guard's limits are not measured.** `safety.Limits` defaults to 65 °C, 2.0 A
  held for 0.3 s, 9.5 V and 0.35 rad of tracking error. The first three want
  checking against what a real footfall does; the last against the servo's actual
  lag at the gait's joint rates.

### Margins worth knowing

The trot stays inside the soft limits on its own — the runtime's clamp never bit
over a sweep of commands up to 0.45 m/s — but the knee gets close: 95.3° against a
99.1° soft limit at the top of that range. At the demo's 0.20 m/s there is room.
The clamp stays anyway, because it has to hold for a source that is *not* the trot.

## The bench, in order

**Build it first.** The fixture is `3d/bench_rig.py` — it imports the same
`mini_dog.sleeve()` and hub pattern the robot uses, so the bench holds the servo
the way the robot will:

```bash
cd 3d && .venv/bin/python bench_rig.py       # -> out/bench/{step,stl} + the numbers
.venv/bin/python tools/slice_orca.py --machine "Qidi Q2 0.4 nozzle - Copy" \
    --process "0.20mm Standard @Qidi Q2 - Copy" --filament "QIDI НИТ petg черный" \
    --name bench_rig --walls 4 --infill 30% \
    servo_gauge bench_stand bench_arm_s bench_arm_l
```

| part | g | why |
|---|---|---|
| `servo_gauge` | 11.8 | **print this one alone first and stop.** It is half a sleeve with a hub arm: check that a real ST3215 drops into the bore, and that the arm's four M2.5 land on the hub, before spending five hours on a stand built out of the same measured numbers. |
| `bench_stand` | 140 | sleeve + the two M3 thrust bolts + a column; axis 150 mm above the base |
| `bench_arm_s` | 6.0 | light arm, reach 45 mm — the direct `J_m` measurement |
| `bench_arm_l` | 9.5 | heavy arm, reach 90 mm — the holds, and the second free swing |

One plate, **167 g, 6 h 04** on a Q2 at 0.2 mm / 0.4 nozzle / 4 walls / 30 %.
5.8 g of that is support and it is only in two places — inside the gauge (3.5 g)
and the base's two clamp slots (2.3 g). Everything else is a prism through the
build direction: the case bore prints vertical and unsupported, which is the
whole reason the parts are modelled in the servo's own frame and exported with no
rotation at all.

Non-printed, and the bench does not work without them:

- a supply whose voltage you can **set** (three points: 12.6 / 11.1 / 9.9 V);
- **two weighed tip masses** — ~250 g for the short arm, ~350-450 g for the long
  one, bolted through the ⌀5.3 and ⌀8.4 tip holes. A mass that swings on its own
  is a second pendulum and ruins every trajectory here. Two bands, both from
  `fit_bam.py`'s own thresholds: the short arm needs **≥ 227 g** or `m·g·r` does
  not clear twice the `tau_c` prior and the release is unreadable, and the long
  arm needs **≤ 530 g** or `J_load` passes the 35 % share it warns at.
  **Buy compact, not a dumbbell disc.** `J_load` is `m·r²` *plus the mass's own
  moment about the axis*, and on a flat disc bolted in the swing plane that second
  term is not small: 500 g of lead or a calibration weight is 4.2e-3 and 34 %, the
  same 500 g as a ⌀130 cast-iron plate is 5.2e-3 and **39 %** — over the line on
  shape alone. Keep the weight under ~50 mm across and it never comes up. A
  kitchen scale is accurate enough: 2 g on 350 g is 0.6 %, against the 6.5 % the
  fit recovers `k_u` to on noise-free data. Scrap metal is fine and the material
  barely matters — 350 g is a 51 mm aluminium cube at 27 % share against 31 mm of
  lead at 27 %; it was only ever the *shape*. Two rules for whatever you bolt on:
  it goes **outboard of the arm's face and nowhere else** (a 51 mm cube straddling
  the plate shares 18 cm³ with the stand somewhere in the sweep — that face is the
  z = 20.3 line the whole layout is built on), and it has to be **one rigid body**.
  A rattling stack of offcuts brings its own friction and its own damping, and the
  free swing's decay envelope is precisely where `tau_c` and `b_v` are separated —
  Coulomb linear in time, viscous exponential — so the fit has nowhere to put it
  except into the servo. One chunk, or a stack pulled hard together through the M8
  on big washers. The offset it stands off the face is harmless: 350 g at 28 mm is
  0.094 N·m out of plane, 7 N per bolt pair on the ⌀14 circle — so **gain the mass
  along the bolt, never in the swing plane**. `I_zz` is set by the in-plane extent
  and does not care how tall the stack is, which makes 3 mm sheet the easiest of
  the lot: nine 40×40 squares are 339 g, 26 % share, 27 mm of stack, and they
  drill in one pass clamped together. Steel box section
  is the awkward case, because a hollow one has to be *long* to weigh anything:
  **50×50×3 is the smallest that works** — 350 g is 79 mm of it, 28 % share. Below
  that it fails both ways at once, and 25×25×2 is the illustration: 242 mm of it
  lies flat out to radius 211 against 150 mm of ground clearance, or stands 123 mm
  off the face, and the share is 36 % either way. Slide the piece onto a long M8
  **along its own axis** and pull it down with a washer over the far end: that
  loads the ring section in compression, needs no hole in the tube at all, and
  tunes by the length of the cut. Do not clamp it across two opposite walls — the
  bolt dents them, and a soft joint is the rattle above;
- **a clamp.** The stand only holds itself down by 1.4× with the heavy arm out
  horizontal — the two slots in the base take an M6 or a G-clamp, and
  `bench_rig.py` prints that ratio on every run;
- 4 × M2.5 × 6 into the driven hub, 2 × M3 × 10 **set screws** + nuts for the thrust
  clamp. Headless is the robot's spec and the reason is on the robot, not here — a
  cap head fouls the fork spine (`3d/README.md`, *The thrust clamp*). The bench has no
  fork, so either works on the stand; use the same screw so there is one box of them.

Bolt the arm on so it **hangs straight down** at the servo's centre position.
That is not a preference: `fit_bam.py` regresses against `m·g·r·sin(q)`, so q = 0
has to be the hanging position or every gravity term in the fit is wrong.

`bench_rig.py` ends by printing the exact `sweep.py` line for each arm — mass,
radius and the arm's own inertia off the real solid. Weigh the printed arm and
correct `--arm-inertia` if it disagrees: the free swing *subtracts* that number
rather than fitting it.

The two things that decide the quality of the result are a supply whose voltage
you can *set* and two different arms.

1. **`--check`** — preflight. Prints every control register. Nothing moves.

2. **`--traj rock`** — 30 seconds, and it settles a modelling question rather
   than fitting a number. Torque off, rock the horn against the play, watch
   Present Position. If it moves by ~0.5°, the encoder is after the gearbox: it
   reads the true joint angle, and the backlash is a hole in the *torque* path,
   not in the measurement. If it does not move, the encoder is before the
   gearbox and the robot cannot observe the play at all. The vendor wiki in
   `3d/ref/` implies the former; `rl/actuator.py`'s `enc_after_backlash` default
   assumes it; step 4's observation wiring depends on it.

3. **`--traj freeswing`, light short arm.** This is the one direct measurement of
   the reflected rotor inertia, which `rl/checks/check_model.py` found to be
   **73× the knee link's own inertia** and which is currently a guess in
   `3d/export_sim.py`. Use the *smallest* mass at the *shortest* radius that
   still overcomes the friction: J_m comes out as J_total − J_arm, a difference
   of two numbers, and a heavy arm destroys it. **Run it with two different arms** —
   one arm cannot separate the inertia from the Coulomb friction, because a
   heavier J with less friction fits the same fall.

4. **`--traj hold`, heavy long arm, at three voltages.** Here a big torque is
   exactly what is wanted. At steady state the torque balance is exact and two
   linear regressions give the torque constant and the loop gain.

5. **`--traj step` / `triangle` / `reversal` / `chirp`**, same arm, same three
   voltages.

The three voltages are not thoroughness. At one voltage the back-EMF damping and
the viscous friction enter every equation as the same coefficient of ω and no
amount of data separates them; `fit_bam.py` checks for this and says so rather
than returning a confident number. Same for the current channel — if
`PRESENT_CURRENT` is at the wrong address for your firmware the fit is
under-determined and will say so.

**Every run records the servo's control registers**, and the fit refuses to merge
runs whose registers disagree. A fit is only valid for the P/D/I coefficients,
dead zone, punch and acceleration the servo had at the time — and the robot then
has to run with the same ones. Changing `P_COEF` after fitting invalidates the
model as thoroughly as changing the gearbox.

## How the fit works, and what it can and cannot find

`fit_bam.py` runs analytic passes first and a trajectory refinement last, in that
order, because the analytic passes are where the identification actually happens:

| pass | gives | from |
|---|---|---|
| free swing | `J_m`, `tau_c` | release acceleration, or the period if it oscillates |
| holds | `k_u·R`, `kp/R`, dead-zone intercept | two linear regressions at ω = 0 |
| saturation | `1/R`, `k_e/R` | one regression over the samples where the duty is pinned |
| reversal | dead zone | the smallest command that actually moves the joint |
| refinement | everything, a little | Powell over a diverse subset — **opt-in, `--refine`** |

Two design decisions in there were forced by measurement, not taste, and both are
recorded in the code:

- **The fit does not run in the model's own coordinates.** R, k_e, k_u and kp are
  coupled — the instruments only ever see `k_u·R`, `kp/R`, `k_e/R` and `1/R` — so
  fitting R and k_u separately lets the optimiser walk along `k_u·R = const`
  where nothing observable changes. Observed on synthetic data with a known
  answer: R recovered to 0.1 % while k_u sat 34 % away on its seed.

- **The refinement is Powell, not `least_squares`, and it is off by default.**
  The residual is not smooth in the parameters: dead zone, punch, backlash
  engagement and duty saturation are all switches, and moving a saturated step
  edge by one sample changes the residual there by an amp. Finite differences
  across that read a first-order optimality of ~10⁹, the trust region collapses
  to steps of 10⁻⁶, and `trf` reports convergence after thirteen evaluations
  having taken a step of exactly zero. Powell does not differentiate and so is
  not defeated by it — but on synthetic data with a known answer it still came
  back 3.5× worse than the analytic seeds after 200 evaluations, and the guard
  discarded its result. So the analytic passes are the fit; `--refine` runs the
  trajectory pass if you want it, and it can decline but not damage.

What it currently recovers, from noise-free synthetic data (`--selftest`):

| identified | error | | not identified here | why |
|---|---|---|---|---|
| `R` | 3.7 % | | `b_v` | the free swing does not oscillate, so it carries no viscous information |
| `k_e` | 2.2 % | | `deadband` | the 8 s window stops before the reversal reaches its small commands |
| `k_u` | 6.5 % | | `punch` | visible in the current just outside the dead zone, same truncation |
| `k_u·R` | 3.1 % | | `theta_bl` | needs a full loaded triangle to show its hysteresis |
| `kp` | 14.8 % | | | |
| `J_m` | 5.2 % | | | |
| `tau_c` | 24.5 % | | | |

The right-hand column is not a list of excuses: the self-test truncates every run
to 8 s to stay quick, and the full trajectories reach the commands those four
need. Run them in full on hardware.

`--selftest` generates data from a *known* servo and tries to find it again. It
is worth more than it looks: if a parameter cannot be recovered from noise-free
data produced by the very model being fitted, no amount of real data will recover
it either — the trajectory set is wrong, not the servo.

## Safety, on the bench and later

`sweep.py` aborts on temperature, on current and on travel, and always leaves
torque disabled in a `finally`. The ST3215's own protection registers
(`PROTECTION_CURRENT`, `OVERLOAD_TORQUE`, `PROTECTIVE_TORQUE`, `MAX_TEMPERATURE`)
are read and recorded but never written — they are the servo's last line and this
code has no business moving it.

Power the servo from the supply directly, not through the URT-1 and not through
USB: a stalled ST3215 draws 2.7 A and a debug adapter is not a power distribution
board. Share the ground, nothing else.
