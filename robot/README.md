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
| `runtime/` | the 50 Hz loop and the safety layer — step 7, not written yet |

Nothing here needs hardware to be exercised:

```bash
python -m feetech.bus --selftest      # packets, checksums, sign-magnitude, SyncWrite
python bench/bus_probe.py --dry-run   # the timing harness
python bench/sweep.py --dry-run --traj all
python bench/fit_bam.py --selftest    # generate a known servo, then find it again
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

## The bench, in order

**Build it first.** The parts list is in the project notes; the two things that
decide the quality of the result are a supply whose voltage you can *set* and
two different arms.

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
