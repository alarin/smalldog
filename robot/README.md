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
| `bench/lift_test.py` | is foot contact visible in the servos at all? set it down, lift it, compare |
| `bench/lift_2p55kg.json` | that comparison at 2.55 kg, both plate positions, run-length encoded |
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

**Measured, twelve servos, mac + CDC adapter, 2026-09-07:** ping 0.27 ms, a
15-byte feedback read 0.47 ms, a 12-servo SyncRead 3.32 ms, and a full tick
(SyncRead then SyncWrite) **p50 ~3.4 ms, p99 ~4.0 ms** against a 20 ms budget.
The wire arithmetic above said 3.16 ms and the host adds about 0.3. Do not read
the mean the bus *scan* prints as this number: a scan spends most of its time
timing out on ids that do not exist, and its 9.2 ms mean is that artefact, not
the bus.

**There is also a rare transport stall of about 24 ms that is not understood and
is not the checksum defect below.** It shows up as a single SyncRead taking ~22 ms
where p95 is 3.9, and it costs 0 to 3 missed deadlines per 500 ticks — measured at
3/500 before the slot-9 fix and 0/500 and 2/500 after it, so the fix did not cause
it and did not cure it. It is almost certainly host-side (USB frame scheduling on
a mac CDC adapter) rather than the wire, which means the number to trust is the one
measured **on the Orange Pi**, not this one. Re-measure there before reading
anything into it; a stall that survives the move is worth chasing, one that does
not was the mac's.

**One frame in twelve arrives with a bad checksum, and it is a slot, not a servo.**
Whoever answers in **slot 9 of 12** gets the top two bits of its trailing checksum
byte set on about a quarter of SyncReads — `0x22` arrives as `0xe2`. The payload is
byte-identical to a good frame; only that last byte is wrong. Addressed on its own
the same servo is perfect over hundreds of reads, and rotating the id list moves the
fault to whoever now sits in slot 9. A UART sends LSB first, so bits 6–7 are the
last on the wire, which is what a talker releasing the half-duplex line a bit-time
early looks like against an idle-high line. `Bus.sync_read` therefore drops that
frame and re-reads **only that id** (0.47 ms) instead of raising, which used to
throw away eleven good frames and cost a full twelve-servo re-read on a quarter of
the ticks. That is a systematic 2 ms off the typical tick, not a cure for the stall
above — the two are separate and the stall outlives the fix. The
re-read is a re-read, never a repair: masking the two bits back off would be
believing a frame precisely because it is corrupt. `--profile` prints the count as
`N checksum (M re-read)`; M rising with N is this defect behaving as understood,
while timeouts or a rising `bus_errors` is something else.

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

**The runtime has now met twelve real servos** (2026-09-07, the mac, all twelve on
one bus at 12 V). What that run settled is recorded below and in
`runtime/calib.json`; what it did *not* touch is anything load-bearing — the robot
was inverted and unloaded the whole time, so `safety.Limits` is still unmeasured and
`rl/params/st3215.json` still says `fitted: false`. The bench, not the robot, is
what retires those.

### Bring-up, the order it has to happen in

1. **Program the ids.** `python runtime/calib.py --ids` prints the map the rest of
   the tree assumes: **`<leg><joint>`**, the leg counted round the robot from the
   front right and the joint counted down the leg, so the twelve ids are
   11-13, 21-23, 31-33, 41-43.

   | leg | 1 front right | 2 front left | 3 rear left | 4 rear right |
   |---|---|---|---|---|
   | 1 roll (at the body) | 11 | 21 | 31 | 41 |
   | 2 pitch | 12 | 22 | 32 | 42 |
   | 3 knee (at the ground) | 13 | 23 | 33 | 43 |

   The leg digit is the builder's numbering and deliberately **not**
   `robot_params.json`'s `fl, fr, rl, rr` order — `calib.py`'s `LEG_DIGIT` is the
   one place the two are tied together, and the map is written down there rather
   than derived, because there is nothing to derive it from. Two things fall out
   of the scheme and both are worth having: no id lands in 1..12, so a servo still
   carrying the factory default of **1** answers as nobody instead of impersonating
   a joint; and the ids run to 43, so anything sweeping the bus has to go past 43
   (`python feetech/bus.py --scan` does).

   Set them with the Feetech tool over the URT-1 **before assembly**
   (`3d/README.md`, "Assembly order", step 2): once a servo is inside a sleeve
   inside a leg, it is still on the bus, but it is a bad time to discover two of
   them answer to 22.

2. **`python runtime/walk.py --preflight`.** Pings all twelve, reads the control
   registers, and checks the host — including `latency_timer`, which ships at 16 ms
   on FTDI and would eat most of the 20 ms tick on its own. Nothing moves.

3. **Centre the servos, then `python runtime/calib.py --capture`.** Torque off;
   hold the robot at the model's mechanical zero — legs straight down — and read.
   This is the only thing that ties the servo's counts to the model's radians, and
   it is a property of *this* assembly: the hub bolts on in any of four positions.

   **Capturing alone is not enough, and the first assembly proved it.** As built,
   six of the twelve joints sat within 330 counts of the 0/4095 encoder wrap and
   could not reach their own soft limits — `fr_knee` needed ±1128 counts and had
   16. `Servo.to_counts` clamps at 0 and 4095 without complaint, so a capture there
   bakes in joints that stop mid-command with nothing reporting it, and wrapping in
   software is worse: with `MIN/MAX_ANGLE_LIMIT` at 0/4095 the servo drives to the
   absolute count, so a wrapped goal sends the leg the long way round through its
   whole ROM. The cure is the servo's own middle-position calibration — write
   **128 to `TORQUE_ENABLE`** while the joint is held at mechanical zero, and it
   computes the `OFFSET` itself. Park each servo's `GOAL_POSITION` on its present
   position first, so a firmware that reads the write as a plain torque-enable holds
   still instead of snapping. The write gets no reply — the servo is busy burning
   EEPROM — so verify by reading back, not by the ack. Then `--capture` records the
   residual and every joint has ±2047 counts to work in.

4. **`python runtime/calib.py --sign all`.** Moves each joint 8.6° and asks a human
   which way it went, because the bus cannot answer it — the servo reports its own
   counts happily whichever way round the fork went on. A wrong sign on one knee is
   a leg that drives itself into the floor the moment torque arrives, so it is
   asked rather than assumed. The prompts are written against the model's
   convention (+X forward, +Y left, +Z up, identical axes on all four legs): a
   positive **roll** swings the foot to the robot's left, a positive **pitch**
   swings it backward, a positive **knee** folds the shin back and shortens the leg.

   **Those prompts are in the robot's frame, and bring-up happens with the robot
   upside down.** Belly-up, left and right reverse — and rear-left and rear-right
   swap with them, which is the second-order version of the same mistake. On the
   first assembly three of the twelve answers came back wrong, all of them roll,
   all from this; pitch and knee were untouched because their prompts ask about
   fore/aft, which the flip leaves alone. Answering was not the problem, *seeing*
   was. So do not settle roll by eye: command one axis on all four legs at once and
   compare the legs against each other, which never requires naming a side. For the
   absolute sense, mark a side first — twitch the knees of legs 2 and 3 — and then
   ask whether a positive roll goes toward the marked side. The map this robot
   came out with is worth knowing before the next one: **roll splits front/rear
   (+1,+1,−1,−1) while pitch and knee split left/right (+1,−1,+1,−1).**

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

  **Recorded 2026-09-07, and the signal is real.** `runtime/contact_baseline.json`
  is 30 s of trot at 0.20 m/s hanging free, 100 % phase coverage on all four legs.
  The curve is smooth and near-sinusoidal over the gait cycle, **±185 units of
  Present Load** about a mean of zero, and the four legs agree closely. Two
  independent 30 s recordings correlate at **0.995** with an RMS difference of 3 %
  of the range, so the baseline repeats to about 11 units and a contact has that
  much to beat. That retires this module's stated doubt — "the result says the
  information is there in principle; it does not say the number the servo reports
  carries it" — for the in-air half. The other half — how far a real footfall moves
  the load off this curve — is measured statically below at 2.55 kg and still open
  dynamically.

  **At 1.55 kg the contact signal is not there on every leg, and that is the
  gearbox.** `bench/lift_test.py` holds the standing pose while the robot is set
  down, lifted clear and set down again — same pose throughout, so the only thing
  that changes is what is under the feet. Measured with no battery, no LiDAR and no
  Pi (62 % of design mass, ~3.8 N per leg): `fl_knee` moved 56 → 0 → 48, a clean
  52-unit step against a baseline that repeats to 11. `fl_roll` moved 40 and
  `rr_pitch` 32. But `fr_knee` and `rl_knee` moved **nothing at all**, and the
  whole rear-left leg reported nothing on any of its three joints. The two
  set-down readings do not even agree with each other (`fl_knee` 56 then 48).

  That is stiction, and it is the same actuator fact `robot/bench` found from the
  other side — the ST3215's friction is 6× the model's, and it held the arm the
  motor was meant to. Below the friction the motor does nothing whether the foot is
  loaded or not, so the duty the servo reports as Present Load carries no
  information. Load is quantised in steps of 8 here, which is not the limit.

  So `--contact` was **not usable at 1.55 kg**, and a threshold found there would
  have been fitted to one leg.

  **At 2.55 kg it is, on all four legs.** Re-run 2026-09-08 with a 1 kg plate on
  the deck — 102 % of the 2.499 kg design mass, though a plate is one lump where
  the missing 937 g is spread, so this brackets the finished robot rather than
  being it. Knee only, which is what `contact.py` reads, |DOWN − LIFTED| in
  Present Load units, measured twice with the plate in two places:

  | leg | plate forward | plate centred |
  |---|---|---|
  | fl | 56 | 56 |
  | fr | 24 | 24 |
  | rl | **8** | **48** |
  | rr | 8 | 16 |

  **The front/rear split is load distribution, not per-servo scatter.** Moving the
  plate back took `rl_knee` from 8 to 48 and `fr_roll`/`fr_pitch` from 0 to 24/32,
  while the front legs held. That is the difference between "two legs are deaf" and
  "two legs were unloaded", and it is why the plate's position is recorded here:
  the real robot carries 267 g of LiDAR, camera and GPS at the nose against a
  central 420 g battery, so it sits between these two columns.

  **The number repeats now, which is the thing 1.55 kg could not do.** At 1.55 kg
  the two set-down readings inside a *single* run disagreed (`fl_knee` 56 then 48).
  Here the joints whose load barely changed between the two runs reproduced **to
  the unit** — `fl_knee` 56/56, `fr_knee` 24/24, `rl_pitch` 32/32, `rl_roll` 0/0.
  Interquartile spread inside every window of both runs is 0 but for a single 8:
  Present Load is *latched*, not noisy. It holds dead flat for the seven seconds
  the robot hangs and steps only when what is under the foot changes, which is a
  much better signal to threshold than the ±185-unit in-air swing suggests.

  Two things to carry into `--contact-threshold`. **The sign is per-leg** — fl/rl
  positive, fr/rr negative, which is `calib.py`'s knee sign map +1,−1,+1,−1, so
  the residual needs the sign or an absolute value. And **`rr` is the marginal
  leg** at 16 units: above the baseline's 11-unit repeatability, under the 24 bar.
  That may well not matter — this is a static test with the feet placed by hand,
  and the gait wants a touchdown *edge* inside a swing rather than a calibrated
  force — but it is the leg to watch when the threshold is set.

  **The first window `lift_test.py` prints is not a contact reading.** It is the
  state the 2 s engage ramp leaves behind, legs driven into the table and the
  gearbox wound up, and it runs 2–4× the second window (`fl_knee` 136 against 56).
  The script used to average the two DOWN windows into one `delta`, so the 1.55 kg
  verdicts above carry that artifact; it now reports DOWN − LIFTED alone as
  `signal` and the wound-up one beside it as `ramped`.

  **The dynamic case is now measured too, and `--contact-threshold` is 50.** Same
  day, same 2.55 kg. `walk.py --baseline` records the free-air curve and the *same
  command with the robot on the bench* records the loaded one; the difference,
  per leg per phase bin, is the contact signal:

  | leg | stance median | swing median | air repeatability |
  |---|---|---|---|
  | fl | 104 | 10 | 1.8 |
  | fr | 65 | 10 | 1.7 |
  | rl | 86 | 1 | 2.3 |
  | rr | 68 | 2 | 1.7 |

  Separation is 55–94 units against a noise floor of ~2, i.e. **32 to 53 σ**, and
  the swing half sits at zero on all four legs, which is the air subtraction doing
  exactly its job. Two independent free-air recordings correlate at **r = 1.000**
  with an RMS difference of 2.3–3.3 units on a ~680-unit range — four times tighter
  than the 11 units the fast gait managed. `runtime/contact_baseline_slow.json` is
  that baseline; `bench/contact_2p55kg.json` is all three curves and the residual.

  **It had to be measured at a slow gait, and that is a finding about the demo gait,
  not a limitation of the method.** `gait.py` rate-limits its own output to
  `joint_velocity_limit × 0.85` = 4.0 rad/s. At 0.20 m/s the trot demands **7.55
  rad/s — 89 % over the limiter** — so the commanded foot path is clipped before a
  servo ever sees it, and `--period` cannot help because `period_for()` pins the
  period to 0.45 s at that speed whatever is passed. The robot then drags: the
  residual peaked in the half the gait calls *swing*, on all four legs, in two
  independent runs. That is the foot on the ground through the whole cycle, fighting
  hardest when it is being told to lift. Measured travel was **400 mm in 6 s, 0.067
  m/s against 0.20 commanded** and against 0.156 m/s for the same gait in MuJoCo.

  The measurement gait is **speed 0.02 m/s, period 1.35 s**, demand 3.36 rad/s,
  inside the limiter. Stride shrinks to 13.5 mm while the lift stays the full 22 mm,
  so there is almost no tangential force to slip on — which matters, because this
  robot slips (below). Tracking error fell from 24–26° to 16.8° at this operating
  point, and the in-air load range nearly doubled, 360 → 680 units, because the foot
  finally completes the lift it is commanded. **Do not carry the threshold of 50
  back to the 0.20 m/s demo gait**: at that operating point there is no clean swing
  phase to subtract against, and the number means nothing.

  **There is now a baseline for the gait the robot actually walks at**, which the
  measurement gait is not — `runtime/contact_baseline_walk.json`, recorded at the
  fitted 0.14 m/s / 1.20 s. Two independent 30 s hangs correlate at **r = 1.000**
  with a noise floor of 1.8–2.3 units on a 719–756 unit range, the same statistics
  as the slow gait, and `bench/contact_walk_air_b.json` is the repeat that says so.
  Run it with `--contact runtime/contact_baseline_walk.json` and `mismatch()` stays
  quiet; point `--contact` at the *slow* baseline while walking and it will tell you
  the period and speed are wrong, which is the mechanism working.

  **Its threshold of 50 is provisional, and the bench is why.** The number was
  measured at the slow gait; carrying it over is an argument from the ranges being
  close (752 against 686) and from 50 sitting between a 55–94 unit separation and a
  ~2 unit noise floor, not a measurement at this gait. Closing it needs a *ground*
  run at 0.14 m/s, and that is what the bench cannot give: 40 cm of travel is about
  three seconds, which is 2.5 gait cycles against 60 phase bins, and at period 1.20
  the phase advances almost exactly one bin per tick. The air half is free — hang it
  and record for as long as you like — but the loaded half needs a corridor. Do that
  before trusting `--contact` to steer anything, and re-check 50 against the stance
  and swing medians the way `bench/contact_2p55kg.json` does.

  Two bins in sixty is also the limit of the recorder: the phase advances `dt/period`
  per tick, so a period under about 1.2 s at 50 Hz skips bins outright and
  `Baseline.coverage()` reports 75 % or worse. It says "run it longer", which is
  wrong — that is aliasing, and longer does not fill a bin the phase never lands in.

  **The sign is per leg, and a scalar could never have worked.** `ServoContact` took
  one `sign` for all four legs, which is right in MuJoCo where the "load" is a clean
  `kp·err − kv·qvel`. On hardware the knee hubs are mirrored left to right: fl and rl
  rise when loaded, fr and rr drop. Replayed against the measured curves, the scalar
  that would have shipped gives **duty 0.00 on fl and rl — two legs of four
  permanently blind**, whichever value is chosen. It now takes a `{leg: ±1}` dict and
  `walk.py --contact-sign auto` (the default) fills it from `calib.json`'s knee
  signs — which came out **+1 −1 +1 −1, identical to the map derived independently
  from the load residuals**, so nothing new has to be calibrated for it.

  What the contact says about this robot, at a threshold of 50: the front feet carry
  for a duty of **0.68 against the commanded 0.50** — touchdown 7–8 % early, liftoff
  10–12 % late — and the rear feet for 0.40–0.43. That is the 1 kg plate sitting
  forward, showing up in the timing exactly as it did in the static lift test. The
  gait's commanded phase is therefore *not* ground truth for contact, and scoring the
  detector against it (82 %) measures the robot's weight distribution, not the
  detector.

  **The feet are the wrong shape for the floor, and the sim does not know.** `foot()`
  is a true hemisphere, R = 13 mm, with no flat. At 6.25 N per foot Hertz gives a
  contact patch of **2.5 mm diameter, 4.8 mm², at 1.3 MPa** (TPU ~95A; 3.4 mm and
  8.9 mm² at ~85A). Rubber friction has an adhesion term that scales with real
  contact area, so a point contact grips far worse than a flat pad of the same
  material — this is not the textbook case where area drops out. Meanwhile the foot
  geom ships `friction="1.2 0.02 0.001"` against a ground plane at `1.0`, so MuJoCo
  runs at μ ≈ 1.2 where TPU on bare bench is realistically 0.3–0.5. The sim assumes
  2.5–4× the grip the robot has, which is most of the 0.156 → 0.067 m/s gap on its
  own. A flat truncation of the sole is a one-parameter change to `foot()` and a full
  `3d/CLAUDE.md` verification ladder; it belongs in `MAC.md`'s queue, not here.

  **Hang it the right way up.** The curve is gravity plus inertia plus friction, and
  only gravity cares which way up the robot is — recorded inverted, the leg's own
  weight loads the knee the other way and the baseline is wrong by twice that term,
  silently, and in the direction of the thing being detected. Upside down is fine
  for calibration and for the first trot; it is not fine for this.
- **The servo model is still the vendor's.** `rl/params/st3215.json` says
  `fitted: false` out loud. That does not block the analytic trot — the trot
  commands positions and the servo's own loop decides the current — but it blocks
  the RL policy, which was trained against the datasheet servo and says so in its
  own run notes.
- **The guard's limits are only partly measured.** `safety.Limits` defaults to 65 °C,
  2.0 A held for 0.3 s, 9.5 V and 0.35 rad of tracking error. Current and voltage
  still want a real footfall at the real mass. Two of them are no longer guesses:

  **What the robot weighs today is not what the limits are for.** As it stands on
  the table it is the printed parts plus twelve servos and the URT-1: **1.55 kg,
  62 % of the 2.495 kg design mass.** The missing 940 g is battery cells 420, BMS
  55, Orange Pi/wiring 195, LiDAR 230, GPS 25, camera 12, IMU 3 — and every gram of it is
  *body* mass on the chassis, so the body is far lighter relative to the legs than
  it will ever be again. Nothing measured under load here transfers: not the
  current, not the sag, not the tracking error, and not the contact threshold.

  Standing at 1.55 kg the whole robot draws **0.4 A at 12 V** at the bench supply,
  while the twelve reported motor currents sum to about 0.07 A. The ~0.33 A
  difference is quiescent — roughly **27 mA per servo** — which says
  `PRESENT_CURRENT` is motor current only and does not see the servo's own
  electronics. Useful for the power budget, and a first sanity check on
  `CURRENT_LSB_A`; it is not a calibration of it, which wants the supply's reading
  against a known load. The holding current being this small is the friction again,
  not an error: the gearbox holds the stance, the motors barely work.

  **`PRESENT_TEMPERATURE` is unusable while the motor drives, and the guard now says
  so.** Still, it is good to ±1 °C — thousands of samples on a stationary ST3215
  produced not one outlier. Driving, the same servo at a true 30 °C returns isolated
  bytes of 41, 48, 54, 56, 98, and over a 21 s trot a raw maximum of **150 °C**, on
  both the SyncRead and the single-read path, with the voltage byte beside it correct
  in every frame. It is PWM noise reaching the ADC, and it tripped the very first
  `--stand`: `over temperature on fl_pitch: 65.00`, on a servo a direct read showed
  at 30 °C a second later. Temperature is the slowest thing the guard watches, so it
  is now median-filtered over nine samples and held for 0.5 s, like every other held
  limit — it was the only one that tripped on a single sample, and it was the one
  that least should have. `--profile` now reports `N temperature spikes discarded`
  beside the true peak; that count rising is the ADC, not the robot. Nine samples
  and not five, because the spikes are isolated: 12000 samples over a 20 s
  twelve-servo trot carried 102 elevated stretches and **every one was a single
  sample long**, so a five-window is fooled only when three spikes fall inside it —
  about a one-in-ten event over a 30 s run at that rate. It duly happened on the
  first `--baseline`, which reported 44 °C on servos a rest reading put at 30. A
  nine-window needs five spikes in nine; the same run then reported 32 °C against a
  true 29..32. Before torque
  comes on nothing is driving, so `Guard.check_at_rest` still refuses on one hot
  reading — that is where a genuinely hot servo gets caught.

  **The tracking limit is too tight for this gait.** Measured on the 21 s trot in
  free air: peak error **26.3°** against a 0.35 rad (20.1°) limit, so it already
  exceeds the threshold and survives only on the 0.3 s hold. Driven at the gait's own
  joint rate the servo lags a median 21.7° and a peak 67°, because ±0.5 rad in a
  0.45 s period asks for 6.98 rad/s against a 4.71 rad/s no-load speed — it
  saturates. Position itself is trustworthy while driving, unlike temperature: 0
  impossible jumps in 321 samples. So the number is real and the limit is the thing
  that is wrong. **It is now 0.70 rad, and that came from the loaded run, not from
  the estimate.** The first 3 s trot on the bench at 2.55 kg tripped after one second
  — `not tracking for 0.32 s … fl_pitch: 0.44 against a limit of 0.35` — on a robot
  doing exactly what it was told, which is a guess stopping the very run that could
  settle it. Re-run with the trip at 0.8 (`--track-rad`, new, because a limit nobody
  can set is a limit nobody can measure), the same trot peaks at **0.61 rad / 35°**,
  twice, reproducible to a tenth of a degree, and travels 350–450 mm in 3 s. 0.70
  keeps ~15 % over the peak; a reversed sign holds 1–2 rad and the 0.30 s hold is
  what separates a transient from a fault.

  Do not read 35° as "the servo tracks badly", and note which direction it moved:
  it is **worse at the fitted gait than at the old one** (35° against 26°) even
  though the joint-rate demand halved, because the stride nearly doubled — lower
  rate, larger excursion, and a loaded stance push against a gearbox whose friction
  is 6× the model's.

  Raising it also broke `safety.py --selftest`, which is the useful part: the test
  fed `q=0.5` as "a jam" against the old 0.35 trip, so at 0.70 the jam stopped being
  one and **the test would have gone green on a guard that no longer guards**. Both
  fixtures now derive from `Limits.q_err_rad`.

### The gait is fitted to the servo, and it doubled the robot's real speed

`walk.py` now refuses to command a trot these servos cannot fly. `gait.py` rate-limits
its own output to 4.0 rad/s; at 0.20 m/s the trot demands **7.55**, so the commanded
foot path is clipped before a servo sees it and the robot **drags**. Measured at
2.55 kg: 0.067 m/s achieved against 0.20 commanded, 0.86 A peaks, and the knee-load
residual peaking in the half the gait calls *swing* on all four legs, in two
independent runs — a foot that never leaves the ground.

`feasible_gait()` searches for the shortest period whose demand fits under the limiter
with the stride still under `max_step`, and caps the speed if none does. It prints what
it changed; `--as-commanded` restores the old behaviour and names the defect. What that
picks, and the two constraints together:

| commanded | best period | stride | demand | |
|---|---|---|---|---|
| 0.10 | 2.00 | 50 mm | 2.64 | comfortable |
| 0.12 | 1.65 | 50 mm | 3.16 | |
| **0.14** | **1.20** | **42 mm** | **3.77** | **what 0.20 is capped to** |
| 0.15 | 1.30 | 49 mm | 3.95 | at the limit |
| 0.20 | 1.00 | 50 mm | 5.24 | drags — infeasible at any period |

**0.20 m/s is not achievable by these servos at this mass**, at any period; cutting
swing to 15 mm only reaches 4.14 and costs the clearance that makes contact readable.
So the cap is not a slower robot, it is a faster one: **350–450 mm in 3 s = 0.12–0.15
m/s against 0.067**, better than 2×, and it lands on the commanded speed, which says
the slip went with the drag — feet that clear and land do not scuff.

Two things this deliberately does not do. It does not touch `gait.py`: **the sim does
not have this bug**, because MuJoCo's feet run at μ ≈ 1.2 and grip, so the sim robot
still makes 0.156 m/s, and fixing it in the gait would move every tuned number in
`ros2/README.md` and what `rl/` trains on to cure something only the hardware suffers.
And `clamp_profile()` caps the scripted demo's own velocities, because `PROFILE`
carries its own and the fit on `--speed` would never have reached it — the one run
meant to be shown to people would still have dragged.

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
| `servo_gauge` | 11.8 | **print this one alone first and stop.** It is half a sleeve with a hub arm: check that a real ST3215 drops into the bore, and that the arm's four M3 land on the hub, before spending five hours on a stand built out of the same measured numbers. |
| `bench_stand` | 154 | sleeve + the two M3 thrust bolts + a column; axis **190 mm** above the base — raised from 150 so the ⌀140 disc clears at the bottom of the swing. Modelled, not yet weighed |
| `bench_arm_s` | **5.07** | light arm, reach 45 mm — the direct `J_m` measurement |
| `bench_arm_l` | **8.17** | heavy arm, reach 90 mm — the holds, and the second free swing |
| `bench_bushing` × 2 | **4.27** ea. | centres the ⌀29 bore on the M6 tip bolt; without them the disc sits up to 11.5 mm off radius **and moves while swinging** |

The three bold figures are weighed off the real print (2026-09-08); the slicer's
fill factors were optimistic by 13 % on the arms and **46 % on the bushing**, which
had no `BENCH_FILL` entry at all and fell through to `mini_dog`'s thin-wall default
of 0.92. They are in `bench_rig.py` now. The two bushings ride at the tip radius, so
they are part of the **tip mass**: 8.5 g on 1053 is 0.8 %, and that is 0.8 % straight
into `k_t`, which the fit gets by dividing measured torque by `m·g·r`.

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
- **a clamp, and it is now mandatory.** The stand does **not** hold itself down: the
  heavy arm out horizontal is 0.330 N·m over the base's front edge against 0.161 N·m
  of stand + servo, i.e. **0.49×**, and `bench_rig.py` prints
  `!! IT WILL GO OVER unclamped` on every run. The 1.4× this line used to quote was
  the old rig, before `AXIS_H` went 150 → 190 to clear the disc — raising the axis
  raised the overturning moment and left the footprint where it was. The two slots in
  the base take an M6 or a G-clamp; fit one before the first release;
- 4 × M3 × 6 into the driven hub, 2 × M3 × 10 **set screws** + nuts for the thrust
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
   gearbox and the robot cannot observe the play at all.

   **ANSWERED 2026-09-08: the encoder is AFTER the gearbox** — Present Position
   moves when the horn is rocked against the play. The vendor wiki in `3d/ref/`
   implied it and `rl/actuator.py`'s `enc_after_backlash = True` assumed it; both
   are now measured rather than inherited. The robot can observe its own play, so
   step 4's observation wiring can use the read-back position as the true joint
   angle. Re-run this after any gearbox or servo swap, not otherwise.

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

6. **`--traj holdbi` and `--traj speed`, heavy long arm, same three voltages.**
   Ten minutes per voltage, and they are the only two measurements here that do
   not go through the fit — `bench/hysteresis.py` reads them by subtracting one
   run from another. `holdbi` walks the hold ladder up and then down so every
   angle is reached from both sides; `speed` is the same triangle at five periods
   from 20 s to 1 s. See "Two things the fit cannot find" below for what they
   found and for the reason they exist: friction is the one quantity an optimiser
   over this model will always hide somewhere else.

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
| **hold ladder, both ways** | **`tau_c`, `mu_load`** | **the up/down difference at each angle — a measurement, not a fit** |
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

**One entry in that report now means the opposite of the others, and it is worth
reading the distinction.** `mu_load` is measured cleanly on hardware and is *not*
recoverable from synthetic data — so for once the self-test is indicting the
MODEL rather than the experiment. See "The model has no static friction" below.

### The load-dependent friction term, and what it moved

`actuator.Params.mu_load` — N·m of friction per N·m crossing the gearbox — went in
on 2026-09-09 (PLAN.md step 2), with `fit_bam.seed_from_holdbi` measuring it off
the bidirectional ladder. It runs **first** among the analytic passes, because it
needs nothing and the free swing needs it: that pass gets `J_m` from
`(m·g·r·sin q₀ − tau_c)/α` and had been subtracting a `tau_c` half the real size.

| | before | after |
|---|---|---|
| `tau_c` | 0.084 | **0.184** |
| `J_m` | 0.0240 | **0.0165** |
| `mu_load` | — | **0.286** |
| position RMS | 8.96° | 8.84° |
| current RMS | 1.25 A | 1.01 A |

The `tau_c` number matters more than its size: the ladder and the free-swing fit
are independent routes that had disagreed by 2×, and they now land on **0.184 and
0.186**. `J_m` moving is a consequence, not a second finding — and it means the
**0.024–0.042 range this project has been quoting for `MJ_ARMATURE` is void**,
because it was computed against the old `tau_c`.

**One trap this cost, and the guard now in `Run`.** `--seconds` truncates every
run, because the cost of a fit is wall-clock and most runs say what they have to
say early. `holdbi` is 36 s against a default of 25, so its descending pass lost
two thirds of its rungs, the two halves stopped lining up, and the pass silently
found fewer than four paired angles and declined — no error, `mu_load` left on its
prior, and a fit that looked fine. A run whose measurement is a DIFFERENCE between
its first half and its second is not shortened by truncation, it is halved.
`Run.WHOLE_RUN` now names those.

### The model had no static friction — half fixed 2026-09-11

**`simulate()` sticks now.** It uses Karnopp instead of `tanh(w/v_eps)`: below
`v_eps`, friction opposes the **net applied torque** up to `tau_c + mu_load*|tau_t|`,
and the shaft is held rather than integrated. Measured on the model, a sub-breakaway
load with the bridge off used to creep at 0.013 rad/s and now comes to rest;
`actuator.py --selftest` probes both directions of that. A hold approached from below
and from above now settles at two different duties instead of one.

**The magnitude is not yet calibrated.** On the bench arm at 0.6 rad the model gives a
half-difference of 0.72 V against the real servo's 0.29 — right order, wrong number —
and the two are not strictly comparable, because the half-difference is "the friction at
that load" and the holdbi ladder's angles are not the one used here. Matching them needs
the ladder replayed through the model, not an adjustment to the law.

**What is NOT fixed is the training path**, and the split is forced rather than chosen.
`simulate()` has every torque in hand, so it can ask "is the net torque below breakaway".
`friction()`/`motor_torque()` cannot: `rl/env/walk.py` and `rl/eval.py` call them with
MuJoCo owning the load, so the net torque is not known at that point and no amount of
care inside `actuator.py` can recover it. Stick there has to be MuJoCo's own
`frictionloss`, which is a proper stick-slip constraint — and `rl/model.py` currently
zeroes it on the grounds that `actuator.py` supplies friction. That swap is the
remaining half of PLAN.md step 2b; it was not done here because this machine has no jax
and the change could not be run end to end.

**It also re-dates the fit.** `rl/params/st3215.json` was fitted against the old
`simulate()`, so it predates the model it describes and a refit is due.

This also means the model cannot reproduce something this project already
documented — that commanded corrections below the breakaway do not move the joint
at all. PLAN.md step 2b has the shape of the fix and the reason it is not a
one-liner: `simulate()` has every torque to hand, but `rl/env/walk.py` does not,
because MuJoCo owns the transmission there.

### The torque constant is 2.2× the datasheet, and it is not friction

The plan assumed the inflated `k_u` — a fitted stall of 4.23 N·m against a spec
2.94 — was missing friction being absorbed. It is not. **Cancelling friction
properly makes it worse:** the friction-cancelled paired holds imply 7.4 N·m.

| route | torque constant | implied stall @ 12 V |
|---|---|---|
| paired holds, current channel | 2.394 N·m/A | 7.40 |
| duty channel × R | 2.382 N·m/A | 7.37 |
| unidirectional holds (the old route) | 1.368 N·m/A | 4.23 |
| vendor spec | 1.089 N·m/A | 2.94 |

Two routes agreeing to 0.3 % and disagreeing with the vendor by 2.2× is a scale
error, not noise, and both candidates are already on this project's "not measured"
list: `PRESENT_CURRENT`'s 6.5 mA LSB, confirmed only against vendor numbers, and
`PRESENT_LOAD`'s per-mille scaling if 1000 is not 100 % duty. An external shunt or
an INA226 would settle it, and so would a scale — **and a scale did: 4.50 N·m,
see the next section. Neither row above is right.** **`tau_c` and `mu_load` are immune** — both are ratios of
quantities in the same register units converted through `m·g·r`, so a constant
scale error cancels — but no stall figure from this fit should be read as a torque
the robot has.

### The stall torque, in newton-metres

**4.50 N·m at 12 V**, measured 2026-09-10 on `3d/torque_rig.py` — a printed C-frame
with a 170 mm arm pressing an M6 anvil onto a 2 kg coffee scale. This is the
measurement `PLAN.md` step 2c was waiting for, and it closes it. `mini_dog.py`'s
`SERVO_STALL_NM` is this number now; it was the vendor's 2.94.

| limit | d·U | scale, cold | τ |
|---|---|---|---|
| 200 | 2.41 V | 0.408 kg | 0.680 N·m |
| 350 | 4.13 V | 0.780 kg | 1.300 N·m |
| 450 | 5.27 V | 1.100 kg | 1.834 N·m |

k_u = **0.400 N·m/V**, friction intercept **−0.30 N·m**, residuals ±0.05 N·m.
Reading the settled rather than the cold value gives 4.38 instead of 4.50.

**Both of the previous answers were wrong, in opposite directions.** The register
routes above implied 7.40 and the datasheet said 2.94; the truth is between them.
So the register channel is high by 1.64×, *not* the 2.2× the previous section
assumed, and the datasheet is optimistic by 1.53× on top of that. There was never
a single scale error to find. The **−0.30 N·m intercept is a third, independent
read on `tau_c`**, against the 0.43 N·m the hysteresis ladder inferred.

**The trap, and it cost most of the session: do not read a scale off
`sweep.py --traj stall`.** Its 0.8 s bursts are correct for identifying the
electrical side — they keep a locked rotor cool and rest at zero position error —
but a scale under an 0.8 s impulse reads the arm's inertia and its own filter
ringing on top of the static force. Measured directly: the same rung read **920 g
in bursts and 780 g held**, and at a lower rung 530 g against 408 g. That is a
roughly **constant ~130 g offset**, so it corrupts the slope as well as the level.
`bench/torque_hold.py` exists for this: one steady push, 8 s, with a per-second
current profile.

**The torque decays while you hold it, by 4–7 % in 8 s.** At fixed duty the servo
sets voltage, not current, so as the winding warms R rises and I falls with it —
0.361 → 0.337 A at limit 350, about 17 °C of winding rise, while the case moved
1 °C. At limit 200 the decay vanishes, because the motor is dissipating 0.28 W
instead of 1.5. So "stall torque" is not one number: the cold value is the one a
leg actually uses in a transient, and it is the one recorded above. `rl/actuator.py`
has no thermal term, so a policy leaning on stall for a sustained push is modelling
a servo ~5 % stronger than the real one after a few seconds.

**Three fixture faults, all of which produced convincing wrong answers first.**
The rig is self-reacting by design — the C-frame closes the load loop through its
own jaw — but a kitchen scale is 120–190 mm across and the jaw is only 33 mm
(`2 × FRAME_Z`), so the scale bridges a beam under its middle with its feet in
air: it rocks, and a distorted load-cell mounting reads several percent off.
Moving the scale to the table fixed that and broke the other half, because the
reaction then goes out through a bench clamp: with one clamp the stand rotated and
**leaned part of its own weight onto the scale**, inflating readings by 70 g at
every rung. A second clamp cured it. **The encoder cannot see any of this** — it
measures the output shaft relative to the servo's own case, so a case turning in
the sleeve, or a whole frame rotating, is invisible. "Zero encoder creep" is not
evidence the fixture is solid.

If this is run again: put the scale back **in the throat** where the design
intended, on a rigid plate spanning the jaw wide enough to carry its feet. That
restores self-reaction *and* supports the load cell, which is the combination
neither configuration achieved. There is room — the platform sits 32 mm below the
axis and the anvil boss reaches to 15, so a plate up to ~13 mm still leaves the
4 mm minimum reach.

**Still open.** All three rungs are 20–45 % duty, so 4.50 is a 2.2× extrapolation
and the pairwise slopes (0.360 then 0.470 N·m/V) are not perfectly straight. One
voltage only — step 4 of the rig's protocol wants 8/10/12 V, and each supply change
power-cycles the servo, which reloads `TORQUE_LIMIT` from `MAX_TORQUE`. And
`PRESENT_CURRENT` is still unusable below ~0.2 A: fitting (current, torque) gives a
+0.60 N·m intercept, which would be torque at zero current, so k_t and R remain
unresolved.

**`TORQUE_LIMIT` is volatile.** Register 48 is SRAM; the servo reloads it from
`MAX_TORQUE` on every power-up. `bench/torque_limit.py` writes it and reads it
back, and refuses to write while torque is enabled. Cap before every session and
after every supply change — nothing else stands between a 2 kg scale and a servo
that might be stronger than you think.

### The no-load speed, and the plateau it turned out to be

**3.86 rad/s at 12.1 V**, measured 2026-09-11 with `bench/noload_speed.py` on the
stand, hub free, no arm — the last vendor number in the chain, and the vendor's
4.71 (0.222 s/60°) was 22 % high. `mini_dog.py`'s `SERVO_NOLOAD_RADS` is 3.86 now.

| `TORQUE_LIMIT` | d·U | ω from position | ω from `PRESENT_SPEED` | ratio |
|---|---|---|---|---|
| 1000 | 12.0 V | **3.864 rad/s** | 3.835 | 1.008 |
| 800 | 9.6 | 3.851 | 3.835 | 1.004 |
| 600 | 7.2 | 3.093 | 2.953 | 1.048 |
| 400 | 4.8 | 2.077 | 1.994 | 1.041 |

Four 120° slews per rung, alternating direction, speed taken off `PRESENT_POSITION`
(known LSB) and the register compared against it. That comparison is the side
result: `SPEED_LSB_COUNTS_PER_S = 1.0` in `feetech/registers.py` was flagged
unverified, and the ratio column verifies it to 1–5 %.

**The main result is the plateau.** 800 and 1000 read the same speed, and the
register sits at exactly 2500 counts/s for both, while 400 and 600 are linear
through the origin at 0.43 rad/s/V — k_e = **2.32 V·s/rad**, between the 2.03
fit and the 2.55 vendor. So the duty stops mattering at a cap of roughly 740.
The tool's first fit (k_e = 3.92, intercept +1.08) was two plateau points dragged
through a line; it now finds the plateau and fits below it.

**Two readings, and the stall number depends on which** (`PLAN.md` 3c has the
consequences): either the position loop's profile caps at 2500 counts/s and
`GOAL_SPEED = 0` means "the firmware's maximum" rather than "unlimited", or the
position loop never drives the bridge past ~75 %. Under the second, the 4.50 N·m
above — extrapolated from rungs 200/350/450, all under this knee — is nearer 3.3.
`noload_speed.py --pwm` decides it: MODE 2, open-loop duty, the same rungs, no
position loop in the way. It was written after the adapter came out and has
**not run**. Ten seconds on the stand; do it before the next torque-rig session.

**Two things the tool got wrong on its first outing**, fixed the same day: `--min-cap`
(default 900) filtered the ladder's own rungs, so `--duty-ladder` silently ran one
rung — the guard now applies to the top rung only — and the fit above. It leaves
`TORQUE_LIMIT` at its last rung (400): re-cap before the arm goes back on, and
remember the register is volatile anyway.

### Two things the fit cannot find, and the ladders that measure them instead

Measured 2026-09-09, 1.066 kg on the 90 mm arm, at 8 / 10 / 12 V.
`bench/sweep.py --traj holdbi` and `--traj speed`, read by `bench/hysteresis.py`.

**The fit was never going to find friction, because it has three other places to put
it.** `rl/actuator.py`'s law is `tau_c*sign(w) + b_v*w`, both independent of load, so a
friction that grows with the torque carried has nowhere to go except `k_u` — which is
why the fitted stall is 4.23 N·m against a spec 2.94, and why `b_v` sits pinned on its
lower bound however much data it is given. Neither symptom is a shortage of runs.

So both new ladders are **differential**: they measure friction as a difference between
two runs of the same trajectory rather than asking an optimiser to infer it.

| | 8 V | 10 V | 12 V | must it move with V? |
|---|---|---|---|---|
| position-loop stiffness, N·m/rad | 28.1 | 34.4 | 40.9 | **yes**, 3.44 per volt |
| effective torque constant, N·m/V | 0.636 | 0.611 | 0.596 | no |
| friction at a hold, unloaded, N·m | 0.192 | 0.181 | 0.186 | no |
| …per N·m of load carried | 0.286 | 0.295 | 0.251 | no |
| kinetic Coulomb at zero load, N·m | 0.161 | 0.160 | 0.168 | no |
| total speed-proportional, N·m·s/rad | 1.403 | 1.364 | 1.334 | no |

The last four rows holding still across a 1.5× supply range is the check that they are
physical and not artefacts of the loop; the first row moving *is* the pack-discharge
effect, and it is the only one allowed to.

Three results, in order of how much they change:

1. **Friction grows with load: +0.28 N·m per N·m carried, on a 0.19 N·m floor.** At the
   0.88 N·m the long arm asks for, friction is 0.43 N·m. This was previously inferred
   from a torque-vs-duty slope ratio and put at +38 % of *motor* torque; the direct
   measurement is about +22 % on the same basis, so the correction runs downward.

2. **The stiffness was 40 % low, and the error hid inside the friction.** A ladder walked
   in one direction measures the whole standing offset from target and calls it droop,
   but only part of it is elastic — the rest is the friction band, which does not
   restore. 40.9 N·m/rad at 12 V, not 28.8. The two errors conceal each other in any
   test that only looks at where the joint ends up, which is most of them.

3. **`b_v` is not resolvable on this bench and the plan's premise for it was wrong.**
   The speed ladder does what it was meant to — a steady speed at last, five of them over
   a 20× range — and it pins the *total* speed-proportional torque to 1.37 N·m·s/rad
   within 5 %. But that total is `b_v + k_t·k_e`, and back-EMF alone accounts for 1.31 of
   it, leaving 0.06 ± 0.07: consistent with zero. The reason is structural. Back-EMF and
   viscous friction each cost a motor voltage proportional to ω and **neither depends on
   supply**, so the three-voltage sweep that separates every other electrical term is
   powerless here; splitting them needs the motor current, and `PRESENT_CURRENT` gives it
   only as `d²U/R` at a 6.5 mA LSB. Use the total. For a simulator it is the total that
   sets how the joint resists being moved.

**The trap this cost, and the guard now in the code.** The three voltages of new data
were captured, `fit_bam.py` was re-run, and every fitted parameter came back **identical
to five decimals** to a fit on the old data alone. Nothing had gone wrong with the
servo: the analytic passes select on `trajectory == "freeswing"` and `== "hold"`, so
`holdbi` and `speed` reached the objective only under `--refine`, and not at all without
it. A run that no pass consumes changes no parameter — so an unchanged fit is not
evidence that new data agreed with old. `check_identifiable` now names every trajectory
in the directory that no analytic pass reads, and says exactly that.

And with `--refine` on, the enlarged set does not converge, it *diverges*: `kp` pinned at
2017, `k_e` at 11.8, `punch` on its bound, an implied no-load speed of 1.02 rad/s against
a spec 4.71.

**That was read as missing friction, and it is not.** Step 2 added the load-dependent
term the same day and re-ran it: `kp` still pinned (1887), `k_e` still 11.3, no-load still
1.06 rad/s. The character of the divergence did not change at all. `kp` railing at its
bound while `k_e` grows to compensate is a fit reaching for torque the electrical model
cannot supply — the same shape as the torque-constant discrepancy below, and the same
prime suspect. Do not expect a friction term to fix it.

One loose end, flagged rather than guessed: at a commanded 2.0 rad/s the servo reaches
1.8 and no more, sitting at 0.74 rad of position error **without the duty ever pinning**,
so it is not the motor's ceiling. `D_COEF` is 32 and `rl/actuator.py` models `kd = 0`,
which would do exactly this — but so would an internal output clamp, and this data does
not separate them.

## Safety, on the bench and later

`sweep.py` aborts on temperature, on current and on travel, and always leaves
torque disabled in a `finally`. The ST3215's own protection registers
(`PROTECTION_CURRENT`, `OVERLOAD_TORQUE`, `PROTECTIVE_TORQUE`, `MAX_TEMPERATURE`)
are read and recorded but never written — they are the servo's last line and this
code has no business moving it.

Power the servo from the supply directly, not through the URT-1 and not through
USB: a stalled ST3215 draws 2.7 A and a debug adapter is not a power distribution
board. Share the ground, nothing else.
