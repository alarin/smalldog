# Fast servos — the ST3215 without its firmware's position loop

Why every RL policy "danced in place" on the floor, what the servo turns out to be under
the firmware, and the plan to run it that way. Measured 2026-09-16 on a free servo (id 1)
from the Orange Pi, `robot/bench/{acc_register,pwm_loop,mode2_protect}.py`; the public
write-up is `ST3215_STS3215_measured_parameters.md`, "Control loop".

**Decision:** run all twelve servos in MODE 2 (open-loop PWM) with the position loop on
the Pi at the bus rate. Taken 2026-09-16: steps 1–3 below are done and the s4 policy,
which danced in place under the firmware loop, walks at ~0.2 m/s under the host loop.

## The three speed limits, and which one mattered

1. **No-load speed 3.86 rad/s** — a firmware plateau, not back-EMF (the bridge at full
   duty does 5.0; 4.7 at the pack's 11.3 V).
2. **Loaded rate ceiling 3.28 rad/s** = `(stall − friction) / damping`. What
   `walk.py`'s `feasible_gait()` fits to; caps the IK trot at 0.11 m/s.
3. **Acceleration profile 7.7 rad/s².** The position loop chases an internal goal that
   ramps at a capped acceleration. A step's peak speed is `2.8·√step`. This is the whole
   frequency response: a ±15° sine passes 74 % at 1 Hz, 25 % at 2 Hz, **7 % at 5 Hz**.

RL trained against a model without (3) found a 5 Hz trot at ~3 rad/s joint speed; the
servos delivered 0.6 rad/s and the closed loop became a 0.74 s rock. Bipeds on the same
servo (Open Duck Mini v2, also STS3215, BAM-fitted, no profile in the model either) walk
because an imitation reward pins the gait to a 1–2 Hz reference where the servo still
passes 74 %. With a task reward and no reference, PPO exploits the missing term.

## The profile is a firmware ceiling, not a register

`ACCELERATION` (41) works in its documented unit, 100 counts/s² = 0.153 rad/s²:
ACC 10 → 1.6 rad/s², 49 → 7.4, 50 → 7.7. **Every write above 50 reads back 50, and 0
selects 50.** Nothing lifts it. Same cap on every Feetech STS/ST with this firmware,
whatever the gearbox (it is in output counts).

## Under the firmware: MODE 2

MODE 2 drives the bridge at a duty (register 44, sign bit 0x400). No profile: a free horn
reaches its steady speed in ~70 ms. A PD loop on the Pi, one servo, 250 Hz:

| | firmware loop | host loop, kp 3000 | **host loop, kp 9000 kd 150 kff 200** |
|---|---|---|---|
| ±15° at 1 Hz | 0.74 | 0.90 | **0.99**, lag 6° |
| 2 Hz | 0.25 | 0.72 | **0.97**, lag 12° |
| 3 Hz | — | 0.58 | **0.95**, lag 17° |
| 5 Hz | 0.07 | 0.42 | **0.62**, lag 72° — the motor's 4.7 rad/s ceiling, not the loop |
| 0.1 rad step, peak speed | 0.8 rad/s | 1.1 | **2.9** (≈ 80 rad/s²) |
| standing error | — | 0.5° | 0.05° |

kp 9000 duty/rad at 11.3 V is ≈ 41 N·m/rad — the firmware's own stiffness, so the s4
policy's model (kp 40.9, no profile) was nearly this robot.

Facts that shape the runtime:

- **`PRESENT_POSITION` is raw in MODE 2** — `OFFSET` (31) is not applied. A centre read
  in position mode is off by exactly `OFFSET`. `calib.json` centres are in the
  position-mode frame.
- **The duty sign is opposite to the encoder's**: +duty turns q negative.
- **The load-based overload protection is off in MODE 2** (5 s free at duty 1000, load
  100 %, nothing tripped) **but `PROTECTION_CURRENT` still acts**: stalled against a clamp
  at duty 1000 the register reads 3.4 → 2.8 A (≈ 2.1 → 1.7 A real) and at 2.0 s the
  firmware cuts torque (status 32, then 8; torque-off + MODE 0 clears it). A joint pinned
  for 2 s goes dead mid-gait — `safety.py` has to clamp current before the firmware does.
- **Current**: the register peaks at 3.8 A (≈ 2.3 A real, the register reads 1.64×
  high) on the 5 Hz sine, 1.2 A rms; a saturated step from rest pulls stall current. Twelve
  servos doing that at once is ~8 A from the pack — `pack_sag.py`'s territory.
- **The bus from the Pi over the CH340 costs 2.0 ms per transaction whatever its size**,
  read or write. A 12-servo read + write tick is ~5.5 ms → ~180 Hz. An FTDI adapter at
  `latency_timer` 1 would roughly halve it. The mac's adapter does one servo at 1.9 kHz
  (0.05 ms per transaction) — the Pi's adapter is the slow part, not the bus.
- **A duty write turns torque on by itself** (`TORQUE_ENABLE` reads 1 after `GOAL_TIME`
  in MODE 2), so the exit order is duty 0, then torque off, then MODE 0.
- **Regen on a bench supply**: one servo braking at 3 Hz pushed the bus from 12.7 to
  15.2 V. The pack absorbs it; on a supply, run twelve of these off the pack.

## Plan

1. **Stall test** ✅ — the firmware still guards current in MODE 2 (2 A for 2 s, then the
   servo is dead until torque-off). `safety.py` must act below that: a per-servo current
   ceiling that folds the duty back, so a pinned joint softens instead of switching off.
2. **Runtime in MODE 2** — written, `robot/runtime/mode2.py`, `--mode2` on `walk.py` and
   `policy.py`. ✅ on one free servo through the real tick: 0.99 / 0.98 / 0.96 / 0.63 at
   1 / 2 / 3 / 5 Hz, the bench table. Centres shift by `OFFSET` at the switch (measured,
   cross-checked against the register — the raw frame appears ~50 ms after the MODE
   write); the duty folds back on the current register before the firmware's 2 A / 2 s
   cut; a runaway check per sub-tick; MODE 0, duty 0, torque off on every exit. Under it
   `feasible_gait()` caps the trot at 0.16 m/s (period 0.95 s) instead of 0.11.
   ✅ All twelve on the Pi: the loop runs at **165 Hz** (the CH340's 2 ms per
   transaction; a UART on the Pi or an ESP32 doing the sub-ticks would give 500+). Stand
   hanging and on the ground: 0.5–1.3° error, peak duty ~200. Trot in air: every joint
   reaches its full amplitude 20–40 ms behind, 1–2° median error (position mode: 30° at a
   slower gait). On the bench, `--go 2` at the fitted 0.16 m/s: **~20 cm in 2 s = 0.10 m/s**
   (position mode 0.083 at 0.11) — the feet slip on the bare bench and the gait's lift-off
   is a velocity step the loop now follows at full duty, so it looks jerky. Small win; the
   gait is the limit now, not the servo. Two facts from the bring-up: the target has to
   slide across the 50 Hz tick (a staircase at kp 9000 is a 600-duty kick every 20 ms and
   the load flapped ±500 in stance), and a loose joint mount looks exactly like a runaway
   (fl_roll after the stairwell fall: the runaway trip caught it twice before a hand did).
3. **s4 on the floor in MODE 2** ✅ — **it walks.** `policy.py ../rl/policy_s4 --mode2
   --kp 5220 --kd 0 --kff 0` (the model's own gains: kp 5.22 duty/rad, no kd, no
   feed-forward — with the trot's 9000 / 150 / 200 it dived forward at command 0, the kff
   turning every action step into a 1000-duty kick). At cmd 0.2 on the bench: **20 cm in
   1.5 s**, the gait starting 0.6 s after the command, so ~0.2 m/s walking — the sim's
   0.22 — with joints at 3–5 rad/s, pitch within −7°, roll ±3°, heading held to 0.3°.
   Twice, `bench/data/mode2_s4_vx02_{a,b}.npz`. The first run tipped at the handover
   because the 2 s stand-up ended nose-down (−16°) and the policy swung ±17° correcting;
   `--ramp 4 --stand-before 2` hands over at −3° and it stands dead still for the 2 s.
   The whole idea is confirmed before any GPU time.
4. **Then train** ⏳ — from s4; the spec is `TRAIN_HOST_LOOP.md`. `actuator.py` gets a host-loop mode: profile off, PD at
   165 Hz with a one-sub-tick delay (6 ms), duty clamp, the current fold (1.4 → 2.0 A
   register, floor 0.25). `chirp_gain.py` must reproduce the measured table above before
   a run starts. What to train for: the 0.6 s start-up and the pitch under way; s4 already
   walks at the sim's speed, so the gain is in robustness, not speed.
5. **The loop rate** — optional. 165 Hz is the USB adapter. The URT-1 on the Pi's own UART,
   or an ESP32 running the sub-ticks with the Pi sending targets at 50 Hz, would give 500+
   Hz and take the 2 ms bus cost off the Pi; do it if step 4's policy wants it.

## What we do not do

- Lift the profile through a register — there is none.
- Change the servo family. Feetech HLS (current loop) or Dynamixel XC430 would do it
  natively, but that is a new CAD interface and a new bench.
- Train against the profiled model for speed. It walks at 0.1 m/s, 1 Hz, and that is its
  ceiling by construction.

## Questions for review

