# Fast servos — the ST3215 without its firmware's position loop

Why every RL policy "danced in place" on the floor, what the servo turns out to be under
the firmware, and the plan to run it that way. Measured 2026-09-16 on a free servo (id 1)
from the Orange Pi, `robot/bench/{acc_register,pwm_loop,mode2_protect}.py`; the public
write-up is `ST3215_STS3215_measured_parameters.md`, "Control loop".

**Decision (proposed):** run all twelve servos in MODE 2 (open-loop PWM) with the position
loop on the Pi at the bus rate. Not taken yet — steps 1–3 below decide it.

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
- **The overload protection is off in MODE 2**: 5 s at duty 1000, load 100 %,
  `PROTECTION_TIME` 2 s — nothing tripped. `PROTECTION_CURRENT` (2 A, 2 s) is **not
  measured** — it needs a stalled horn.
- **Current**: the register peaks at 3.8 A (≈ 2.3 A real, the register reads 1.64×
  high) on the 5 Hz sine, 1.2 A rms; a saturated step from rest pulls stall current. Twelve
  servos doing that at once is ~8 A from the pack — `pack_sag.py`'s territory.
- **The bus from the Pi over the CH340 costs 2.0 ms per transaction whatever its size**,
  read or write. A 12-servo read + write tick is ~5.5 ms → ~180 Hz. An FTDI adapter at
  `latency_timer` 1 would roughly halve it.

## Plan

1. **Stall test** — clamp servo 1's horn (a clamp, not fingers: 3 N·m on a 25 mm horn is
   120 N), `mode2_protect.py --stall`. Decides whether the firmware still guards current in
   MODE 2 or `safety.py` must. ⏳ needs the bench.
2. **Runtime in MODE 2.** `loop.py` runs the PD at the bus rate on all twelve (kp 9000 /
   kd 150 / kff 200, duty ±1000, a duty slew or a current-derived clamp so twelve joints do
   not draw stall current together); `calib.py` reads centres after the mode switch;
   `safety.py` owns current, temperature, runaway, and puts MODE 0 back on exit. Test on
   the stand with the IK trot: `feasible_gait()`'s 0.11 m/s cap came from limit (2); in
   MODE 2 the ceiling is 4.7 rad/s and acceleration is free, so the trot alone should reach
   ~0.2 m/s. A result with no training.
3. **s4 on the floor in MODE 2.** It was trained against almost this actuator. If it
   walks, the whole idea is confirmed before any GPU time; if it dances, the difference is
   the host loop's 5 ms delay and the current limit, and that goes into the model.
4. **Then train.** `actuator.py` gets a host-loop mode: profile off, PD at ~180 Hz with a
   one-tick delay, duty clamp, the current limit. `chirp_gain.py` must reproduce the
   measured table above before a run starts. Start from s4 or bc-ft depending on step 3.

## What we do not do

- Lift the profile through a register — there is none.
- Change the servo family. Feetech HLS (current loop) or Dynamixel XC430 would do it
  natively, but that is a new CAD interface and a new bench.
- Train against the profiled model for speed. It walks at 0.1 m/s, 1 Hz, and that is its
  ceiling by construction.

## Questions for review

