# Behaviour — making it a dog

What to build on top of the walker and Nav2 so the robot reads as an animal, not a
platform. One week of evenings. No LLM in this pass: everything below is a state machine
over `/scan`, `/imu`, the camera and a microphone, and it has to be fun with no model at
all before a model is allowed near it.

**Decision:** a `smalldog_dog` node that owns `/cmd_vel` when the pad and the explorer do
not, runs at 10 Hz, and is nothing but *wants* → *mood* → *reflex*. Nav2 stays the legs
for anything farther than 1 m; short moves go straight to `/cmd_vel`.

## What it has to work with

| sense | topic / source | what it can tell |
|---|---|---|
| L2 | `/scan`, `/lidar/points` | walls, doorways, a person as a moving 0.3–0.6 m blob, a drop in the floor |
| IMU | `/imu` | tilt, being picked up (free fall + rotation), being patted (a tap on z) |
| camera | IMX415, no node yet | motion in the frame, a face, a hand, brightness (lights off) |
| mic | **to buy**, USB | loudness, direction (two mics), a clap, a name |
| speaker | **to buy**, USB or the HDMI out | one short sound per event, never speech in this pass |
| servos | `/joint_states` effort | a leg being held or pushed — the runtime already reads Present Load |
| map | `/map`, slam_toolbox | where it is, where it has not been, where the person usually is |

## The three layers

**Reflexes** — run every tick, override everything, take no decisions.
- fall guard and tilt trip (exist, `robot/runtime/safety.py`)
- keep 0.5 m from anything closer than that in the scan, back off slowly
- a drop in the floor is a wall (exists in `cloud_to_scan`); a drop within 0.4 m is a
  freeze-then-back, whatever the mood
- picked up (IMU): legs go limp — `stand:=false` — and stay limp until set down for 2 s

**Mood** — four numbers in [0, 1], drifting slowly, set by events:
- `energy` — decays with walking, refills lying down; low energy = sleep
- `curiosity` — rises with time since anything new, drops when a room is fully mapped
- `attachment` — rises when the person is near and quiet, drops when ignored or shouted at
- `alarm` — spikes on a loud noise, a fast blob, a kick; decays in a minute

Mood is what a bigger brain would set later. In this pass it is arithmetic.

**Wants** — a priority list; the top entry whose condition holds gets the body:

| want | condition | what it does | done when |
|---|---|---|---|
| don't die | battery low (`/smalldog/battery`, verify the topic) | walks to the charger corner, lies down, stops walking | plugged in |
| safety | `alarm` > 0.7 | retreats behind the nearest wall corner in the map, faces the room, freezes | alarm < 0.3 |
| greet | a blob enters from a doorway after > 10 min alone | goes to it fast, stops at 0.6 m, wiggles (roll ±5° at 2 Hz, 3 s), then sits | 5 s |
| follow | `attachment` > 0.5 and the person walks off | keeps 1 m behind, stops when they stop, sits after 20 s of them not moving | person gone 30 s |
| check on you | no blob movement for 1 h, person's usual spot known | walks to the spot, looks (yaw ±30°), one sound, walks off | always |
| explore | `curiosity` > 0.6 and frontiers left | the explorer, slow speed (0.06) | no frontier, or curiosity < 0.3 |
| investigate | a loud noise (mic) or a new object in the scan that was not in the map | walks toward it to 0.8 m, stares 5 s, decides: nothing (mood ↓) or alarm | 5 s |
| beg | the person sits still for > 2 min and `attachment` > 0.6 | sits at 0.8 m facing them, one sound every 60 s, gives up after 3 | patted (IMU tap) or 3 min |
| sleep | `energy` < 0.2 | goes to the last sleep spot (or the nearest wall), lies down, servos torque-off | energy > 0.8, or any event above `alarm` 0.5 |
| idle | nothing else | the default, below | — |

**Idle is most of the day and is the whole illusion.** A stopped robot must never be
still:
- every 3–8 s a look-around: yaw the body ±15° with the pitch axes, back to centre
- every 20–40 s a small step, turn, or a stretch (front legs down, rear up, 2 s)
- lies down after 3 min of nothing; stands up on any event
- when a person is in the scan, the "front" of the body tracks them slowly

## Sounds

One short synthesised sound per event, chosen from about six: greet, question, content,
alarm, complaint, sleepy. Not speech. Speech is what turns it into an assistant. A dog
you can understand is a toy; a dog you have to interpret is a pet.

## What "fun" actually rests on

- **It wants things and you can see it deciding.** A 1 s pause before a want changes,
  a look toward the new target, then the move.
- **It mostly ignores you.** Greet and follow should fire a few times an hour, not
  always. `attachment` decays on purpose.
- **It has a memory.** The person's usual spot, the sleep corner, the room it was told
  "no" in (a shout while investigating there marks the spot; curiosity there drops for a
  day). Stored in a JSON next to the map, survives a reboot.
- **It gets things wrong in dog-shaped ways.** The vacuum is an intruder every time. A
  reflection in a low mirror gets stared at. A hand near the face starts a greet.
- **Being handled means something.** Pick it up: limp, with a complaint sound. Set it
  down: shake (roll ±8°, 1 s), look around, carry on. A push on a leg while standing:
  push back a little (the runtime's load reading), then step away.

## What we do not do

- No LLM, no TTS. Revisit after a week of it being alive in the room, and then only as
  ears (speech → a want) and as the thing that sets mood from a camera frame.
- No face recognition, no names. One person, "the person".
- No tricks on command. Nothing responds to a word in this pass; the mic gives loudness,
  direction and a clap.
- No running. `speed` stays ≤ 0.08 in every want; the gait is not re-tuned for this.

## The week

| day | build | it can do |
|---|---|---|
| 1 | `smalldog_dog` node, `/cmd_vel` arbitration with the pad and explorer, idle looks and stretches | stand in the room and not be still |
| 2 | blob tracking on `/scan` (nearest moving cluster, its track), the person's usual spot | turn to face you, know where you sit |
| 3 | mood arithmetic, greet, follow, sleep | greet you at the door, follow to the kitchen, fall asleep |
| 4 | IMU events: picked up, set down, tap; go limp, shake, the push-back | be handled |
| 5 | mic (buy day 1): loudness, clap, direction; investigate, alarm, the "no" | react to noise, be told off |
| 6 | speaker, six sounds; memory JSON; check-on-you, beg | be a pest, quietly |
| 7 | camera motion + brightness: lights off → sleep, waving → greet; a day alive in the room | — |

Each day ends with the robot on the floor for an hour, and the note of what looked
like a dog and what looked like a bug in a loop.

## Open

- Battery topic: the runtime reads the bus voltage; is it published? (verify)
- Servo torque-off while lying: does the runtime allow it under ROS (`safety.py`)? If
  not, "sleep" holds the lying pose at low gain instead — hotter, but works.
- Where the person's blob goes when they sit: a chair leg and a shin look alike in a
  0.10–0.35 m slice. The camera may have to confirm "person".
- Speaker: the HDMI audio out needs a display attached to enumerate. A USB speaker is
  simpler.
- `/cmd_vel` arbitration: three writers now (pad, Nav2, dog). A mux node, or the dog
  node is the only writer and forwards the others.

## Review questions

