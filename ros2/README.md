# SmallDog — ROS 2 + MuJoCo

12-DOF quadruped simulation for the printed ST3215 dog designed in [`../3d`](../3d).
Layout follows the hexapod project (`ogonek25-spider/ros2`): description → ros2_control →
gait node, with MuJoCo standing in for the hardware.

![trot](docs/trot.png)

## Packages

| Package | Type | Purpose |
|---|---|---|
| `smalldog_description` | ament_cmake | URDF, link meshes, MuJoCo model — **all generated from the CAD** |
| `smalldog_ros_control` | ament_cmake | ros2_control wiring + MuJoCo launch |
| `smalldog_walker` | ament_python | trot gait + analytic leg IK, `/cmd_vel` → joint trajectory |
| `smalldog_teleop` | ament_python | keyboard teleop |
| `tools/` | — | standalone MuJoCo sim, no ROS needed |

External dependency, same as the hexapod: **`mujoco_ros2_control`**. Add it next to the
packages (the spider project uses a fork pinned to the `kilted` branch):

```bash
git submodule add -b kilted git@github.com:ogonek-spider/mujoco_ros2_control.git
```

## Build & run

There is no ROS 2 in this repo — the workspace borrows the **pixi `kilted` environment and
the `mujoco_ros2_control` build from the spider project** next door. `tools/env.sh` wires
both up; the two run scripts source it, so you never assemble the environment by hand.

### Build once

```bash
source tools/env.sh
colcon build --symlink-install
```

### Run

Two terminals — the teleop reads raw keys, so it needs its own TTY. It cannot share a
terminal with the launch output, and it cannot be driven from a script's stdin.

```bash
# terminal 1 — MuJoCo + ros2_control + gait node
./tools/sim.sh

# terminal 2 — keyboard control
./tools/teleop.sh
```

`sim.sh` kills stray `robot_state_publisher` processes first: leftovers from a previous
launch keep the next `controller_manager` from ever coming up, and the symptom is an
endless `waiting for service /controller_manager/list_controllers`.

`teleop.sh` waits for `/smalldog_walker` to appear before starting, so the order of the two
terminals does not matter.

On macOS you can open the teleop in its own Terminal window from anywhere:

```bash
osascript -e 'tell application "Terminal" to do script "'$PWD'/tools/teleop.sh"'
```

### Stopping

```bash
pkill -f "ros2 launch smalldog"
pkill -f robot_state_publisher      # always, or the next launch hangs
```

`tools/env.sh` picks `local_setup.zsh` or `local_setup.bash` to match the running shell.
That matters: ROS 2's `local_setup.bash` finds its own directory through `$BASH_SOURCE`,
which zsh does not set, so sourcing the `.bash` file from zsh silently looks in `$PWD`
and the workspace overlay is never applied — `ros2 run smalldog_teleop keyboard` then
fails with "package not found" while `ros2` itself works fine.

### Doing it by hand

```bash
eval "$(pixi shell-hook --manifest-path ../../ogonek25-spider/ros2/pixi-robostack/pixi.toml -e kilted)"
source ../../ogonek25-spider/ros2/install/local_setup.bash   # mujoco_ros2_control
source install/local_setup.bash
ros2 launch smalldog_ros_control smalldog-mujoco.launch.py
ros2 run smalldog_teleop keyboard                            # second terminal
```

### Checking it is alive

```bash
ros2 control list_controllers        # both must say "active"
ros2 node list | grep smalldog       # walker, controller, keyboard_teleop
ros2 topic info /cmd_vel             # 1 publisher (teleop), 1 subscriber (walker)
ros2 topic echo /joint_states --once
```

If the legs do not move, check `/cmd_vel` first: the teleop status line in its own terminal
shows the `vx / vy / wz` it thinks it is sending.

## Keyboard

```
  w / s      walk forward / back
  a / d      strafe left / right
  q / e      turn left / right
  space      stop
  r / f      body up / down          (0.09 … 0.20 m)
  , / .      speed  -  / +           (0.05 … 0.45 m/s)
  t          gait enable / disable
```

Published: `/cmd_vel` (Twist), `/smalldog/body_height` (Float64), `/smalldog/enable` (Bool).

## Without ROS 2

The gait and IK have no ROS imports, so the whole thing runs from a bare Python env
with `mujoco` and `numpy`:

```bash
python tools/standalone_sim.py             # interactive viewer, same key bindings
python tools/standalone_sim.py --headless   # self-test: stand, trot, turn
```

Current self-test result:

```
model ok: 19 dof, 12 actuators, mass 2.098 kg
  stand    z= 199.4 mm  roll= +0.0 pitch= +0.0
  hold 1s  z= 170.1 mm  roll= +0.0 pitch= +0.2  drift=  9.9 mm
  trot 5s  z= 169.7 mm  roll= +2.3 pitch= -0.2  travelled x= 787.9 mm  y=  -9.8 mm
  turn 4s  z= 169.5 mm  roll= +0.2 pitch= +0.0
RESULT: OK — stands and trots forward
```

0.16 m/s against a 0.20 m/s command, 6 mm lateral drift over 5 s, attitude within 1.3°.

## Regenerating the model from CAD

`smalldog_description` is **not hand-written**. Everything comes out of the CAD:

```bash
../3d/.venv/bin/python smalldog_description/scripts/generate_model.py
```

It imports `../3d/mini_dog.py`, exports one STL per link in that link's own frame,
computes real mass properties from the tessellated solids (printed parts at PETG ×40 %
infill, feet in TPU, plus servo / battery / Orange Pi / LiDAR point masses), and writes
`urdf/smalldog.urdf`, `mujoco/robot.xml`, `mujoco/scene.xml`, `mujoco/defaults.xml`
and `robot_params.json`. Change a dimension in the CAD, re-run, rebuild.

**Every density and point mass comes from `mini_dog.py` section 4** — this script keeps no
copies. It used to, and its servo mass drifted to 60 g while the CAD side said 55 g, so
the two models of the same robot differed by 60 g. If you need to change one, change it in
the CAD.

This is also not the only exporter of that CAD: `../3d/export_sim.py` writes its own
URDF/MJCF into `3d/out/sim/` with a different link decomposition (the foot is a separate
part there, merged into the shin here). Both read the same `mini_dog.py` and both have to
be re-run after a model change; `3d/CLAUDE.md` step 6 is the checklist. As of the last
run both report the same 2.098 kg.

`robot_params.json` is the single source the gait reads at runtime — link lengths,
hip offsets, joint limits and the nominal stance all come from there, so the walker
can never drift out of sync with the mechanics.

## Model facts

| | |
|---|---|
| total mass | 2.098 kg (base 1.363 kg incl. battery, Orange Pi, LiDAR) |
| leg reach | 102…152 mm from the hip-pitch axis → usable body height 154…170 mm |
| joints | `{fl,fr,rl,rr}_{roll,pitch,knee}` — 12 servo IDs 1…12 in that order |
| joint limits | roll ±0.90, pitch ±1.30, knee ±1.85 rad — from the CAD interference scan |
| MuJoCo hard stops | 0.03 rad **inside** the URDF limits, so the measured position can never trip ros2_control's joint limiter |
| gait soft limits | 0.12 rad inside the mechanical limits |
| joint effort / velocity | 3.0 N·m / 4.7 rad/s (ST3215 stall + 0.222 s per 60°) |
| nominal stance | base 181 mm above ground, gait default 158 mm |
| meshes | 13 link meshes + 12 ST3215 bodies (visual only — their mass is already in each link's `<inertial>`, so they are drawn, never weighed twice) |
| links | `base_link` + `{leg}_hip` / `{leg}_thigh` / `{leg}_shin` (foot fused into shin) |

The 12 servos are drawn in place (`{leg}_roll_servo.stl` etc., dark) so the model reads as
the real machine. They carry no geometry mass: each link's `<inertial>` already accounts
for its servo as a 60 g uniform box at the case centre, and the MJCF's explicit `<inertial>`
overrides any geom-derived mass. `sum(model.body_mass)` matches `robot_params.json` exactly.

Collision model is deliberately simple: boxes/capsules for the links, a sphere per foot
with its own friction. Feet and body collide with the ground; **self-collision is off** —
joint limits already come from the CAD sweep, so the sim does not need to re-discover them.

## Gait

`smalldog_walker/gait.py` — trot, diagonal pairs `(FL,RR)` / `(FR,RL)`, duty 0.5.

- `period` 0.45 s, `swing_height` 22 mm, `max_step` ±60 mm fore/aft, ±30 mm lateral
- **the leg is short**: 75 + 82 mm with a ±1.85 rad knee gives only ~50 mm of vertical foot
  travel, so the gait cannot lift the foot much. `body_height` is clamped at runtime to the
  band where stance *and* swing apex both stay inside that reach; ask for more and the
  setter shrinks the swing instead of producing an unreachable target
- yaw is folded in as `v + ω × r_hip`, so turning and translating compose
- a `_moving` blend keeps the feet planted when the command drops to zero, instead of
  freezing mid-swing
- joint outputs are clamped to the soft limits, then rate-limited to 0.85 × the servo's
  4.7 rad/s, so start-up ramps into stance instead of stepping there

`leg_kinematics.py` is a closed-form solver, not numeric: the roll angle comes from the
requirement that the foot lands in the thigh/shin plane (`py·cos q1 + pz·sin q1 = dy`),
then a plain 2-link solve inside that plane. FK/IK round-trip is tested to 1e-6.

```bash
python -m pytest smalldog_walker/test -q     # 7 passed
```

## Verified run

Built and run end-to-end in the spider project's `pixi` `kilted` environment on macOS
(`colcon build` → `ros2 launch` → `ros2 run smalldog_teleop keyboard`):

```
walker up: 12 joints -> /smalldog_controller/joint_trajectory @ 100 Hz
leg reach 102..152 mm -> body height 154..170 mm, using 158 mm, swing 22 mm

keyboard -> /cmd_vel -> walker -> controller -> MuJoCo joints
  idle (no key)                max joint swing 0.010 rad
  after 'w'                    max joint swing 0.376 rad
  after space                  max joint swing 0.011 rad
controllers: both active, 0 deactivations
```

Keys were confirmed individually against `/cmd_vel`:
`w`→vx +0.20, `d`→vy −0.20, `e`→wz −1.20, space→0, `.`→speed 0.20→0.25, `r`→height 158→162 mm.

Four real defects were found and fixed by running it, not by reading it:

1. `robot_description` must be wrapped in `ParameterValue(..., value_type=str)` on Kilted —
   the launch aborted immediately without it.
2. The **controller deactivated itself mid-run**: `fl_knee` physically reached 1.7599 rad
   against a 1.75 rad URDF limit and `joint_limiter` threw. Fixed structurally — MuJoCo's
   own joint ranges now sit 0.03 rad inside the URDF limits, so the measured state can
   never violate them.
3. The gait was asking for a 45 mm swing the leg cannot reach; the knee spent the swing
   phase pinned at its clamp. `body_height` is now validated against the real reach band.
4. `KeyboardTeleop.handle()` shadowed rclpy's `Node.handle` property — the teleop node
   crashed on construction and had never run. Also fixed its shutdown race (segfault on
   Ctrl-C) by owning the executor instead of `rclpy.spin` in a daemon thread.

**Real-time factor ≈ 6.2×** on this machine: `mujoco_ros2_control` does not throttle to
wall clock, so a 0.45 s gait period plays back in ~0.07 s and the robot looks like it is
sprinting. Command freshness in the walker is therefore measured on the **wall** clock
while the gait phase integrates in **sim** time — mixing them makes a steady 20 Hz teleop
look stale on the sim clock.

## Known gaps

- No hardware interface yet. The real robot needs an ST3215 bus node (ESP32 + URT-1);
  `smalldog_ros_control` currently only wires up the MuJoCo system, mirroring
  `spider-mujoco.urdf.xacro`. The serial half is the spider's `spider_hardware_interface.cpp`
  equivalent and is not written.
- Open-loop gait: the IMU and foot-contact sensors are published by the MuJoCo model
  (`imu_quat`, `imu_gyro`, `imu_acc`, `{leg}_contact`) but nothing consumes them yet —
  no balance controller, no odometry.
- The launch file's `teleop:=true` path shells out to `xterm`; not tested. On macOS run
  the teleop node in a second terminal instead — that is the path that was verified.
- Nothing throttles the sim to real time; see the real-time factor note above.
- Stale nodes from a killed launch (`robot_state_publisher` especially) block the next
  `controller_manager` from coming up. `pkill -f robot_state_publisher` before relaunching.
- `mujoco_ros2_control` gains in `smalldog-mujoco.urdf.xacro` (kp 120 / kd 3) are a
  starting point; the fork drives MuJoCo's own position actuators
  (`defaults.xml`: kp 25, dampratio 1), so tune there first.
