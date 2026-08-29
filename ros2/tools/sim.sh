#!/bin/bash
# MuJoCo + ros2_control + gait node.  Launch arguments pass straight through:
#   ./tools/sim.sh terrain:=true      # rough ground instead of the flat plane
#   ./tools/sim.sh teleop:=false      # no keyboard node; drive it from tools/teleop.sh
#
# Keyboard control needs no second terminal: click the MuJoCo window and type (w/a/s/d,
# space, r/f, ,/., t).  The sim node publishes what the viewer sees on ~/key and the
# teleop node spawned by the launch turns it into /cmd_vel.
source "$(dirname "$0")/env.sh"
pkill -f robot_state_publisher 2>/dev/null   # stale nodes block the controller_manager
sleep 1
exec ros2 launch smalldog_ros_control smalldog-mujoco.launch.py "$@"
