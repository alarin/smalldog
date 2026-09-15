#!/bin/bash
# The robot maps the walled room by itself: MuJoCo + SLAM + Nav2 + the frontier explorer.
#   ./tools/explore.sh                     # watch on Foxglove, ws://localhost:8765, frame map
#   ./tools/explore.sh rtf:=2.0 save_map:=/tmp/room
# Arguments pass through to smalldog_nav/launch/sim_explore.launch.py.
source "$(dirname "$0")/env.sh"
pkill -f robot_state_publisher 2>/dev/null   # stale nodes block the controller_manager
sleep 1
exec ros2 launch smalldog_nav sim_explore.launch.py "$@"
