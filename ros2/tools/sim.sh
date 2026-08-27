#!/bin/bash
# MuJoCo + ros2_control + gait node.
source "$(dirname "$0")/env.sh"
pkill -f robot_state_publisher 2>/dev/null   # stale nodes block the controller_manager
sleep 1
exec ros2 launch smalldog_ros_control smalldog-mujoco.launch.py
