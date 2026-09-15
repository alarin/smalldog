#!/bin/bash
# Walk the real robot straight under ROS 2 without a keyboard: robot.launch.py, then
# /cmd_vel for S seconds, then Ctrl-C (the node sits down and cuts torque).
#
#     tools/robot_go.sh [S=4] [VX=0.08] [launch args...]
#
# Run on the Pi from ros2/ with /opt/ros/jazzy and install/ sourced. The launch is
# started in its own session so the Ctrl-C reaches every node the way a terminal's would.
S=${1:-4}; VX=${2:-0.08}; shift 2 2>/dev/null
LOG=${LOG:-/tmp/robot_go.log}
setsid ros2 launch smalldog_hardware robot.launch.py "$@" > "$LOG" 2>&1 &
LP=$!
for i in $(seq 1 60); do grep -q "streaming the walker" "$LOG" && break; sleep 0.5; done
if ! grep -q "streaming the walker" "$LOG"; then
  echo "!! the robot did not stand up; see $LOG"; kill -INT -- -$LP; exit 1
fi
sleep 1
timeout "$S" ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: $VX}}" > /dev/null 2>&1
sleep 1.5
kill -INT -- -$LP
for i in $(seq 1 30); do kill -0 $LP 2>/dev/null || break; sleep 0.5; done
kill -0 $LP 2>/dev/null && { echo "!! launch still alive, SIGTERM"; kill -TERM -- -$LP; }
grep -v "affinity\|^\[INFO\] \[launch\]\|Traceback\|File \|\^\^\|rclpy\|wait_set\|handler, entity\|return next\|executor.spin\|self._spin" "$LOG"
