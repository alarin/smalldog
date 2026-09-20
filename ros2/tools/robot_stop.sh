#!/bin/bash
# robot_stop.sh — stop everything of ours on the robot: the launches, and the nodes a
# launch leaves behind (the walker outlives its launch's Ctrl-C; the servo node then
# refuses to arm: "2 publishers on /smalldog_controller/joint_trajectory"). INT first so
# the servo node sits the robot down and writes its tick log, TERM after 6 s.
#     tools/robot_stop.sh
me=$$
pids() { pgrep -f 'ros2 launch smalldog|smalldog_walker/lib|smalldog_hardware/lib|straight_test.py|tools/moves.sh|robot_explore.sh|ros2 bag record|smalldog_nav/lib|nav2_|slam_toolbox' | grep -vx "$me"; }
p=$(pids); [ -z "$p" ] && { echo "nothing running"; exit 0; }
kill -INT $p 2>/dev/null
for i in $(seq 1 12); do [ -z "$(pids)" ] && break; sleep 0.5; done
p=$(pids); [ -n "$p" ] && { kill -TERM $p 2>/dev/null; sleep 1; }
p=$(pids); [ -n "$p" ] && { kill -KILL $p 2>/dev/null; }
echo "stopped"
