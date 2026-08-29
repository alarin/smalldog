#!/bin/bash
# Keyboard teleop in its own terminal — it reads raw stdin, so it needs a real TTY.
# Only needed for `sim.sh teleop:=false`: by default the launch spawns this node itself and
# it takes its keys from the MuJoCo window.  Running both makes two publishers on /cmd_vel.
source "$(dirname "$0")/env.sh"
echo "waiting for the walker to appear on /cmd_vel ..."
for i in $(seq 1 20); do
  ros2 node list 2>/dev/null | grep -q smalldog_walker && break
  sleep 1
done
exec ros2 run smalldog_teleop keyboard
