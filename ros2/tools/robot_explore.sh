#!/bin/bash
# The real robot maps the room by itself: robot.launch.py (walker, IMU, L2) then
# nav.launch.py with SLAM + Nav2 + the frontier explorer.
#
#     tools/robot_explore.sh [robot.launch args...]        # e.g. mode2:=false speed:=0.08
#     SPEED=0.08 tools/robot_explore.sh
#     tools/robot_explore.sh stop                          # Ctrl-C both launches
#
# Run on the Pi from ros2/ with /opt/ros/jazzy and install/ sourced, detached
# (`setsid nohup tools/robot_explore.sh ... &`); logs in ~/smalldog_logs/explore_<stamp>/,
# with a bag of the scan, the map, TF, the commands and the raw L2 cloud (~1 MB/s), so a
# run can be replayed: `ros2 bag play bag/ --clock` under `ros2 launch smalldog_nav
# nav.launch.py use_sim_time:=true cloud:=/lidar/points` replays the drop detection.
# The trot walks under the firmware loop only (README "On the robot"), so mode2:=false is
# the default here; the servo node's black box goes to ~/smalldog_logs as usual.
HERE=$(cd "$(dirname "$0")/.." && pwd)
PIDF=$HOME/smalldog_logs/explore.pids
if [ "$1" = "stop" ]; then
  [ -f "$PIDF" ] || { echo "nothing running"; exit 0; }
  for p in $(tac "$PIDF"); do kill -INT -- -"$p" 2>/dev/null; done
  for i in $(seq 1 40); do alive=0; for p in $(cat "$PIDF"); do kill -0 "$p" 2>/dev/null && alive=1; done; [ $alive = 0 ] && break; sleep 0.5; done
  for p in $(cat "$PIDF"); do kill -0 "$p" 2>/dev/null && kill -TERM -- -"$p"; done
  rm -f "$PIDF"; echo "stopped"; exit 0
fi
SPEED=${SPEED:-0.08}
# and it listens: «псина, стоп» / «псина, гуляй» pause and resume the explorer (the ears
# node, robot/sound/ears.py, ~30 cm range). VOICE=0 keeps it deaf
VOICE_ARG=voice:=$([ "${VOICE:-1}" != 0 ] && echo true || echo false)
# the L2 feed (3d/tools/stream_pcd.cpp) is a plain process, not a service: without it the
# lidar node dies at start and the explorer stands waiting for a cloud that never comes
STREAM_PCD=${STREAM_PCD:-$HOME/unilidar_sdk2/unitree_lidar_sdk/bin/stream_pcd}
if ! ss -ltn | grep -q ':9910 '; then
  echo "starting stream_pcd"
  (setsid nohup "$STREAM_PCD" 9910 192.168.1.62 192.168.1.2 > "$HOME/stream_pcd.log" 2>&1 < /dev/null &)
  sleep 3
fi
STAMP=$(date +%Y%m%d_%H%M%S)
OUT=$HOME/smalldog_logs/explore_$STAMP; mkdir -p "$OUT"; echo "$OUT" > "$HOME/smalldog_logs/explore.latest"
setsid ros2 launch smalldog_hardware robot.launch.py mode2:=false imu:=true lidar:=true lidar_idle_stop:=0 $VOICE_ARG "$@" > "$OUT/robot.log" 2>&1 &
echo $! > "$PIDF"
for i in $(seq 1 80); do grep -q "streaming the walker" "$OUT/robot.log" && break; sleep 0.5; done
grep -q "streaming the walker" "$OUT/robot.log" || { echo "!! the robot did not stand up; see $OUT/robot.log"; "$0" stop; exit 1; }
echo "standing; starting SLAM + Nav2 + explorer at $SPEED m/s"
setsid ros2 launch smalldog_nav nav.launch.py use_sim_time:=false cloud:=/lidar/points speed:=$SPEED explore:=true > "$OUT/nav.log" 2>&1 &
echo $! >> "$PIDF"
setsid ros2 bag record -s mcap -o "$OUT/bag" /scan /map /tf /tf_static /cmd_vel /odom /imu /lidar/points \
    /smalldog/foot_load /joint_states /rosout > "$OUT/bag.log" 2>&1 &
echo $! >> "$PIDF"
echo "logs: $OUT"
# a dog that explores a house barks now and then: a random cut of the recording every
# 40-160 s (robot/sound/bark), for as long as the launches run. BARKS=0 keeps it quiet
if [ "${BARKS:-1}" != 0 ] && command -v bark >/dev/null; then
  setsid bash -c 'while true; do sleep $((40 + RANDOM % 121)); bark bark $((RANDOM % 3 + 1)).$((RANDOM % 10)) >/dev/null 2>&1; done' &
  echo $! >> "$PIDF"
fi
wait
