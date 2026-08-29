#!/bin/bash
# Activate this workspace's own ROS 2 environment.  Source it, do not execute it:
#
#     source tools/env.sh
#
# The pixi env lives in ros2/pixi (see pixi/pixi.toml).  It used to be borrowed from
# ../../ogonek25-spider, which meant smalldog could not run without that unrelated
# project checked out beside it, and it shipped whatever controller set the spider
# happened to need - imu_sensor_broadcaster was not among them, so the gait ran blind.
#
# First time, or after pulling a change to pixi.toml:
#     pixi install --manifest-path ros2/pixi/pixi.toml     # ~2.3 GB
#     git submodule update --init                          # src/mujoco_ros2_control
#     source tools/env.sh && colcon build --symlink-install
#
# The setup file has to match the shell: ROS 2's local_setup.bash resolves its own path
# through $BASH_SOURCE, which zsh does not set, so under zsh it silently looks in $PWD
# and the workspace overlay never gets applied.

_sd_ws="$(cd "$(dirname "${BASH_SOURCE[0]:-${(%):-%x}}")/.." && pwd)"

if [ -n "$ZSH_VERSION" ]; then _sd_ext=zsh; else _sd_ext=bash; fi

eval "$(pixi shell-hook --manifest-path "$_sd_ws/pixi/pixi.toml")"

if [ -f "$_sd_ws/install/local_setup.$_sd_ext" ]; then
  source "$_sd_ws/install/local_setup.$_sd_ext"
else
  echo "tools/env.sh: no install/ yet - run 'colcon build --symlink-install' in $_sd_ws" >&2
fi
unset _sd_ext _sd_ws
