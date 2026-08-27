#!/bin/bash
# Activate ROS 2 (pixi "kilted" env borrowed from the spider project) + both workspaces.
# Source this, do not execute it:   source tools/env.sh
#
# The setup file has to match the shell: ROS 2's local_setup.bash resolves its own path
# through $BASH_SOURCE, which zsh does not set, so under zsh it silently looks in $PWD
# and the workspace overlay never gets applied.

PIXI_MANIFEST=/Users/alarin/Documents/art/ogonek25-spider/ros2/pixi-robostack/pixi.toml
SPIDER_WS=/Users/alarin/Documents/art/ogonek25-spider/ros2
DOG_WS=/Users/alarin/Documents/art/smalldog/ros2

if [ -n "$ZSH_VERSION" ]; then _sd_ext=zsh; else _sd_ext=bash; fi

eval "$(pixi shell-hook --manifest-path "$PIXI_MANIFEST" -e kilted)"
source "$SPIDER_WS/install/local_setup.$_sd_ext"     # provides mujoco_ros2_control
source "$DOG_WS/install/local_setup.$_sd_ext"
unset _sd_ext
