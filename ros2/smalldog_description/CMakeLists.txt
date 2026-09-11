cmake_minimum_required(VERSION 3.16)
project(smalldog_description)

find_package(ament_cmake REQUIRED)

# No launch/ any more: view_robot.launch.py and smalldog.rviz were the only things in it
# and both were RViz, which this workspace deliberately does not have - see README.md,
# "Looking at it": foxglove_bridge is the viewer, and rviz2 is not in the pixi env.
install(
  DIRECTORY urdf mujoco meshes
  DESTINATION share/${PROJECT_NAME}
)
install(
  FILES robot_params.json
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
