cmake_minimum_required(VERSION 3.16)
project(smalldog_description)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY launch urdf mujoco meshes
  DESTINATION share/${PROJECT_NAME}
)
install(
  FILES robot_params.json smalldog.rviz
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
