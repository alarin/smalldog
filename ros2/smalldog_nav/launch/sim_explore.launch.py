"""One launch for the whole thing in the sim: MuJoCo in the walled room, SLAM, Nav2 and
the explorer, with a Foxglove bridge to watch the map grow.

    ros2 launch smalldog_nav sim_explore.launch.py                # or tools/explore.sh
    ros2 launch smalldog_nav sim_explore.launch.py rtf:=2.0 save_map:=/tmp/room

Foxglove: ws://localhost:8765, a 3D panel, frame `map`, topics /map, /scan,
/global_costmap/costmap, /plan. The MuJoCo window shows the robot; keys typed over it
are NOT read (teleop:=false here, so Nav2 owns /cmd_vel).
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description():
    sim = os.path.join(get_package_share_directory('smalldog_ros_control'), 'launch',
                       'smalldog-mujoco.launch.py')
    nav = os.path.join(get_package_share_directory('smalldog_nav'), 'launch', 'nav.launch.py')
    return LaunchDescription([
        DeclareLaunchArgument('rtf', default_value='1.0'),
        DeclareLaunchArgument('speed', default_value='0.15'),
        DeclareLaunchArgument('explore', default_value='true'),
        DeclareLaunchArgument('save_map', default_value=''),
        DeclareLaunchArgument('home', default_value='true'),
        DeclareLaunchArgument('autostart', default_value='true'),
        DeclareLaunchArgument('foxglove', default_value='true'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(sim), launch_arguments={
            'room': 'true', 'teleop': 'false', 'imu': 'true',
            'foxglove': LaunchConfiguration('foxglove'),
            'rtf': LaunchConfiguration('rtf')}.items()),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(nav), launch_arguments={
            'use_sim_time': 'true',
            'speed': LaunchConfiguration('speed'),
            'explore': LaunchConfiguration('explore'),
            'save_map': LaunchConfiguration('save_map'),
            'home': LaunchConfiguration('home'),
            'autostart': LaunchConfiguration('autostart')}.items()),
    ])
