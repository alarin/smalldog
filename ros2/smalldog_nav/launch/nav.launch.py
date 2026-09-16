"""SLAM + Nav2 (+ the explorer) on top of a running walker, sim or robot.

    ros2 launch smalldog_nav nav.launch.py                        # sim: cloud from MuJoCo
    ros2 launch smalldog_nav nav.launch.py explore:=true          # ... and walk the room out
    ros2 launch smalldog_nav nav.launch.py use_sim_time:=false cloud:=/lidar/points speed:=0.12 \
                                           explore:=true autostart:=false
                          # the robot (robot.launch.py imu:=true lidar:=true joy:=true): Back starts it
    ros2 launch smalldog_nav nav.launch.py explore:=true save_map:=/tmp/room

The chain:  <cloud> --cloud_to_scan--> /scan --slam_toolbox--> /map, map->odom
            /map + /scan --Nav2 (planner, RPP controller, behaviors, BT)--> /cmd_vel
            /map --explore--> navigate_to_pose goals

What it needs from below: the walker's `/odom` and the TF pair odom -> base_footprint
-> base_link (walker_node.py, on by default), robot_state_publisher for base_link ->
lidar_link, and the IMU live at the walker (the level frame is the IMU's roll and pitch;
without it base_footprint is base_link lifted, and the floor leaks into the scan on a
slope). The walker takes /cmd_vel from Nav2 exactly as it does from the teleop, so the
teleop must be off or silent (sim.sh teleop:=false) or the two fight.

The scan is a slice of the cloud between 0.10 and 0.35 m above the floor, in the level
frame: below the L2's own 0.25 m axis and above what the trot's residual tilt puts the
floor at out to 6 m. The L2's cone is +-96 deg about +x, so the scan is that wide and no
wider - the missing rear must not read as "clear to max range" (an empty bin is inf,
which the costmap and slam_toolbox both drop).
"""
import math
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

# the L2's cone, 3d/mini_dog.py LIDAR_FOV_NEGA: 96 deg either side of the axis
HALF_FOV = math.radians(96.0)


def _nodes(context):
    share = get_package_share_directory('smalldog_nav')
    sim = LaunchConfiguration('use_sim_time').perform(context).lower() in ('true', '1')
    speed = float(LaunchConfiguration('speed').perform(context))
    cloud = LaunchConfiguration('cloud').perform(context)
    common = {'use_sim_time': sim}
    nav2_yaml = os.path.join(share, 'config', 'nav2.yaml')
    slam_yaml = os.path.join(share, 'config', 'slam.yaml')
    lifecycle = ['controller_server', 'planner_server', 'behavior_server', 'bt_navigator']

    return [
        Node(package='smalldog_nav', executable='cloud_to_scan', name='cloud_to_scan',
             output='screen', remappings=[('cloud', cloud)],
             parameters=[common, {
                 'target_frame': 'base_footprint',
                 'min_height': 0.10, 'max_height': 0.35,
                 'angle_min': -HALF_FOV, 'angle_max': HALF_FOV,
                 'angle_increment': math.radians(0.5),
                 'scan_time': 1.0 / 12.0,
                 'range_min': 0.30, 'range_max': 6.0,
                 # a return below the floor is a drop, walled at its edge (the stairwell)
                 'drop_depth': 0.12, 'drop_range': 2.0}]),

        # slam_toolbox is a lifecycle node in Kilted and comes up unconfigured - no scan
        # subscription, no map - until something transitions it. Its own manager, with
        # no bond, ahead of Nav2's: the planner's global costmap waits for the `map`
        # frame at activation and gives up after a minute without it.
        Node(package='slam_toolbox', executable='async_slam_toolbox_node',
             name='slam_toolbox', output='screen',
             parameters=[slam_yaml, common, {'use_lifecycle_manager': True}]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_slam', output='screen',
             parameters=[common, {'autostart': True, 'node_names': ['slam_toolbox']}]),

        Node(package='nav2_controller', executable='controller_server', output='screen',
             parameters=[nav2_yaml, common,
                         {'FollowPath.desired_linear_vel': speed}],
             remappings=[('cmd_vel', '/cmd_vel')]),
        Node(package='nav2_planner', executable='planner_server', output='screen',
             parameters=[nav2_yaml, common]),
        Node(package='nav2_behaviors', executable='behavior_server', output='screen',
             parameters=[nav2_yaml, common],
             remappings=[('cmd_vel', '/cmd_vel')]),
        Node(package='nav2_bt_navigator', executable='bt_navigator', output='screen',
             parameters=[nav2_yaml, common]),
        Node(package='nav2_lifecycle_manager', executable='lifecycle_manager',
             name='lifecycle_manager_navigation', output='screen',
             parameters=[common, {'autostart': True, 'node_names': lifecycle,
                                  'bond_timeout': 10.0}]),

        Node(package='smalldog_nav', executable='explore', name='smalldog_explore',
             output='screen', condition=IfCondition(LaunchConfiguration('explore')),
             parameters=[common, {
                 'save_map': ParameterValue(LaunchConfiguration('save_map'), value_type=str),
                 'home': ParameterValue(LaunchConfiguration('home'), value_type=bool),
                 'autostart': ParameterValue(LaunchConfiguration('autostart'), value_type=bool),
                 'spin_first': ParameterValue(LaunchConfiguration('spin_first'), value_type=bool)}]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('use_sim_time', default_value='true',
                              description='true under the MuJoCo launch, false on the robot'),
        DeclareLaunchArgument('cloud', default_value='/mujoco_ros2_control_node/lidar/points',
                              description='the L2 PointCloud2, in lidar_link; the robot '
                                          'publishes /lidar/points (robot.launch.py lidar:=true)'),
        DeclareLaunchArgument('speed', default_value='0.15',
                              description='m/s Nav2 asks for on a straight; 0.08 on the '
                                          'robot with the heading hold, 0.11 blind '
                                          '(robot.launch.py)'),
        DeclareLaunchArgument('explore', default_value='false',
                              description='walk the frontiers until the map is closed'),
        DeclareLaunchArgument('save_map', default_value='',
                              description='path stem the explorer saves the finished map '
                                          'to (slam_toolbox/save_map -> stem.pgm + .yaml); '
                                          'empty = do not'),
        DeclareLaunchArgument('home', default_value='true',
                              description='walk back to where SLAM started when explored'),
        DeclareLaunchArgument('autostart', default_value='true',
                              description='explore as soon as the map is there; false = '
                                          'wait for /smalldog/explore true, i.e. the '
                                          'gamepad\'s Back button (robot.launch.py joy:=true)'),
        DeclareLaunchArgument('spin_first', default_value='true',
                              description='one full turn before the first goal, to see '
                                          'the ring. false on the robot: its turn in place '
                                          'runs at ~0.09 rad/s for a 0.5 command and '
                                          'drifts backwards, so a 2 pi spin is a minute '
                                          'of walking backwards'),
        OpaqueFunction(function=_nodes),
    ])
