"""The trot on the real robot, under ROS 2: servos + walker (+ teleop).

    ros2 launch smalldog_hardware robot.launch.py                  # stand, wait for /cmd_vel
    ros2 launch smalldog_hardware robot.launch.py imu:=true        # levelling + heading hold
    ros2 launch smalldog_hardware robot.launch.py dry_run:=true    # loopback bus, no hardware
    ros2 run smalldog_teleop keyboard --ros-args -p speed:=0.08 -p turn:=0.65   # 2nd terminal

Or without a keyboard:

    ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.08}}"

The gait numbers are an operating point fitted to the servo's 3.28 rad/s ceiling
(`robot/runtime/walk.py --dry-run` prints the fit; `joint_rate_demand` there is the
table), not the sim's defaults. The sim's 0.20 m/s at period 0.45 s demands 7.55 rad/s
and drags the feet (`robot/README.md`, "The gait is fitted to the servo"). Two points
are measured on the floor (2026-09-15):

    blind (imu:=false)   0.11 m/s, period 1.35 s        demand 3.09 rad/s, walks
    imu:=true            0.08 m/s, period 1.35 s, the heading hold capped at 0.2 rad/s
                         demand 3.04 rad/s, walks, 44 deg peak tracking error; at
                         0.11 m/s with the gait's own 0.5 rad/s cap it is 4.65, a knee
                         fell 0.7 rad behind and the guard tripped

`speed` here sets the teleop's speed and the gait's `stride_max` (so `period_for` admits
the period); it does not cap `/cmd_vel` from elsewhere — publish the same number.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

PERIOD, TURN = 1.35, 0.65


def _nodes(context):
    speed = float(LaunchConfiguration('speed').perform(context))
    yaw_max = float(LaunchConfiguration('yaw_max').perform(context))
    return [
        Node(package='smalldog_hardware', executable='servos', name='smalldog_servos',
             output='screen',
             parameters=[{'port': LaunchConfiguration('port'),
                          'imu': LaunchConfiguration('imu'),
                          'dry_run': LaunchConfiguration('dry_run')}]),

        Node(package='smalldog_walker', executable='walker', name='smalldog_walker',
             output='screen',
             parameters=[{'use_sim_time': False,
                          'rate': 50.0,
                          'imu_topic': '/imu',
                          'period': PERIOD,
                          'stride_max': speed * PERIOD / 2.0,
                          'yaw_max': yaw_max}]),

        Node(package='smalldog_teleop', executable='keyboard', name='smalldog_keyboard_teleop',
             output='screen', condition=IfCondition(LaunchConfiguration('teleop')),
             parameters=[{'speed': speed, 'turn': TURN, 'read_stdin': False,
                          'key_topic': '/smalldog/key'}]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('imu', default_value='false'),
        DeclareLaunchArgument('dry_run', default_value='false'),
        DeclareLaunchArgument('speed', default_value='0.08',
                              description='m/s; 0.11 fits blind, 0.08 leaves room for the heading hold'),
        DeclareLaunchArgument('yaw_max', default_value='0.2',
                              description='rad/s the heading hold may add; 0 = the gait\'s own 0.5'),
        # off by default: the teleop reads keys from a TTY it does not have under launch
        DeclareLaunchArgument('teleop', default_value='false'),
        OpaqueFunction(function=_nodes),
    ])
