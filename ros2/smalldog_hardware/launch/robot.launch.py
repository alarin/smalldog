"""The trot on the real robot, under ROS 2: servos + walker (+ teleop).

    ros2 launch smalldog_hardware robot.launch.py                  # stand, wait for /cmd_vel
    ros2 launch smalldog_hardware robot.launch.py imu:=true        # levelling + heading hold
    ros2 launch smalldog_hardware robot.launch.py dry_run:=true    # loopback bus, no hardware
    ros2 run smalldog_teleop keyboard --ros-args -p speed:=0.11 -p turn:=0.65   # 2nd terminal

Or without a keyboard:

    ros2 topic pub -r 20 /cmd_vel geometry_msgs/msg/Twist "{linear: {x: 0.11}}"

The gait numbers are the operating point `robot/runtime/walk.py --dry-run` fits to the
servo's 3.28 rad/s ceiling, not the sim's defaults: 0.11 m/s at period 1.35 s, 0.65 rad/s
of turn on the spot, with `stride_max` raised so the gait's own schedule admits the
period. Ask for the sim's 0.20 m/s / 0.45 s here and the feet drag
(`robot/README.md`, "The gait is fitted to the servo").
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

SPEED, PERIOD, TURN = 0.11, 1.35, 0.65


def generate_launch_description():
    port = LaunchConfiguration('port')
    imu = LaunchConfiguration('imu')
    dry = LaunchConfiguration('dry_run')
    teleop = LaunchConfiguration('teleop')
    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('imu', default_value='false'),
        DeclareLaunchArgument('dry_run', default_value='false'),
        # off by default: the teleop reads keys from a TTY it does not have under launch
        DeclareLaunchArgument('teleop', default_value='false'),

        Node(package='smalldog_hardware', executable='servos', name='smalldog_servos',
             output='screen',
             parameters=[{'port': port, 'imu': imu, 'dry_run': dry}]),

        Node(package='smalldog_walker', executable='walker', name='smalldog_walker',
             output='screen',
             parameters=[{'use_sim_time': False,
                          'rate': 50.0,
                          'imu_topic': '/imu',
                          'period': PERIOD,
                          'stride_max': SPEED * PERIOD / 2.0}]),

        Node(package='smalldog_teleop', executable='keyboard', name='smalldog_keyboard_teleop',
             output='screen', condition=IfCondition(teleop),
             parameters=[{'speed': SPEED, 'turn': TURN, 'read_stdin': False,
                          'key_topic': '/smalldog/key'}]),
    ])
