"""The trot on the real robot, under ROS 2: servos + walker (+ teleop).

    ros2 launch smalldog_hardware robot.launch.py                  # stand, wait for /cmd_vel
    ros2 launch smalldog_hardware robot.launch.py imu:=true        # levelling + heading hold
    ros2 launch smalldog_hardware robot.launch.py joy:=true        # + the gamepad (joy_node + smalldog_teleop joy)
    ros2 launch smalldog_hardware robot.launch.py dry_run:=true    # loopback bus, no hardware
    ros2 launch smalldog_hardware robot.launch.py imu:=true lidar:=true   # + the L2 on /lidar/points,
                                                                   # for smalldog_nav (stream_pcd running)
    ros2 launch smalldog_hardware robot.launch.py imu:=true camera:=true foxglove:=true
                                                                   # + the IMX415 on /camera/image/compressed
                                                                   # and a Foxglove bridge on ws://<pi>:8765
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
import os

from ament_index_python.packages import get_package_share_directory
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue

PERIOD, TURN = 1.35, 0.65


def _nodes(context):
    speed = float(LaunchConfiguration('speed').perform(context))
    yaw_max = float(LaunchConfiguration('yaw_max').perform(context))
    urdf = os.path.join(get_package_share_directory('smalldog_description'), 'urdf',
                        'smalldog.urdf')
    with open(urdf) as f:
        robot_description = f.read()
    return [
        # TF for the body: base_link -> the legs (from the servo node's /joint_states) and
        # the fixed sensor frames. smalldog_nav needs lidar_link; without this node it
        # does not exist on the robot and the scan cannot be built.
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             output='both',
             parameters=[{'robot_description': ParameterValue(robot_description,
                                                              value_type=str)}]),

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
                          # and never shorter: `period_for` counts a turn as speed and
                          # cut the cycle to 0.86 s at Nav2's 0.5 rad/s, where the
                          # servos lag a foot into the wrong half of the stride and the
                          # turn's direction is a coin toss (+0.5 read -36 and +30 deg,
                          # -0.5 read +34, 2026-09-16); 0.3 at 1.35 s reads +46
                          'period_min': PERIOD,
                          'yaw_max': yaw_max,
                          # the floor veers the blind trot ~7 deg/s; a P hold sat 4 deg
                          # off against that on three straight runs (2026-09-15)
                          'yaw_ki': 0.5,
                          # every /cmd_vel scaled to the servo budget (walker_node.py,
                          # fit_cmd): Nav2 asks forward + turn together
                          'fit_cmd': True,
                          # levelling off until the corrected leg map (calib.json,
                          # 2026-09-15) has read a positive turn on +wz on the floor;
                          # the roll loop diverged on the mirrored map (walker_node.py,
                          # level_kp). The IMU still feeds the heading hold and the
                          # level frame for smalldog_nav
                          'level_kp': 0.0,
                          # three 7 s walks measured by the LiDAR scan match
                          # (tools/straight_test.py, 2026-09-15): 0.38 / 0.41 / 0.41 m
                          # real against 0.55 / 0.55 / 0.55 m of stance-foot travel
                          # -> 0.68-0.75 with the P hold, 0.78-0.82 once the
                          # integral term stopped it turning; the map-based 0.5 was one walk
                          'odom_scale': 0.75}]),

        Node(package='smalldog_teleop', executable='keyboard', name='smalldog_keyboard_teleop',
             output='screen', condition=IfCondition(LaunchConfiguration('teleop')),
             parameters=[{'speed': speed, 'turn': TURN, 'read_stdin': False,
                          'key_topic': '/smalldog/key'}]),

        # the gamepad: joy_node owns /dev/input/js0, the mapping node turns it into /cmd_vel
        Node(package='joy', executable='joy_node', name='joy_node', output='screen',
             condition=IfCondition(LaunchConfiguration('joy')),
             parameters=[{'device_id': 0, 'deadzone': 0.05, 'autorepeat_rate': 20.0}]),
        Node(package='smalldog_teleop', executable='joy', name='smalldog_joy_teleop',
             output='screen', condition=IfCondition(LaunchConfiguration('joy')),
             parameters=[{'speed': speed, 'turn': TURN}]),

        # the L2, off stream_pcd's TCP feed (3d/tools/stream_pcd.cpp, running on the Pi)
        # yaw_offset: the SDK's X about the axis against the CAD's, MEASURED 2026-09-15
        # off the L2's own accelerometer with the robot standing: up is 43.5 deg from the
        # SDK's +X towards +Y; in lidar_link (rpy 0 pi/2 0) up is -X, so +136.5 deg.
        Node(package='smalldog_hardware', executable='lidar', name='smalldog_lidar',
             output='screen', condition=IfCondition(LaunchConfiguration('lidar')),
             parameters=[{'source': LaunchConfiguration('lidar_source'),
                          'yaw_offset': 2.382}]),

        # the IMX415, the UVC module's own MJPEG frames passed through (camera_node.py)
        Node(package='smalldog_hardware', executable='camera', name='smalldog_camera',
             output='screen', condition=IfCondition(LaunchConfiguration('camera')),
             parameters=[{'input': LaunchConfiguration('camera_input'),
                          'device': LaunchConfiguration('camera_device'),
                          'width': ParameterValue(LaunchConfiguration('camera_width'), value_type=int),
                          'height': ParameterValue(LaunchConfiguration('camera_height'), value_type=int)}]),

        # Foxglove over the LAN: everything above (the robot itself from
        # /robot_description), plus /map, /scan and the costmaps when nav.launch.py is
        # up. 0.0.0.0, unlike the sim's bridge — the Pi is the robot, the mac is where
        # the screen is. README, "Watching the robot".
        Node(package='foxglove_bridge', executable='foxglove_bridge', name='foxglove_bridge',
             output='screen', condition=IfCondition(LaunchConfiguration('foxglove')),
             parameters=[{'address': '0.0.0.0', 'port': 8765,
                          'send_buffer_limit': 100_000_000}]),
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
        DeclareLaunchArgument('joy', default_value='false', description='the USB gamepad'),
        DeclareLaunchArgument('lidar', default_value='false',
                              description='publish the L2 on /lidar/points (needs stream_pcd)'),
        DeclareLaunchArgument('lidar_source', default_value='127.0.0.1:9910',
                              description='where stream_pcd serves'),
        DeclareLaunchArgument('camera', default_value='false',
                              description='publish the IMX415 on /camera/image/compressed'),
        DeclareLaunchArgument('camera_device', default_value='/dev/video0'),
        DeclareLaunchArgument('camera_input', default_value='v4l2',
                              description='ffmpeg input format; lavfi + '
                                          'camera_device:=testsrc=size=1280x720:rate=20 '
                                          'is a test pattern on a machine with no camera'),
        DeclareLaunchArgument('camera_width', default_value='1280'),
        DeclareLaunchArgument('camera_height', default_value='720'),
        DeclareLaunchArgument('foxglove', default_value='false',
                              description='foxglove_bridge on ws://<this host>:8765; open '
                                          'smalldog_hardware/foxglove/robot.json in Foxglove'),
        OpaqueFunction(function=_nodes),
    ])
