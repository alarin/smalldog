from launch import LaunchDescription
from launch.actions import RegisterEventHandler, DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import (Command, FindExecutable, PathJoinSubstitution,
                                  LaunchConfiguration, PythonExpression)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    teleop = LaunchConfiguration('teleop')
    terrain = LaunchConfiguration('terrain')

    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]), ' ',
        PathJoinSubstitution([FindPackageShare('smalldog_ros_control'),
                              'urdf', 'smalldog-mujoco.urdf.xacro']),
    ])
    robot_description = {'robot_description':
                         ParameterValue(robot_description_content, value_type=str)}

    controllers = PathJoinSubstitution([
        FindPackageShare('smalldog_ros_control'), 'config', 'smalldog-controllers.yaml'])

    # same world either way; scene_terrain.xml swaps the ground plane for the heightfield
    scene = PythonExpression(["'scene_terrain.xml' if '", terrain,
                              "'.lower() in ('true', '1') else 'scene.xml'"])
    mujoco_xml = PathJoinSubstitution([
        FindPackageShare('smalldog_description'), 'mujoco', scene])

    mujoco_node = Node(
        package='mujoco_ros2_control', executable='mujoco_ros2_control', output='screen',
        parameters=[robot_description, controllers, {'mujoco_model_path': mujoco_xml},
                    {'real_time_factor': ParameterValue(LaunchConfiguration('rtf'),
                                                        value_type=float)}],
    )

    robot_state_pub = Node(
        package='robot_state_publisher', executable='robot_state_publisher',
        output='both', parameters=[robot_description],
    )

    jsb_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['joint_state_broadcaster', '--controller-manager', '/controller_manager',
                   '--param-file', controllers],
    )
    ctrl_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['smalldog_controller', '--controller-manager', '/controller_manager',
                   '--param-file', controllers],
    )
    ctrl_after_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(target_action=jsb_spawner, on_exit=[ctrl_spawner]))

    # The IMU is what makes the gait terrain-aware; without it the walker trots blind.
    # It needs imu_sensor_broadcaster, which is not in every ros2_controllers install --
    # if the spawner fails, everything else still comes up, blind. `imu:=false` skips it.
    imu_spawner = Node(
        package='controller_manager', executable='spawner',
        arguments=['imu_sensor_broadcaster', '--controller-manager', '/controller_manager',
                   '--param-file', controllers],
        condition=IfCondition(LaunchConfiguration('imu')),
    )
    imu_after_jsb = RegisterEventHandler(
        event_handler=OnProcessExit(target_action=jsb_spawner, on_exit=[imu_spawner]))

    walker = Node(
        package='smalldog_walker', executable='walker', name='smalldog_walker',
        output='screen',
        # the gait ramps and integrates in simulated time, like the controllers do.
        # imu_topic is the broadcaster's own topic: controllers publish under their
        # controller name, and remapping into a controller_manager plugin is worse than
        # just telling the walker where to look.
        parameters=[{'use_sim_time': True,
                     'imu_topic': '/imu_sensor_broadcaster/imu'}],
    )

    # Foxglove instead of RViz: rviz2 on macOS is a large install and behaves badly, and
    # the LiDAR cloud is the first thing here that actually needs looking at.  The bridge
    # is headless - it serves a websocket the Foxglove app (or app.foxglove.dev in a
    # browser) connects to - so it costs nothing when nobody is watching.
    # Bound to localhost on purpose: the default is 0.0.0.0, and this is a dev machine.
    # Point it at 0.0.0.0 yourself if you want to watch from another box.
    foxglove = Node(
        package='foxglove_bridge', executable='foxglove_bridge', name='foxglove_bridge',
        output='screen',
        parameters=[{'use_sim_time': True, 'address': '127.0.0.1', 'port': 8765}],
        condition=IfCondition(LaunchConfiguration('foxglove')),
    )

    # Keyboard teleop takes its keys from the MuJoCo render window: the sim node
    # republishes every printable key pressed over its viewer on ~/key. So the teleop
    # needs no TTY of its own and runs here, in the launch, with stdin reading off.
    keyboard = Node(
        package='smalldog_teleop', executable='keyboard', name='smalldog_keyboard_teleop',
        output='screen',
        parameters=[{'key_topic': '/mujoco_ros2_control_node/key',
                     'read_stdin': False}],
        condition=IfCondition(teleop),
    )

    return LaunchDescription([
        DeclareLaunchArgument('teleop', default_value='true',
                              description='drive the robot from the MuJoCo window: keys '
                                          'pressed over the viewer go to the teleop node '
                                          'spawned here. Set false to run '
                                          '`ros2 run smalldog_teleop keyboard` yourself in '
                                          'a second terminal instead — two of them publish '
                                          'to /cmd_vel at once and fight'),
        DeclareLaunchArgument('imu', default_value='true',
                              description='spawn imu_sensor_broadcaster, which is what '
                                          'feeds the gait its terrain feedback; set false '
                                          'if that controller is not installed'),
        DeclareLaunchArgument('foxglove', default_value='false',
                              description='serve the topics to Foxglove on '
                                          'ws://localhost:8765 (foxglove_bridge). This is '
                                          'how the LiDAR cloud is meant to be looked at on '
                                          'macOS; see README, "Looking at it"'),
        DeclareLaunchArgument('rtf', default_value='1.0',
                              description='how fast the sim runs against the wall clock. '
                                          '1.0 is real time; 2.0 is twice as fast; 0 turns '
                                          'the pacing off entirely and lets it run as fast '
                                          'as the frame costs, which on this machine is '
                                          '~2.9x and is what it did before the pacing '
                                          'existed'),
        DeclareLaunchArgument('terrain', default_value='false',
                              description='stand the robot on mujoco/scene_terrain.xml '
                                          '(rough ground) instead of the flat plane'),
        mujoco_node,
        robot_state_pub,
        jsb_spawner,
        imu_after_jsb,
        ctrl_after_jsb,
        walker,
        keyboard,
        foxglove,
    ])
