from launch import LaunchDescription
from launch.actions import RegisterEventHandler, DeclareLaunchArgument, ExecuteProcess
from launch.conditions import IfCondition
from launch.event_handlers import OnProcessExit
from launch.substitutions import (Command, FindExecutable, PathJoinSubstitution,
                                  LaunchConfiguration)
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    teleop = LaunchConfiguration('teleop')

    robot_description_content = Command([
        PathJoinSubstitution([FindExecutable(name='xacro')]), ' ',
        PathJoinSubstitution([FindPackageShare('smalldog_ros_control'),
                              'urdf', 'smalldog-mujoco.urdf.xacro']),
    ])
    robot_description = {'robot_description':
                         ParameterValue(robot_description_content, value_type=str)}

    controllers = PathJoinSubstitution([
        FindPackageShare('smalldog_ros_control'), 'config', 'smalldog-controllers.yaml'])

    mujoco_xml = PathJoinSubstitution([
        FindPackageShare('smalldog_description'), 'mujoco', 'scene.xml'])

    mujoco_node = Node(
        package='mujoco_ros2_control', executable='mujoco_ros2_control', output='screen',
        parameters=[robot_description, controllers, {'mujoco_model_path': mujoco_xml}],
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

    walker = Node(
        package='smalldog_walker', executable='walker', name='smalldog_walker',
        output='screen',
        # the gait ramps and integrates in simulated time, like the controllers do
        parameters=[{'use_sim_time': True}],
    )

    # keyboard teleop needs its own TTY, so it is launched in a new terminal window
    keyboard = Node(
        package='smalldog_teleop', executable='keyboard', name='smalldog_keyboard_teleop',
        output='screen', emulate_tty=True, prefix='xterm -e',
        condition=IfCondition(teleop),
    )

    return LaunchDescription([
        DeclareLaunchArgument('teleop', default_value='false',
                              description='spawn keyboard teleop in an xterm; '
                                          'otherwise run `ros2 run smalldog_teleop keyboard` '
                                          'yourself in a second terminal'),
        mujoco_node,
        robot_state_pub,
        jsb_spawner,
        ctrl_after_jsb,
        walker,
        keyboard,
    ])
