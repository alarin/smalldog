from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command, FindExecutable, PathJoinSubstitution


def generate_launch_description():
    urdf = PathJoinSubstitution([FindPackageShare('smalldog_description'), 'urdf', 'smalldog.urdf'])
    robot_description = {'robot_description': ParameterValue(
        Command([FindExecutable(name='cat'), ' ', urdf]), value_type=str)}
    rviz = PathJoinSubstitution([FindPackageShare('smalldog_description'), 'smalldog.rviz'])

    return LaunchDescription([
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             output='screen', parameters=[robot_description]),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui',
             output='screen'),
        Node(package='rviz2', executable='rviz2', output='screen', arguments=['-d', rviz]),
    ])
