from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(package='smalldog_walker', executable='walker',
             name='smalldog_walker', output='screen'),
    ])
