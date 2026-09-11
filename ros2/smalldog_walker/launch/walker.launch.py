from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        # The same two parameters smalldog_ros_control's smalldog-mujoco.launch.py sets,
        # and for the same reasons: the gait ramps and integrates in SIMULATED time, and
        # imu_topic has to name the broadcaster's own topic because controllers publish
        # under their controller name.  Without them this launch file started a walker
        # that clocked itself off the wall and never saw an IMU - i.e. open loop, silently.
        Node(package='smalldog_walker', executable='walker',
             name='smalldog_walker', output='screen',
             parameters=[{'use_sim_time': True,
                          'imu_topic': '/imu_sensor_broadcaster/imu'}]),
    ])
