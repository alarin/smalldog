import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'smalldog_hardware'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         glob(os.path.join('launch', '*.launch.py'))),
        (os.path.join('share', package_name, 'foxglove'),
         glob(os.path.join('foxglove', '*.json'))),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alarin',
    maintainer_email='me@alarin.ru',
    description='The real robot as a ROS 2 node',
    license='MIT',
    entry_points={
        'console_scripts': [
            'servos = smalldog_hardware.servo_node:main',
            'lidar = smalldog_hardware.lidar_node:main',
            'camera = smalldog_hardware.camera_node:main',
        ],
    },
)
