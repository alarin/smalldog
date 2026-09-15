from glob import glob
from setuptools import find_packages, setup

package_name = 'smalldog_nav'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alarin',
    maintainer_email='me@alarin.ru',
    description='SLAM, Nav2 and frontier exploration for SmallDog',
    license='MIT',
    entry_points={
        'console_scripts': [
            'explore = smalldog_nav.explore:main',
            'cloud_to_scan = smalldog_nav.cloud_to_scan:main',
        ],
    },
)
