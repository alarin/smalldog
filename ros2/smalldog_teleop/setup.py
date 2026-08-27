from setuptools import find_packages, setup

package_name = 'smalldog_teleop'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='alarin',
    maintainer_email='me@alarin.ru',
    description='Keyboard teleop for SmallDog',
    license='MIT',
    entry_points={
        'console_scripts': [
            'keyboard = smalldog_teleop.keyboard_teleop:main',
        ],
    },
)
