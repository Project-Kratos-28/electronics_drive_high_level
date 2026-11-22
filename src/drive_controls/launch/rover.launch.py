import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    params = os.path.join(
        get_package_share_directory('drive_controls'), 'config', 'rover_params.yaml')

    return LaunchDescription([
        Node(package='joy', executable='joy_node', name='joy_node'),
        Node(package='drive_controls', executable='joy_mapper_node.py',
             name='joy_mapper_node', parameters=[params], output='screen'),
        Node(package='drive_controls', executable='drive_controller.py',
             name='drive_controller', parameters=[params], output='screen'),
    ])
