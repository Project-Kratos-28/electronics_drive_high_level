import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, TimerAction
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
import xacro


def generate_launch_description():
    # 1. Paths and xacro -> URDF string
    pkg_path = get_package_share_directory('rover_description')
    xacro_file = os.path.join(pkg_path, 'urdf', 'kratos_rover.urdf.xacro')
    robot_description_config = xacro.process_file(xacro_file)
    robot_description = {'robot_description': robot_description_config.toxml()}

    # 2. Robot State Publisher (sim time so TF matches Gazebo's clock)
    node_robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description, {'use_sim_time': True}],
    )

    # 3. Gazebo Harmonic (gz sim), empty world, start running immediately
    gz_sim = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('ros_gz_sim'), 'launch', 'gz_sim.launch.py')),
        launch_arguments={'gz_args': '-r empty.sdf'}.items(),
    )

    # 4. Spawn the robot (slightly above ground so wheels don't clip)
    spawn_entity = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=['-topic', 'robot_description',
                   '-name', 'kratos_rover',
                   '-z', '0.3'],
        output='screen',
    )

    # 5. Bridge Gazebo's clock to ROS (/clock)
    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )

    # 6. ros2_control controller spawners
    def make_spawner(name):
        return Node(
            package='controller_manager',
            executable='spawner',
            arguments=[name, '--controller-manager', '/controller_manager'],
            output='screen',
        )

    spawn_jsb = make_spawner('joint_state_broadcaster')
    spawn_steering = make_spawner('steering_controller')
    spawn_velocity = make_spawner('velocity_controller')

    # Delay the first spawner by 8.0 seconds to give Gazebo time to load ros2_control hardware interfaces
    delayed_jsb_spawner = TimerAction(
        period=8.0,
        actions=[spawn_jsb]
    )

    # Chain subsequent controllers after joint_state_broadcaster succeeds
    after_jsb = RegisterEventHandler(
        OnProcessExit(target_action=spawn_jsb, on_exit=[spawn_steering]))
    after_steering = RegisterEventHandler(
        OnProcessExit(target_action=spawn_steering, on_exit=[spawn_velocity]))

    # 7. Joystick -> /joy (autorepeat keeps /cmd_vel alive for the watchdog)
    joy_node = Node(
        package='joy',
        executable='joy_node',
        name='joy_node',
        output='screen',
        parameters=[{'autorepeat_rate': 20.0}],
    )

    # 8. Your nodes: /joy -> /cmd_vel + /drive_mode -> controller commands
    joy_mapper = Node(
        package='drive_controls',
        executable='joy_mapper_node.py',
        name='joy_mapper_node',
        output='screen',
    )

    drive_controller = Node(
        package='drive_controls',
        executable='drive_controller.py',
        name='drive_controller',
        output='screen',
    )

    # Splits /drive_controller/commands (8 values) into steering (4) + velocity (4)
    command_splitter = Node(
        package='drive_controls',
        executable='command_splitter.py',
        name='command_splitter',
        output='screen',
    )

    # Give robot_state_publisher time to publish the robot description before
    # starting the rest of the simulation and control nodes.
    delayed_simulation_start = TimerAction(
        period=5.0,
        actions=[
            gz_sim,
            clock_bridge,
            spawn_entity,
            delayed_jsb_spawner,
            after_jsb,
            after_steering,
            joy_node,
            joy_mapper,
            drive_controller,
            command_splitter,
        ],
    )

    return LaunchDescription([
        node_robot_state_publisher,
        delayed_simulation_start,
    ])