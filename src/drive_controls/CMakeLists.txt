cmake_minimum_required(VERSION 3.8)
project(drive_controls)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclpy REQUIRED)

# Install executable Python scripts
install(PROGRAMS
  scripts/joy_mapper_node.py
  scripts/drive_controller.py
  scripts/command_splitter.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()