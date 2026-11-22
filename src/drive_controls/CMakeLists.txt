cmake_minimum_required(VERSION 3.8)
project(drive_controls)

find_package(ament_cmake REQUIRED)

install(PROGRAMS
  scripts/drive_controller.py
  scripts/joy_mapper_node.py
  DESTINATION lib/${PROJECT_NAME})

install(DIRECTORY launch config
  DESTINATION share/${PROJECT_NAME})

ament_package()
