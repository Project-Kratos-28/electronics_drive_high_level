#!/usr/bin/env python3
"""Splits the single 8-element drive command into the two ros2_control forward controllers.

/drive_controller/commands  [steer FL,FR,RL,RR, vel FL,FR,RL,RR]
    -> /steering_controller/commands  (first 4, position, rad)
    -> /velocity_controller/commands  (last 4, velocity, rad/s)
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray


class CommandSplitter(Node):
    def __init__(self):
        super().__init__('command_splitter')
        self.steer_pub = self.create_publisher(
            Float64MultiArray, '/steering_controller/commands', 10)
        self.vel_pub = self.create_publisher(
            Float64MultiArray, '/velocity_controller/commands', 10)
        self.create_subscription(
            Float64MultiArray, '/drive_controller/commands', self.callback, 10)
        self.get_logger().info("Command splitter active (8 -> 4 + 4).")

    def callback(self, msg: Float64MultiArray):
        if len(msg.data) != 8:
            self.get_logger().warn(
                f"Expected 8 values, got {len(msg.data)}; ignoring.",
                throttle_duration_sec=2.0)
            return

        steer = Float64MultiArray()
        steer.data = list(msg.data[0:4])
        vel = Float64MultiArray()
        vel.data = list(msg.data[4:8])

        self.steer_pub.publish(steer)
        self.vel_pub.publish(vel)


def main(args=None):
    rclpy.init(args=args)
    node = CommandSplitter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
