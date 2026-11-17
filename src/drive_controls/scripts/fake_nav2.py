#!/usr/bin/env python3
"""Fake Nav2 for testing the auto switch.

Publishes random (vx, wz) on /cmd_vel_nav at 10 Hz, ALWAYS (it ignores /auto_mode),
picking a new random pair every `change_period` seconds so the change is easy to see.
joy_mapper_node forwards it to /cmd_vel only while /auto_mode is True.

Values stay inside what DOUBLE_ACKERMANN can do (max wheel angle 0.5 rad):
  |wz| <= |vx| * 1.0   (max curvature ~1.06 1/m)

Send /auto_mode True / False separately from a terminal.

Run (no build needed):
  python3 fake_nav2.py
"""
import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist


class FakeNav2(Node):
    def __init__(self):
        super().__init__('fake_nav2')
        self.declare_parameter('max_vx', 1.0)           # m/s
        self.declare_parameter('change_period', 2.0)    # s between new random values

        self.max_vx = self.get_parameter('max_vx').value
        self.change_period = self.get_parameter('change_period').value

        self.pub = self.create_publisher(Twist, '/cmd_vel_nav', 10)

        self.vx = 0.0
        self.wz = 0.0
        self.pick_new()

        self.create_timer(0.1, self.publish_cmd)                  # 10 Hz
        self.create_timer(self.change_period, self.pick_new)
        self.get_logger().info(
            "Fake Nav2 running. Always publishing random /cmd_vel_nav.")

    def pick_new(self):
        self.vx = random.uniform(0.2, self.max_vx)
        self.wz = random.uniform(-1.0, 1.0) * self.vx
        self.get_logger().info(f"new target: vx={self.vx:.2f} m/s  wz={self.wz:.2f} rad/s")

    def publish_cmd(self):
        msg = Twist()
        msg.linear.x = self.vx
        msg.angular.z = self.wz
        self.pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = FakeNav2()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.pub.publish(Twist())                      # final stop
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()