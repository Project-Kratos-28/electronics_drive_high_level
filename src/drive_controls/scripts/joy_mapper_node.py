#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class JoyMapperNode(Node):
    def __init__(self):
        super().__init__('joy_mapper_node')

        self.modes = ['DOUBLE_ACKERMANN', 'DIFFERENTIAL', 'POINT_TURN', 'CRAB_WALK']
        self.mode_idx = 0
        self.prev_button_b = 0

        # Pad layout (as observed): axes 0/1 = right stick, axis 2 = left stick vertical.
        # Twist convention expected by drive_controller:
        #   linear.x = FORWARD (stick up = +), linear.y = SIDEWAYS (stick left = +)
        self.AXIS_FORWARD = 1    # Right stick vertical   -> linear.x
        self.AXIS_LATERAL = 0    # Right stick horizontal -> linear.y
        self.AXIS_YAW = 2        # Left stick vertical    -> angular.z (point turn)
        self.AXIS_CRAB_ANGLE = 5 # Crab walk wheel angle      -> angular.x
        self.BUTTON_MODE = 1     # 'B' button

        self.max_lin_vel = 1.0
        self.max_ang_vel = 1.0

        self.last_mode_switch = self.get_clock().now()
        self.debounce_sec = 0.3

        mode_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.sub_joy = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_mode = self.create_publisher(String, '/drive_mode', mode_qos)

        self.get_logger().info("Joy Mapper Active: /joy -> /cmd_vel & /drive_mode")
        self.publish_mode()

    def publish_mode(self):
        mode_msg = String()
        mode_msg.data = self.modes[self.mode_idx]
        self.pub_mode.publish(mode_msg)
        self.get_logger().info(f"Drive Mode Set To: {mode_msg.data}")

    def joy_callback(self, msg: Joy):
        # Mode switching on rising edge of B
        if len(msg.buttons) > self.BUTTON_MODE:
            current_b = msg.buttons[self.BUTTON_MODE]
            if current_b == 1 and self.prev_button_b == 0:
                now = self.get_clock().now()
                if (now - self.last_mode_switch).nanoseconds * 1e-9 > self.debounce_sec:
                    self.mode_idx = (self.mode_idx + 1) % len(self.modes)
                    self.publish_mode()
                    self.last_mode_switch = now
            self.prev_button_b = current_b

        # Velocity output
        cmd = Twist()
        if len(msg.axes) > self.AXIS_FORWARD:
            cmd.linear.x = msg.axes[self.AXIS_FORWARD] * self.max_lin_vel
        if len(msg.axes) > self.AXIS_LATERAL:
            cmd.linear.y = msg.axes[self.AXIS_LATERAL] * self.max_lin_vel
        if len(msg.axes) > self.AXIS_YAW:
            cmd.angular.z = msg.axes[self.AXIS_YAW] * self.max_ang_vel
        if len(msg.axes) > self.AXIS_CRAB_ANGLE:
            cmd.angular.x = msg.axes[self.AXIS_CRAB_ANGLE]  # raw -1..1, scaled in drive_controller
        self.pub_cmd_vel.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = JoyMapperNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()