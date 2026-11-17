#!/usr/bin/env python3
"""Joystick -> STANDARD /cmd_vel for every drive mode.

/cmd_vel     geometry_msgs/Twist, real units:
               linear.x   m/s, + = forward
               linear.y   m/s, + = left
               angular.z  rad/s, + = counter-clockwise
             Published only while the joystick is actually commanding motion (plus one
             final zero Twist on release), so it never fights Nav2 while idle.
/drive_mode  std_msgs/String (latched): current drive mode name
/crab_aim    std_msgs/Float64: crab wheel angle (rad, + = right). Lets the wheels pre-aim /
             re-center while stopped without putting a fake vector on /cmd_vel.

Crab walk: the wheel angle is kept here (axis 5 swings it, button 3 re-centers it) and is
published as the direction of the velocity vector: vx = v*cos(a), vy = -v*sin(a), where
a is the wheel angle (+ = right).
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64, Float64MultiArray, String


class JoyMapperNode(Node):
    def __init__(self):
        super().__init__('joy_mapper_node')

        self.modes = ['DOUBLE_ACKERMANN', 'DIFFERENTIAL', 'POINT_TURN', 'CRAB_WALK']
        self.mode_idx = 0
        self.prev_button_b = 0
        self.prev_button_pt = 0
        # B cycles only these; POINT_TURN is reached with its own button and left with B.
        self.cycle_modes = ['DOUBLE_ACKERMANN', 'DIFFERENTIAL', 'CRAB_WALK']

        # Pad layout (as observed): axes 0/1 = right stick, axis 2 = left stick vertical.
        self.AXIS_FORWARD = 1     # Right stick vertical   -> forward amount (up = +)
        self.AXIS_LATERAL = 0     # Right stick horizontal -> steering / turn (left = +)
        self.AXIS_YAW = 2         # Speed trigger (rest = -1); raw stick in POINT_TURN
        self.AXIS_CRAB_ANGLE = 5  # Crab walk wheel angle
        self.BUTTON_MODE = 1          # 'B' button: cycle drive mode (not POINT_TURN)
        self.BUTTON_POINT_TURN = 4    # jump straight to POINT_TURN; B returns to the cycle
        self.BUTTON_CRAB_CENTER = 3   # crab wheels back to straight ahead

        # --- Scaling to real units ---
        self.deadzone = 0.15
        # m/s at full speed trigger: 100 rpm on 0.30 m wheels (keep in sync with drive_controller)
        self.max_speed = 100.0 * 2.0 * math.pi / 60.0 * 0.30
        self.max_yaw_rate = self.max_speed   # rad/s at full stick (point turn)
        # Geometry (keep in sync with drive_controller)
        self.wheelbase = 1.032        # m
        self.track_width = 0.79       # m
        self.max_steer_rad = 0.5      # rad, wheel angle at full stick (double Ackermann)

        # --- Signal conventions ---
        self.INVERT_POINT_TURN_W = -1.0
        self.INVERT_CRAB_ANGLE = -1.0
        self.CRAB_SPEED_IS_TRIGGER = True    # speed axis rest = -1 -> mapped to 0..1
        self.CRAB_ANGLE_IS_TRIGGER = False   # angle axis is a stick resting at 0

        # --- Crab walk wheel angle (moved here from drive_controller) ---
        self.control_dt = 0.02               # loop period (s)
        self.joy_timeout = 0.5               # stop publishing if /joy goes quiet
        self.CRAB_STEER_RATE_DEG_S = 60.0    # wheel swing speed at full axis deflection
        self.CRAB_MAX_ANGLE_DEG = 90.0       # end stops, +/- from straight ahead
        self.STEER_FEEDBACK_TOPIC = '/steering_controller/feedback'
        self.FEEDBACK_TIMEOUT = 0.5
        self.PUBLISH_SERVO_DEG = False       # feedback units (same flag as drive_controller)
        self.CENTER_ANGLE_DEG = 180.0
        self.crab_angle_rad = 0.0            # commanded crab wheel angle (+ = right)
        self.fb_angle_rad = None
        self.fb_time = None

        self.last_joy = None
        self.last_joy_time = None
        self.last_mode_switch = self.get_clock().now()
        self.debounce_sec = 0.3

        mode_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.sub_joy = self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.sub_feedback = self.create_subscription(
            Float64MultiArray, self.STEER_FEEDBACK_TOPIC, self.feedback_callback, 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_mode = self.create_publisher(String, '/drive_mode', mode_qos)
        self.pub_aim = self.create_publisher(Float64, '/crab_aim', 10)
        self.was_active = False
        self.timer = self.create_timer(self.control_dt, self.tick)

        self.get_logger().info("Joy Mapper Active: /joy -> /cmd_vel (m/s, rad/s) & /drive_mode")
        self.publish_mode()

    def publish_mode(self):
        mode_msg = String()
        mode_msg.data = self.modes[self.mode_idx]
        self.pub_mode.publish(mode_msg)
        self.get_logger().info(f"Drive Mode Set To: {mode_msg.data}")

    # ------------------------------------------------------------------
    def apply_deadzone(self, val):
        if abs(val) < self.deadzone:
            return 0.0
        sign = 1.0 if val > 0 else -1.0
        scaled = sign * (abs(val) - self.deadzone) / (1.0 - self.deadzone)
        return 0.0 if abs(scaled) < 1e-3 else scaled

    def axis(self, msg, idx):
        return msg.axes[idx] if len(msg.axes) > idx else 0.0

    def speed_command(self, msg):
        """Speed axis -> 0..max_speed (m/s)."""
        raw = self.axis(msg, self.AXIS_YAW)
        if self.CRAB_SPEED_IS_TRIGGER:
            norm = max(0.0, min(1.0, (raw + 1.0) / 2.0))
        else:
            norm = max(0.0, min(1.0, raw))
        return self.apply_deadzone(norm) * self.max_speed

    def feedback_callback(self, msg: Float64MultiArray):
        """Rover wheel angles -> one mean angle (rad from center, wrapped to [-pi, pi])."""
        if not msg.data:
            return
        if self.PUBLISH_SERVO_DEG:
            angles = [math.radians(d - self.CENTER_ANGLE_DEG) for d in msg.data]
        else:
            angles = list(msg.data)
        s = sum(math.sin(a) for a in angles)
        c = sum(math.cos(a) for a in angles)
        self.fb_angle_rad = math.atan2(s, c)
        self.fb_time = self.get_clock().now()

    def measured_angle(self):
        if self.fb_angle_rad is None or self.fb_time is None:
            return None
        age = (self.get_clock().now() - self.fb_time).nanoseconds * 1e-9
        return self.fb_angle_rad if age <= self.FEEDBACK_TIMEOUT else None

    def crab_update_angle(self, msg, center):
        """Axis 5 -> crab wheel angle (rad, 0 = forward, + = right).

        Each cycle the target moves from the *measured* angle by
        deflection * rate * dt (falls back to the last command without feedback), so the
        wheels swing gradually. A released axis holds the angle; the re-center button snaps
        it back to straight ahead.
        """
        if center:
            self.crab_angle_rad = 0.0
            return

        raw = self.INVERT_CRAB_ANGLE * self.axis(msg, self.AXIS_CRAB_ANGLE)
        if self.CRAB_ANGLE_IS_TRIGGER:
            norm = max(0.0, min(1.0, (raw + 1.0) / 2.0))
        else:
            norm = max(-1.0, min(1.0, raw))
        deflection = self.apply_deadzone(norm)
        if deflection == 0.0:
            return

        fb = self.measured_angle()
        base = fb if fb is not None else self.crab_angle_rad
        step = deflection * math.radians(self.CRAB_STEER_RATE_DEG_S) * self.control_dt
        limit = math.radians(self.CRAB_MAX_ANGLE_DEG)
        self.crab_angle_rad = max(-limit, min(limit, base + step))

    # ------------------------------------------------------------------
    def set_mode(self, name):
        self.mode_idx = self.modes.index(name)
        self.crab_angle_rad = 0.0
        self.publish_mode()
        self.last_mode_switch = self.get_clock().now()

    def debounced(self):
        now = self.get_clock().now()
        return (now - self.last_mode_switch).nanoseconds * 1e-9 > self.debounce_sec

    def joy_callback(self, msg: Joy):
        # B (rising edge): next mode in the cycle. From POINT_TURN, back to the cycle start.
        if len(msg.buttons) > self.BUTTON_MODE:
            current_b = msg.buttons[self.BUTTON_MODE]
            if current_b == 1 and self.prev_button_b == 0 and self.debounced():
                current = self.modes[self.mode_idx]
                if current in self.cycle_modes:
                    i = self.cycle_modes.index(current)
                    nxt = self.cycle_modes[(i + 1) % len(self.cycle_modes)]
                else:
                    nxt = self.cycle_modes[0]
                self.set_mode(nxt)
            self.prev_button_b = current_b

        # Button 4 (rising edge): jump to POINT_TURN
        if len(msg.buttons) > self.BUTTON_POINT_TURN:
            current_pt = msg.buttons[self.BUTTON_POINT_TURN]
            if (current_pt == 1 and self.prev_button_pt == 0 and self.debounced()
                    and self.modes[self.mode_idx] != 'POINT_TURN'):
                self.set_mode('POINT_TURN')
            self.prev_button_pt = current_pt

        self.last_joy = msg
        self.last_joy_time = self.get_clock().now()

    def tick(self):
        """Periodic: latest joystick state -> standard /cmd_vel."""
        if self.last_joy is None:
            return
        age = (self.get_clock().now() - self.last_joy_time).nanoseconds * 1e-9
        if age > self.joy_timeout:
            return   # /joy went quiet: stop publishing, drive_controller's watchdog stops the rover

        msg = self.last_joy
        mode = self.modes[self.mode_idx]
        fwd = self.apply_deadzone(self.axis(msg, self.AXIS_FORWARD))
        side = self.apply_deadzone(self.axis(msg, self.AXIS_LATERAL))   # left = +
        speed = self.speed_command(msg)                                  # 0..max_speed

        cmd = Twist()
        if mode == 'DOUBLE_ACKERMANN':
            vx = fwd * speed
            # Right stick -> wheel angle (right = +), then yaw rate = v * curvature.
            delta = (-side) * self.max_steer_rad
            curvature = -math.tan(delta) * 2.0 / self.wheelbase          # + = turning left
            cmd.linear.x = vx
            cmd.angular.z = vx * curvature

        elif mode == 'DIFFERENTIAL':
            # 4-sector logic, expressed as a standard Twist.
            # Stick angle: 0 = straight forward, + = right, +-180 = straight back.
            steer_x = -side                      # + = stick right
            if speed < 1e-4 or abs(fwd) + abs(steer_x) < 1e-4:
                pass                             # stop: zero Twist
            else:
                theta_deg = math.degrees(math.atan2(steer_x, fwd))
                if abs(theta_deg) <= 45.0:       # FRONT sector: straight forward
                    cmd.linear.x = speed
                elif abs(theta_deg) >= 135.0:    # BACK sector: straight backward
                    cmd.linear.x = -speed
                else:
                    # Spin in place with wheel rim speed = speed:
                    # wz = (v_right - v_left) / track.  Stick right = clockwise = negative.
                    spin = speed / (self.track_width / 2.0)
                    cmd.angular.z = -spin if steer_x > 0 else spin

        elif mode == 'POINT_TURN':
            raw_w = self.apply_deadzone(self.axis(msg, self.AXIS_YAW))
            cmd.angular.z = self.INVERT_POINT_TURN_W * raw_w * self.max_yaw_rate

        elif mode == 'CRAB_WALK':
            center = (len(msg.buttons) > self.BUTTON_CRAB_CENTER
                      and bool(msg.buttons[self.BUTTON_CRAB_CENTER]))
            self.crab_update_angle(msg, center)

            # Wheel angle goes on its own topic so /cmd_vel stays a plain Twist.
            aim = Float64()
            aim.data = self.crab_angle_rad
            self.pub_aim.publish(aim)

            s = fwd * speed                      # signed speed along the wheel direction
            # Wheel angle a (+ = right)  ->  velocity direction in the standard frame
            cmd.linear.x = s * math.cos(self.crab_angle_rad)
            cmd.linear.y = -s * math.sin(self.crab_angle_rad)

        # Only speak on /cmd_vel while actually commanding motion, plus one final zero on
        # release. An idle joystick stays silent so it doesn't override Nav2.
        active = (abs(cmd.linear.x) > 1e-4 or abs(cmd.linear.y) > 1e-4
                  or abs(cmd.angular.z) > 1e-4)
        if active or self.was_active:
            self.pub_cmd_vel.publish(cmd)
        self.was_active = active


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