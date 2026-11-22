#!/usr/bin/env python3
"""Joystick / Nav2 source selector -> STANDARD /cmd_vel.

Manual vs autonomous
  /auto_mode     std_msgs/Bool (input). False = manual (default at boot), True = autonomous.
  /cmd_vel       geometry_msgs/Twist, the only topic drive_controller listens to:
                   manual -> this node publishes it from the joystick
                   auto   -> this node stays silent; Nav2 publishes /cmd_vel directly
                 One zero Twist is sent on every handover. While auto, the drive mode is
                 forced to AUTO_DRIVE_MODE (Nav2 only gives vx and wz). In manual, make sure
                 Nav2 has no active goal (cancel it), or both will publish on /cmd_vel.
  Gamepad override: if the 'manual override' button is set (param), pressing it during
  auto switches back to manual immediately and ignores /auto_mode True until a False
  has been received.
  Optional heartbeat: auto_timeout > 0 -> True must keep arriving or this node falls back
  to manual.

Other outputs
  /drive_mode  std_msgs/String (latched): current drive mode name
  /crab_aim    std_msgs/Float64: crab wheel angle (rad, + = right), lets the wheels pre-aim
               while stopped without a fake vector on /cmd_vel

Crab walk: the wheel angle is kept here (axis swings it, button re-centers it) and is
published as a velocity vector: vx = v*cos(a), vy = -v*sin(a), a = wheel angle (+ = right).

Parameters: see config/rover_params.yaml (shared with drive_controller).
"""
import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float64, String


class JoyMapperNode(Node):
    def __init__(self):
        super().__init__('joy_mapper_node')

        # --- Parameters (shared ones must match drive_controller: use one yaml) ---
        self.declare_parameter('wheelbase', 1.032)
        self.declare_parameter('track_width', 0.79)
        self.declare_parameter('wheel_radius', 0.30)
        self.declare_parameter('max_wheel_rpm', 100.0)
        self.declare_parameter('max_steer_rad', 0.5)
        self.declare_parameter('crab_max_angle_deg', 90.0)
        # Joy-mapper only
        self.declare_parameter('auto_timeout', 0.0)          # 0 = no heartbeat required
        self.declare_parameter('button_manual_override', -1)  # gamepad button index, -1 = off

        p = self.get_parameter
        self.wheelbase = p('wheelbase').value
        self.track_width = p('track_width').value
        wheel_radius = p('wheel_radius').value
        max_rpm = p('max_wheel_rpm').value
        self.max_steer_rad = p('max_steer_rad').value
        self.CRAB_MAX_ANGLE_DEG = p('crab_max_angle_deg').value
        self.AUTO_TIMEOUT = p('auto_timeout').value
        self.BUTTON_MANUAL_OVERRIDE = p('button_manual_override').value

        # Joystick full trigger = max_wheel_rpm, converted ONCE to m/s because /cmd_vel is a Twist
        self.max_speed = max_rpm * 2.0 * math.pi / 60.0 * wheel_radius
        self.max_yaw_rate = self.max_speed   # rad/s at full stick (point turn)

        # --- Modes ---
        self.modes = ['DOUBLE_ACKERMANN', 'DIFFERENTIAL', 'POINT_TURN', 'CRAB_WALK']
        self.mode_idx = 0
        self.prev_button_b = 0
        self.prev_button_pt = 0
        self.prev_button_override = 0
        # B cycles only these; POINT_TURN has its own button and is left with B.
        self.cycle_modes = ['DOUBLE_ACKERMANN', 'DIFFERENTIAL', 'CRAB_WALK']

        # Pad layout (as observed): axes 0/1 = right stick, axis 2 = speed trigger.
        self.AXIS_FORWARD = 1     # Right stick vertical   -> forward amount (up = +)
        self.AXIS_LATERAL = 0     # Right stick horizontal -> steering / turn (left = +)
        self.AXIS_YAW = 2         # Speed trigger (rest = -1); raw stick in POINT_TURN
        self.AXIS_CRAB_ANGLE = 5  # Crab walk wheel angle
        self.BUTTON_MODE = 1          # cycle drive mode (not POINT_TURN)
        self.BUTTON_POINT_TURN = 4    # jump to POINT_TURN; B returns to the cycle
        self.BUTTON_CRAB_CENTER = 3   # crab wheels back to straight ahead

        # --- Signal conventions ---
        self.deadzone = 0.15
        self.INVERT_POINT_TURN_W = -1.0
        self.INVERT_CRAB_ANGLE = -1.0
        self.CRAB_SPEED_IS_TRIGGER = True    # speed axis rest = -1 -> mapped to 0..1
        self.CRAB_ANGLE_IS_TRIGGER = False   # angle axis is a stick resting at 0

        # --- Timing / crab ---
        self.control_dt = 0.02
        self.joy_timeout = 0.5               # stop publishing if /joy goes quiet
        self.CRAB_STEER_RATE_DEG_S = 60.0    # wheel swing speed at full axis deflection
        self.crab_angle_rad = 0.0            # commanded crab wheel angle (+ = right)

        self.last_joy = None
        self.last_joy_time = None
        self.last_mode_switch = self.get_clock().now()
        self.debounce_sec = 0.3

        # --- Auto / manual ---
        self.auto = False
        self.auto_locked_out = False         # set by the gamepad override
        self.AUTO_DRIVE_MODE = 'DOUBLE_ACKERMANN'
        self.last_auto_true = None

        mode_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Bool, '/auto_mode', self.auto_callback, 10)
        self.create_subscription(Joy, '/joy', self.joy_callback, 10)
        self.pub_cmd_vel = self.create_publisher(Twist, '/cmd_vel', 10)
        self.pub_mode = self.create_publisher(String, '/drive_mode', mode_qos)
        self.pub_aim = self.create_publisher(Float64, '/crab_aim', 10)
        self.was_active = False
        self.timer = self.create_timer(self.control_dt, self.tick)

        self.get_logger().info(
            f"Joy Mapper active (manual). Full speed = {self.max_speed:.3f} m/s")
        self.publish_mode()

    def publish_mode(self):
        msg = String()
        msg.data = self.modes[self.mode_idx]
        self.pub_mode.publish(msg)
        self.get_logger().info(f"Drive Mode Set To: {msg.data}")

    # ------------------------------------------------------------------
    # Auto / manual switching
    # ------------------------------------------------------------------
    def auto_callback(self, msg: Bool):
        """/auto_mode: True -> Nav2 drives, False -> joystick drives."""
        if not msg.data:
            self.auto_locked_out = False          # explicit False clears the lockout
        elif self.auto_locked_out:
            return                                # gamepad override is active
        else:
            self.last_auto_true = self.get_clock().now()   # heartbeat
        self.set_auto(msg.data)

    def set_auto(self, on: bool):
        if on == self.auto:
            return
        self.auto = on
        self.pub_cmd_vel.publish(Twist())     # one clean stop on every handover
        self.was_active = False
        if self.auto:
            self.set_mode(self.AUTO_DRIVE_MODE)
        self.get_logger().info(f"AUTO {'ON (Nav2 drives)' if self.auto else 'OFF (manual)'}")

    # ------------------------------------------------------------------
    # Joystick helpers
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

    def crab_update_angle(self, msg, center):
        """Crab axis -> wheel angle (rad, 0 = forward, + = right).

        Each cycle the target moves from the last commanded angle by
        deflection * rate * dt. A released axis holds the angle; the re-center button
        snaps it back to straight ahead.
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

        base = self.crab_angle_rad
        step = deflection * math.radians(self.CRAB_STEER_RATE_DEG_S) * self.control_dt
        limit = math.radians(self.CRAB_MAX_ANGLE_DEG)
        self.crab_angle_rad = max(-limit, min(limit, base + step))

    def set_mode(self, name):
        self.mode_idx = self.modes.index(name)
        self.crab_angle_rad = 0.0
        self.publish_mode()
        self.last_mode_switch = self.get_clock().now()

    def debounced(self):
        now = self.get_clock().now()
        return (now - self.last_mode_switch).nanoseconds * 1e-9 > self.debounce_sec

    def button(self, msg, idx):
        return msg.buttons[idx] if 0 <= idx < len(msg.buttons) else 0

    # ------------------------------------------------------------------
    def joy_callback(self, msg: Joy):
        # Gamepad override works in both modes (edge-triggered).
        ov = self.button(msg, self.BUTTON_MANUAL_OVERRIDE)
        if ov == 1 and self.prev_button_override == 0 and self.auto:
            self.auto_locked_out = True
            self.set_auto(False)
            self.get_logger().warn("Manual override pressed -> auto OFF")
        self.prev_button_override = ov

        if self.auto:
            # Ignore mode buttons in auto, but keep edge state in sync so a button held
            # during the handover doesn't fire when manual resumes.
            self.prev_button_b = self.button(msg, self.BUTTON_MODE)
            self.prev_button_pt = self.button(msg, self.BUTTON_POINT_TURN)
            self.last_joy = msg
            self.last_joy_time = self.get_clock().now()
            return

        # B (rising edge): next mode in the cycle. From POINT_TURN, back to the cycle start.
        current_b = self.button(msg, self.BUTTON_MODE)
        if current_b == 1 and self.prev_button_b == 0 and self.debounced():
            current = self.modes[self.mode_idx]
            if current in self.cycle_modes:
                i = self.cycle_modes.index(current)
                nxt = self.cycle_modes[(i + 1) % len(self.cycle_modes)]
            else:
                nxt = self.cycle_modes[0]
            self.set_mode(nxt)
        self.prev_button_b = current_b

        # Point-turn button (rising edge)
        current_pt = self.button(msg, self.BUTTON_POINT_TURN)
        if (current_pt == 1 and self.prev_button_pt == 0 and self.debounced()
                and self.modes[self.mode_idx] != 'POINT_TURN'):
            self.set_mode('POINT_TURN')
        self.prev_button_pt = current_pt

        self.last_joy = msg
        self.last_joy_time = self.get_clock().now()

    def tick(self):
        """Periodic: latest joystick state -> standard /cmd_vel (manual only)."""
        if self.auto:
            if self.AUTO_TIMEOUT > 0.0 and self.last_auto_true is not None:
                age = (self.get_clock().now() - self.last_auto_true).nanoseconds * 1e-9
                if age > self.AUTO_TIMEOUT:
                    self.get_logger().warn("/auto_mode heartbeat lost -> back to manual")
                    self.set_auto(False)
            return   # autonomous: Nav2 publishes /cmd_vel itself
        if self.last_joy is None:
            return
        age = (self.get_clock().now() - self.last_joy_time).nanoseconds * 1e-9
        if age > self.joy_timeout:
            return   # /joy quiet: drive_controller's watchdog stops the rover

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
            # 4-sector logic. Stick angle: 0 = forward, + = right, +-180 = back.
            steer_x = -side
            if speed < 1e-4 or abs(fwd) + abs(steer_x) < 1e-4:
                pass                             # stop: zero Twist
            else:
                theta_deg = math.degrees(math.atan2(steer_x, fwd))
                if abs(theta_deg) <= 45.0:
                    cmd.linear.x = speed
                elif abs(theta_deg) >= 135.0:
                    cmd.linear.x = -speed
                else:
                    # Spin in place with wheel rim speed = speed (stick right = clockwise).
                    spin = speed / (self.track_width / 2.0)
                    cmd.angular.z = -spin if steer_x > 0 else spin

        elif mode == 'POINT_TURN':
            raw_w = self.apply_deadzone(self.axis(msg, self.AXIS_YAW))
            cmd.angular.z = self.INVERT_POINT_TURN_W * raw_w * self.max_yaw_rate

        elif mode == 'CRAB_WALK':
            center = bool(self.button(msg, self.BUTTON_CRAB_CENTER))
            self.crab_update_angle(msg, center)

            aim = Float64()
            aim.data = self.crab_angle_rad
            self.pub_aim.publish(aim)

            s = fwd * speed                      # signed speed along the wheel direction
            cmd.linear.x = s * math.cos(self.crab_angle_rad)
            cmd.linear.y = -s * math.sin(self.crab_angle_rad)

        # Speak on /cmd_vel only while commanding motion, plus one final zero on release.
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
        try:
            node.pub_cmd_vel.publish(Twist())
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()