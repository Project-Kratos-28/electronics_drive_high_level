#!/usr/bin/env python3
"""Swerve drive controller (real rover).

Inputs
  /cmd_vel      geometry_msgs/Twist, standard units (manual and Nav2 both arrive here,
                selected by joy_mapper_node):
                  linear.x  forward speed (m/s)    linear.y  left speed (m/s, crab walk)
                  angular.z yaw rate (rad/s, + = counter-clockwise)
  /drive_mode   std_msgs/String (latched): DOUBLE_ACKERMANN / DIFFERENTIAL / POINT_TURN / CRAB_WALK
  /crab_aim     std_msgs/Float64: crab wheel angle (rad, + = right). Only turns the wheels
                while there is no /cmd_vel vector; it can never drive the rover.

Output
  /drive_controller/rover  Float64MultiArray, 8 values, wheel order FL, FR, RL, RR:
                  [0:4] steering  (servo deg 0..180, 90 = straight; rad from center if
                        publish_servo_deg is false)
                  [4:8] wheel velocity in rpm (always)

Pure rotation: in DOUBLE_ACKERMANN, a command with ~no forward speed but a yaw rate (what
Nav2 sends to turn in place) is executed as a point turn at that yaw rate.

Units: /cmd_vel (m/s, rad/s) is converted to rpm once, at RPM_PER_MS. Wheel speeds, the
speed limit and the output are all rpm.

Parameters (see config/rover_params.yaml; shared with joy_mapper_node)
"""

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Float64, Float64MultiArray, String
from geometry_msgs.msg import Twist


def wrap_pi(a):
    """Wrap an angle to [-pi, pi]."""
    return math.atan2(math.sin(a), math.cos(a))


class DriveController(Node):
    def __init__(self):
        super().__init__('drive_controller')

        # --- Parameters ---
        self.declare_parameter('wheelbase', 1.032)            # m
        self.declare_parameter('track_width', 0.79)           # m
        self.declare_parameter('wheel_radius', 0.30)          # m
        self.declare_parameter('max_wheel_rpm', 100.0)        # rpm limit on every wheel
        self.declare_parameter('max_steer_rad', 0.5)          # Ackermann limit (~28.6 deg)
        self.declare_parameter('crab_max_angle_deg', 90.0)    # crab end stops
        self.declare_parameter('cmd_timeout', 0.5)            # s without /cmd_vel -> stop
        self.declare_parameter('turn_lin_eps', 0.01)          # m/s: below this = no forward speed
        self.declare_parameter('turn_ang_eps', 0.05)          # rad/s: above this = turning
        self.declare_parameter('publish_servo_deg', True)     # True: servo deg (90 = center)
        self.declare_parameter('command_topic', '/drive_controller/rover')

        p = self.get_parameter
        self.L = p('wheelbase').value
        self.W = p('track_width').value
        self.wheel_radius = p('wheel_radius').value
        self.MAX_WHEEL_RPM = p('max_wheel_rpm').value
        self.max_steer_rad = p('max_steer_rad').value
        self.CRAB_MAX_ANGLE_DEG = p('crab_max_angle_deg').value
        self.cmd_timeout = p('cmd_timeout').value
        self.TURN_LIN_EPS = p('turn_lin_eps').value
        self.TURN_ANG_EPS = p('turn_ang_eps').value
        self.PUBLISH_SERVO_DEG = p('publish_servo_deg').value
        self.COMMAND_TOPIC = p('command_topic').value

        # --- Derived limits ---
        self.control_dt = 0.02        # Control loop period (s)
        self.MIN_CMD = 1e-3           # Below this (m/s or rad/s) a command counts as zero
        # The one m/s -> rpm conversion: wheel rpm per 1 m/s at the rim.
        self.RPM_PER_MS = 60.0 / (2.0 * math.pi * self.wheel_radius)

        # Angle convention: servo frame 0..180, straight ahead = 90 (+/-90 deg of travel).
        self.STEER_RANGE_DEG = 90.0
        self.CENTER_ANGLE_DEG = self.STEER_RANGE_DEG
        self.SERVO_MIN_DEG = 0.0
        self.SERVO_MAX_DEG = 2.0 * self.STEER_RANGE_DEG

        # Crab walk: below this speed (m/s) the vector has no direction, so the last angle
        # is held (or a fresh /crab_aim is followed, without driving).
        self.CRAB_DIR_MIN = 1e-4
        self.AIM_TOPIC = '/crab_aim'
        self.AIM_TIMEOUT = 0.5
        self.aim_angle_rad = None
        self.aim_time = None

        # --- State ---
        self.modes = ['DOUBLE_ACKERMANN', 'DIFFERENTIAL', 'POINT_TURN', 'CRAB_WALK']
        self.current_mode = 'DOUBLE_ACKERMANN'

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.crab_angle_rad = 0.0   # commanded crab wheel angle (+ = right, held when stopped)
        self.last_cmd_time = self.get_clock().now()

        # --- ROS 2 interfaces ---
        mode_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self.create_subscription(Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.create_subscription(String, '/drive_mode', self.mode_callback, mode_qos)
        self.create_subscription(Float64, self.AIM_TOPIC, self.aim_callback, 10)
        self.cmd_pub = self.create_publisher(Float64MultiArray, self.COMMAND_TOPIC, 10)

        self.timer = self.create_timer(self.control_dt, self.control_loop)
        self.get_logger().info(
            f"Drive Controller active. Max wheel speed {self.MAX_WHEEL_RPM:.0f} rpm | "
            f"output: wheels in rpm, "
            f"steering in {'servo deg' if self.PUBLISH_SERVO_DEG else 'rad'}"
        )

    # ------------------------------------------------------------------
    def cmd_vel_callback(self, msg: Twist):
        self.cmd_vx = msg.linear.x
        self.cmd_vy = msg.linear.y
        self.cmd_wz = msg.angular.z
        self.last_cmd_time = self.get_clock().now()

    def mode_callback(self, msg: String):
        if msg.data in self.modes:
            self.current_mode = msg.data
            self.crab_angle_rad = 0.0
            self.get_logger().info(f"Switched Drive Mode -> {self.current_mode}")

    def aim_callback(self, msg: Float64):
        self.aim_angle_rad = msg.data
        self.aim_time = self.get_clock().now()

    def fresh_aim(self):
        """Latest /crab_aim angle (rad), or None if missing/stale."""
        if self.aim_angle_rad is None or self.aim_time is None:
            return None
        age = (self.get_clock().now() - self.aim_time).nanoseconds * 1e-9
        return self.aim_angle_rad if age <= self.AIM_TIMEOUT else None

    def map_angle_deg(self, delta_deg):
        """Offset from center -> [0, 180] servo range."""
        angle = self.CENTER_ANGLE_DEG + delta_deg
        return max(self.SERVO_MIN_DEG, min(self.SERVO_MAX_DEG, angle))

    def publish_commands(self, steer_angles_deg, v_wheels):
        """Publish ONE 8-element array: [steer FL,FR,RL,RR, vel FL,FR,RL,RR]."""
        if self.PUBLISH_SERVO_DEG:
            steer = [float(d) for d in steer_angles_deg]
        else:
            steer = [math.radians(d - self.CENTER_ANGLE_DEG) for d in steer_angles_deg]

        # v_wheels are rpm. Final hard limit on the published numbers.
        lim = self.MAX_WHEEL_RPM
        vel = [float(max(-lim, min(lim, v))) for v in v_wheels]

        msg = Float64MultiArray()
        msg.data = steer + vel
        self.cmd_pub.publish(msg)

    def point_turn(self, wz):
        """Spin in place at yaw rate wz -> (steering deltas in deg, wheel rpm)."""
        alpha_deg = math.degrees(math.atan2(self.L, self.W))
        radius = math.hypot(self.L / 2.0, self.W / 2.0)
        rpm = wz * radius * self.RPM_PER_MS
        return [alpha_deg, -alpha_deg, -alpha_deg, alpha_deg], [-rpm, rpm, -rpm, rpm]

    def stop(self):
        """Zero speed, wheels straight."""
        self.publish_commands([self.CENTER_ANGLE_DEG] * 4, [0.0] * 4)

    # ------------------------------------------------------------------
    def crab_from_twist(self, vx, vy):
        """(vx, vy) -> crab wheel angle (self.crab_angle_rad) and signed wheel speed.

        Wheels point along the velocity vector (angle + = right, 0 = forward). The same
        motion works at angle a with speed +v, or at a-180 deg with speed -v; the one
        inside the +/-CRAB_MAX_ANGLE_DEG end stops (closest to the current angle) is used.
        With no vector the wheels hold their angle, or follow a fresh /crab_aim without
        driving.
        """
        limit = math.radians(self.CRAB_MAX_ANGLE_DEG)
        v = math.hypot(vx, vy)
        if v < self.CRAB_DIR_MIN:
            aim = self.fresh_aim()
            if aim is not None:
                self.crab_angle_rad = max(-limit, min(limit, aim))
            return 0.0

        h = math.atan2(-vy, vx)
        alt = h - math.copysign(math.pi, h)
        cands = [(h, v), (alt, -v)]
        valid = [c for c in cands if abs(c[0]) <= limit + 1e-3]
        if not valid:
            valid = [min(cands, key=lambda c: abs(c[0]))]
        ang, spd = min(valid, key=lambda c: abs(wrap_pi(c[0] - self.crab_angle_rad)))
        self.crab_angle_rad = max(-limit, min(limit, ang))
        return spd

    # ------------------------------------------------------------------
    def control_loop(self):
        try:
            # Watchdog: stop if commands stop arriving
            age = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
            if age > self.cmd_timeout:
                self.cmd_vx = self.cmd_vy = self.cmd_wz = 0.0

            mode = self.current_mode
            vx = self.cmd_vx
            wz = self.cmd_wz

            v_wheels = [0.0] * 4
            delta_angles_deg = [0.0] * 4

            # --- 1. DOUBLE ACKERMANN ---
            if mode == 'DOUBLE_ACKERMANN':
                if abs(vx) < self.TURN_LIN_EPS and abs(wz) > self.TURN_ANG_EPS:
                    # Pure rotation (e.g. Nav2 turning in place): do a point turn.
                    delta_angles_deg, v_wheels = self.point_turn(wz)
                elif abs(vx) < self.MIN_CMD:
                    self.stop()
                    return
                else:
                    # Path curvature (+ = left) -> wheel angle (+ = right)
                    curvature = wz / vx
                    steer_input_rad = math.atan(-curvature * self.L / 2.0)
                    steer_input_rad = max(-self.max_steer_rad,
                                          min(self.max_steer_rad, steer_input_rad))
                    speed_input = vx * self.RPM_PER_MS      # rpm

                    if abs(steer_input_rad) < 1e-3:
                        v_wheels = [speed_input] * 4
                    else:
                        delta = abs(steer_input_rad)
                        R = (self.L / 2.0) / math.tan(delta)

                        delta_in = math.atan2(self.L / 2.0, R - (self.W / 2.0))
                        delta_out = math.atan2(self.L / 2.0, R + (self.W / 2.0))

                        R_in = math.hypot(R - (self.W / 2.0), self.L / 2.0)
                        R_out = math.hypot(R + (self.W / 2.0), self.L / 2.0)

                        # vx is the speed of the rover center (distance R from the ICR),
                        # so each wheel runs at vx * (its distance to ICR) / R.
                        v_in = speed_input * (R_in / R)
                        v_out = speed_input * (R_out / R)

                        d_in_deg = math.degrees(delta_in)
                        d_out_deg = math.degrees(delta_out)

                        if steer_input_rad > 0:  # Turning Right
                            delta_angles_deg = [d_out_deg, d_in_deg, -d_out_deg, -d_in_deg]
                            v_wheels = [v_out, v_in, v_out, v_in]
                        else:  # Turning Left
                            delta_angles_deg = [-d_in_deg, -d_out_deg, d_in_deg, d_out_deg]
                            v_wheels = [v_in, v_out, v_in, v_out]

            # --- 2. DIFFERENTIAL (skid steer: wheels straight, left/right speeds differ) ---
            elif mode == 'DIFFERENTIAL':
                if abs(vx) < self.MIN_CMD and abs(wz) < self.MIN_CMD:
                    self.stop()
                    return

                half_track = self.W / 2.0
                v_left = (vx - wz * half_track) * self.RPM_PER_MS
                v_right = (vx + wz * half_track) * self.RPM_PER_MS
                v_wheels = [v_left, v_right, v_left, v_right]

            # --- 3. POINT TURN (spin in place at yaw rate wz) ---
            elif mode == 'POINT_TURN':
                if abs(wz) < self.MIN_CMD:
                    self.stop()
                    return

                delta_angles_deg, v_wheels = self.point_turn(wz)

            # --- 4. CRAB WALK (wheels point along the (vx, vy) velocity vector) ---
            elif mode == 'CRAB_WALK':
                speed = self.crab_from_twist(vx, self.cmd_vy)

                delta_angles_deg = [math.degrees(self.crab_angle_rad)] * 4
                v_wheels = [speed * self.RPM_PER_MS] * 4

            steer_angles_deg = [self.map_angle_deg(d) for d in delta_angles_deg]

            # Scale so no wheel exceeds max_wheel_rpm (keeps the ratio between wheels)
            max_rpm = max(abs(v) for v in v_wheels)
            if max_rpm > self.MAX_WHEEL_RPM:
                scale_factor = self.MAX_WHEEL_RPM / max_rpm
                v_wheels = [v * scale_factor for v in v_wheels]

            self.publish_commands(steer_angles_deg, v_wheels)

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")
            try:
                self.stop()
            except Exception:
                pass


def main(args=None):
    rclpy.init(args=args)
    node = DriveController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        try:
            node.stop()          # last command out is always zero speed
        except Exception:
            pass
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()