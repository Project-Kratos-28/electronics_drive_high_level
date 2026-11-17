#!/usr/bin/env python3
"""Swerve drive controller.

Inputs
  /cmd_vel      geometry_msgs/Twist, STANDARD units (all modes):
                  linear.x  forward speed (m/s)    linear.y  left speed (m/s, crab walk)
                  angular.z yaw rate (rad/s, + = counter-clockwise)
  /drive_mode   std_msgs/String (latched): DOUBLE_ACKERMANN / DIFFERENTIAL / POINT_TURN / CRAB_WALK
  /crab_aim     std_msgs/Float64 (optional): crab wheel angle (rad, + = right). Only turns the
                wheels while there is no /cmd_vel vector; it can never drive the rover.
  /steering_controller/feedback   measured wheel angles (optional, used in crab walk)

Output
  /drive_controller/commands  [steer FL,FR,RL,RR, vel FL,FR,RL,RR]
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

        # --- Physical Dimensions (Meters) ---
        self.L = 1.032  # Wheelbase
        self.W = 0.79  # Track width
        self.wheel_radius = 0.30      # Wheel radius (m)

        # --- Control & Hardware Limits ---
        self.max_steer_rad = 0.5      # Ackermann limit (~28.6 deg)
        self.cmd_timeout = 0.5        # Seconds without input before stopping
        self.control_dt = 0.02        # Control loop period (s)
        self.MIN_CMD = 1e-3           # Below this (m/s or rad/s) a command counts as zero

        # Max wheel speed is defined in RPM; everything else is derived from it.
        #   100 rpm -> 100 * 2*pi / 60 = 10.472 rad/s -> * 0.30 m = 3.1416 m/s
        self.MAX_WHEEL_RPM = 100.0
        self.max_wheel_omega = self.MAX_WHEEL_RPM * 2.0 * math.pi / 60.0   # rad/s
        self.max_speed = self.max_wheel_omega * self.wheel_radius          # m/s (rim speed)

        # Angle convention (full-circle wheels): servo frame 0..360, straight ahead = 180.
        # Every mode computes a deflection from straight-ahead (-180..+180).
        self.STEER_RANGE_DEG = 180.0
        self.CENTER_ANGLE_DEG = self.STEER_RANGE_DEG
        self.SERVO_MIN_DEG = 0.0
        self.SERVO_MAX_DEG = 2.0 * self.STEER_RANGE_DEG

        # Crab walk: the wheel angle is the direction of the (vx, vy) vector. Below this
        # speed (m/s) there is no direction information, so the last angle is held (or a
        # fresh /crab_aim is followed, without driving).
        self.CRAB_DIR_MIN = 1e-4
        # Joystick pre-aim: wheel angle (rad from center) published by joy_mapper_node.
        self.AIM_TOPIC = '/crab_aim'
        self.AIM_TIMEOUT = 0.5        # Seconds before an aim command is considered stale
        self.aim_angle_rad = None
        self.aim_time = None
        # Rover steering feedback (Float64MultiArray, 4 wheels). Same units as the command:
        # radians from center, or servo-frame degrees if PUBLISH_SERVO_DEG is True.
        self.STEER_FEEDBACK_TOPIC = '/steering_controller/feedback'
        self.FEEDBACK_TIMEOUT = 0.5   # Seconds before feedback is considered stale
        # Crab walk end stops: wheel angle is capped at +/- this from straight ahead.
        self.CRAB_MAX_ANGLE_DEG = 90.0
        # False: publish steering as radians from center. True: servo-frame degrees.
        self.PUBLISH_SERVO_DEG = False
        # True : wheel velocities in rpm (command_splitter converts to rad/s for Gazebo).
        # False: wheel velocities in rad/s.
        self.PUBLISH_RPM = False

        # --- Topics ---
        # Combined command: One Float64MultiArray, 8 elements, wheel order FL, FR, RL, RR:
        #   data[0:4] = steering angles   (rad from center, or servo deg if PUBLISH_SERVO_DEG)
        #   data[4:8] = wheel velocities  (rad/s, or rpm if PUBLISH_RPM)
        self.COMMAND_TOPIC = '/drive_controller/commands'

        # --- State ---
        self.modes = ['DOUBLE_ACKERMANN', 'DIFFERENTIAL', 'POINT_TURN', 'CRAB_WALK']
        self.current_mode = 'DOUBLE_ACKERMANN'

        self.cmd_vx = 0.0           # m/s
        self.cmd_vy = 0.0           # m/s (+ = left; used in crab walk)
        self.cmd_wz = 0.0           # rad/s
        self.last_steer_delta_deg = [0.0] * 4
        self.crab_angle_rad = 0.0   # commanded crab wheel angle (+ = right, held when stopped)
        self.fb_angle_rad = None    # measured wheel angle (circular mean of 4 wheels)
        self.fb_time = None
        self.last_cmd_time = self.get_clock().now()

        # --- ROS 2 interfaces ---
        mode_qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.sub_cmd_vel = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10
        )
        self.sub_mode = self.create_subscription(
            String, '/drive_mode', self.mode_callback, mode_qos
        )
        self.sub_feedback = self.create_subscription(
            Float64MultiArray, self.STEER_FEEDBACK_TOPIC, self.feedback_callback, 10
        )
        self.sub_aim = self.create_subscription(
            Float64, self.AIM_TOPIC, self.aim_callback, 10
        )

        # Single publisher: [steer x4, velocity x4]
        self.cmd_pub = self.create_publisher(
            Float64MultiArray, self.COMMAND_TOPIC, 10
        )

        self.timer = self.create_timer(self.control_dt, self.control_loop)
        self.get_logger().info(
            f"Drive Controller Active. Max wheel speed: {self.MAX_WHEEL_RPM:.0f} rpm "
            f"= {self.max_wheel_omega:.3f} rad/s = {self.max_speed:.3f} m/s"
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
        """Offset from center -> [0, 360] servo range."""
        angle = self.CENTER_ANGLE_DEG + delta_deg
        return max(self.SERVO_MIN_DEG, min(self.SERVO_MAX_DEG, angle))

    def publish_commands(self, v_wheels, steer_angles_deg):
        """Publish ONE 8-element array: [steer FL,FR,RL,RR, vel FL,FR,RL,RR]."""
        self.last_steer_delta_deg = [deg - self.CENTER_ANGLE_DEG for deg in steer_angles_deg]

        if self.PUBLISH_SERVO_DEG:
            steer = [float(d) for d in steer_angles_deg]
        else:
            steer = [math.radians(d) for d in self.last_steer_delta_deg]

        # m/s at the rim -> rad/s at the wheel (or rpm if PUBLISH_RPM)
        unit = 60.0 / (2.0 * math.pi) if self.PUBLISH_RPM else 1.0
        vel = [float(v / self.wheel_radius * unit) for v in v_wheels]

        msg = Float64MultiArray()
        msg.data = steer + vel
        self.cmd_pub.publish(msg)

    def stop(self):
        self.publish_commands([0.0] * 4, [self.CENTER_ANGLE_DEG] * 4)

    # ------------------------------------------------------------------
    def measured_angle(self):
        """Latest wheel angle feedback (rad), or None if missing/stale."""
        if self.fb_angle_rad is None or self.fb_time is None:
            return None
        age = (self.get_clock().now() - self.fb_time).nanoseconds * 1e-9
        return self.fb_angle_rad if age <= self.FEEDBACK_TIMEOUT else None

    def crab_from_twist(self, vx, vy):
        """Standard (vx, vy) -> crab wheel angle (self.crab_angle_rad) and signed wheel speed.

        The wheels point along the velocity vector (angle + = right, 0 = forward). The
        same motion can be made with the wheels at a and speed +v, or at a-180 deg and
        speed -v, so the one inside the +/-CRAB_MAX_ANGLE_DEG end stops (closest to the
        current angle) is used. Reversing therefore keeps the wheels where they are and
        just drives backwards.

        Any non-zero vector is a real drive command, however slow. With no vector, the
        wheels hold their angle, or follow a fresh /crab_aim (joystick pre-aim) without
        driving.
        """
        limit = math.radians(self.CRAB_MAX_ANGLE_DEG)
        v = math.hypot(vx, vy)
        if v < self.CRAB_DIR_MIN:
            aim = self.fresh_aim()
            if aim is not None:
                self.crab_angle_rad = max(-limit, min(limit, aim))
            return 0.0                      # never moves without a velocity command

        h = math.atan2(-vy, vx)             # direction as a right-positive wheel angle
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
            vx = self.cmd_vx      # m/s
            wz = self.cmd_wz      # rad/s, + = counter-clockwise

            v_wheels = [0.0] * 4
            delta_angles_deg = [0.0] * 4

            # --- 1. DOUBLE ACKERMANN ---
            if mode == 'DOUBLE_ACKERMANN':
                if abs(vx) < self.MIN_CMD:
                    self.stop()
                    return

                # Path curvature (+ = left) -> wheel angle (+ = right)
                curvature = wz / vx
                steer_input_rad = math.atan(-curvature * self.L / 2.0)
                steer_input_rad = max(-self.max_steer_rad,
                                      min(self.max_steer_rad, steer_input_rad))
                speed_input = vx

                if abs(steer_input_rad) < 1e-3:
                    delta_angles_deg = [0.0, 0.0, 0.0, 0.0]
                    v_wheels = [speed_input] * 4
                else:
                    delta = abs(steer_input_rad)
                    R = (self.L / 2.0) / math.tan(delta)

                    delta_in = math.atan2(self.L / 2.0, R - (self.W / 2.0))
                    delta_out = math.atan2(self.L / 2.0, R + (self.W / 2.0))

                    R_c = math.hypot(R, self.L / 2.0)
                    R_in = math.hypot(R - (self.W / 2.0), self.L / 2.0)
                    R_out = math.hypot(R + (self.W / 2.0), self.L / 2.0)

                    v_in = speed_input * (R_in / R_c)
                    v_out = speed_input * (R_out / R_c)

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
                v_left = vx - wz * half_track
                v_right = vx + wz * half_track
                v_wheels = [v_left, v_right, v_left, v_right]

            # --- 3. POINT TURN (spin in place at yaw rate wz) ---
            elif mode == 'POINT_TURN':
                if abs(wz) < self.MIN_CMD:
                    self.stop()
                    return

                alpha_deg = math.degrees(math.atan2(self.L, self.W))
                radius = math.hypot(self.L / 2.0, self.W / 2.0)

                delta_angles_deg = [alpha_deg, -alpha_deg, -alpha_deg, alpha_deg]
                v_wheels = [-wz * radius, wz * radius, -wz * radius, wz * radius]

            # --- 4. CRAB WALK (wheels point along the (vx, vy) velocity vector) ---
            elif mode == 'CRAB_WALK':
                speed = self.crab_from_twist(vx, self.cmd_vy)

                # Drive only as far as the wheels have reached the target (no scrubbing
                # while turning). Needs feedback; without it the target is assumed reached.
                fb = self.measured_angle()
                if fb is not None:
                    remaining = wrap_pi(self.crab_angle_rad - fb)
                    speed *= max(0.0, math.cos(remaining))

                delta_angles_deg = [math.degrees(self.crab_angle_rad)] * 4
                v_wheels = [speed] * 4

            steer_angles_deg = [self.map_angle_deg(d) for d in delta_angles_deg]

            # Clamp so no wheel exceeds 100 rpm (keeps the ratio between wheels)
            max_calculated_v = max(abs(v) for v in v_wheels)
            if max_calculated_v > self.max_speed and max_calculated_v > 0:
                scale_factor = self.max_speed / max_calculated_v
                v_wheels = [v * scale_factor for v in v_wheels]

            self.publish_commands(v_wheels, steer_angles_deg)

        except Exception as e:
            self.get_logger().error(f"Control loop error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = DriveController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()