#!/usr/bin/env python3

import math

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
from std_msgs.msg import Float64MultiArray, String
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
        self.deadzone = 0.15          # Stick drift filter threshold
        self.max_steer_rad = 0.5      # Ackermann limit (~28.6 deg)
        self.cmd_timeout = 0.5        # Seconds without /cmd_vel before stopping
        self.control_dt = 0.02        # Control loop period (s)

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

        # Crab walk: wheel swing speed (deg/s) at full axis[5] deflection. Tune to your servo.
        self.CRAB_STEER_RATE_DEG_S = 60.0
        # Rover steering feedback (Float64MultiArray, 4 wheels). Same units as the command:
        # radians from center, or servo-frame degrees if PUBLISH_SERVO_DEG is True.
        self.STEER_FEEDBACK_TOPIC = '/steering_controller/feedback'
        self.FEEDBACK_TIMEOUT = 0.5   # Seconds before feedback is considered stale
        # Crab walk end stops: wheel angle is capped at +/- this from straight ahead.
        self.CRAB_MAX_ANGLE_DEG = 90.0
        # False: publish steering as radians from center. True: servo-frame degrees.
        self.PUBLISH_SERVO_DEG = False

        # --- Differential drive tuning ---
        # Turn authority while driving forward/back. 1.0 = old behaviour (hard pivots
        # on diagonal stick), lower = wider, gentler arcs. At fwd = 0 (stick sideways)
        # the turn is always full authority, so the rover can still pivot in place.
        self.DIFF_TURN_GAIN = 0.5

        # --- Combined command topic ---
        # One Float64MultiArray, 8 elements, wheel order FL, FR, RL, RR:
        #   data[0:4] = steering angles   (rad from center, or servo deg if PUBLISH_SERVO_DEG)
        #   data[4:8] = wheel velocities  (rad/s)
        self.COMMAND_TOPIC = '/drive_controller/commands'

        # --- Inversion & Signal Conventions ---
        self.INVERT_LS_X = -1.0
        self.INVERT_LS_Y = 1.0
        self.INVERT_POINT_TURN_W = -1.0
        # Crab walk speed axis (angular.z). Use -1.0 if the axis runs the other way.
        self.INVERT_CRAB_SPEED = 1.0
        # True : speed axis is a trigger (rest = -1, full press = +1) -> mapped to 0..1.
        # False: speed axis is a stick resting at 0, only the positive side drives.
        self.CRAB_SPEED_IS_TRIGGER = True
        # Crab walk angle axis (axis[5] -> angular.x). Use -1.0 to reverse the turn direction.
        self.INVERT_CRAB_ANGLE = -1.0
        # False: angle axis is a stick resting at 0 (+ = turn right, - = turn left).
        # True : angle axis is a trigger (rest = -1, full = +1) -> mapped to 0..1, turns one way only.
        self.CRAB_ANGLE_IS_TRIGGER = False

        # --- State ---
        self.modes = ['DOUBLE_ACKERMANN', 'DIFFERENTIAL', 'POINT_TURN', 'CRAB_WALK']
        self.current_mode = 'DOUBLE_ACKERMANN'

        self.cmd_vx = 0.0
        self.cmd_vy = 0.0
        self.cmd_wz = 0.0
        self.cmd_received = False
        self.cmd_fresh = False
        self.last_steer_delta_deg = [0.0] * 4
        self.cmd_ax = 0.0           # crab angle axis (angular.x)
        self.crab_angle_rad = 0.0   # commanded crab wheel angle (held when axis released)
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
    def apply_deadzone(self, val):
        if abs(val) < self.deadzone:
            return 0.0
        sign = 1.0 if val > 0 else -1.0
        scaled = sign * (abs(val) - self.deadzone) / (1.0 - self.deadzone)
        return 0.0 if abs(scaled) < 1e-3 else scaled

    def cmd_vel_callback(self, msg: Twist):
        self.cmd_vx = msg.linear.x
        self.cmd_vy = msg.linear.y
        self.cmd_wz = msg.angular.z
        self.cmd_ax = msg.angular.x
        self.cmd_received = True
        self.last_cmd_time = self.get_clock().now()

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

    def mode_callback(self, msg: String):
        if msg.data in self.modes:
            self.current_mode = msg.data
            self.crab_angle_rad = 0.0
            self.get_logger().info(f"Switched Drive Mode -> {self.current_mode}")

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

        # m/s at the rim -> rad/s at the wheel
        vel = [float(v / self.wheel_radius) for v in v_wheels]

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

    def crab_update_angle(self):
        """Angle axis (axis[5]) -> crab wheel angle (rad, 0 = forward, + = right).

        Each cycle the target moves from the *measured* angle by
        deflection * rate * dt, so the wheels swing gradually and the speed of the
        swing scales with how far the axis is pushed. Released axis holds the angle.
        Falls back to integrating the last command if feedback is missing.
        """
        raw = self.INVERT_CRAB_ANGLE * self.cmd_ax
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

    def crab_speed_command(self):
        """Speed axis -> 0..max_speed. Zero if /cmd_vel is stale or never received."""
        if not (self.cmd_received and self.cmd_fresh):
            return 0.0

        raw = self.INVERT_CRAB_SPEED * self.cmd_wz
        if self.CRAB_SPEED_IS_TRIGGER:
            norm = max(0.0, min(1.0, (raw + 1.0) / 2.0))
        else:
            norm = max(0.0, min(1.0, raw))

        return self.apply_deadzone(norm) * self.max_speed

    # ------------------------------------------------------------------
    def control_loop(self):
        try:
            # Watchdog: stop if commands stop arriving
            age = (self.get_clock().now() - self.last_cmd_time).nanoseconds * 1e-9
            self.cmd_fresh = age <= self.cmd_timeout
            if not self.cmd_fresh:
                self.cmd_vx = self.cmd_vy = self.cmd_wz = self.cmd_ax = 0.0

            mode = self.current_mode

            fwd = self.apply_deadzone(self.cmd_vx)
            side = self.apply_deadzone(self.cmd_vy)

            # Yaw axis (angular.z) is the velocity scale, same as crab walk (0..max_speed)
            speed_scale = self.crab_speed_command()

            speed_input = fwd * speed_scale
            steer_x = self.INVERT_LS_X * side

            if mode == 'DOUBLE_ACKERMANN' and abs(speed_input) < 1e-4:
                self.stop()
                return
            if mode == 'DIFFERENTIAL' and (
                speed_scale < 1e-4 or abs(fwd) + abs(steer_x) < 1e-4
            ):
                self.stop()
                return

            v_wheels = [0.0] * 4
            delta_angles_deg = [0.0] * 4

            # --- 1. DOUBLE ACKERMANN ---
            if mode == 'DOUBLE_ACKERMANN':
                steer_ratio = max(-1.0, min(1.0, steer_x))
                steer_input_rad = steer_ratio * self.max_steer_rad

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

            # --- 2. DIFFERENTIAL (sector based) ---
            elif mode == 'DIFFERENTIAL':
                delta_angles_deg = [0.0, 0.0, 0.0, 0.0]

                # Stick angle: 0 = straight forward, + = right, +-180 = straight back
                theta_deg = math.degrees(math.atan2(steer_x, fwd))
                speed = speed_scale   # speed comes only from the speed axis

                if abs(theta_deg) <= 45.0:            # FRONT sector: straight forward
                    linear_vel, angular_vel = speed, 0.0
                elif abs(theta_deg) >= 135.0:         # BACK sector: straight backward
                    linear_vel, angular_vel = -speed, 0.0
                elif steer_x > 0:                     # RIGHT sector: spin right
                    linear_vel, angular_vel = 0.0, speed
                else:                                 # LEFT sector: spin left
                    linear_vel, angular_vel = 0.0, -speed

                v_left = linear_vel + angular_vel
                v_right = linear_vel - angular_vel
                v_wheels = [v_left, v_right, v_left, v_right]


            # --- 3. POINT TURN ---
            elif mode == 'POINT_TURN':
                raw_w = self.apply_deadzone(self.cmd_wz)

                if raw_w == 0.0:
                    self.stop()
                    return

                w_pt = self.INVERT_POINT_TURN_W * raw_w * self.max_speed
                alpha_deg = math.degrees(math.atan2(self.L, self.W))
                radius = math.hypot(self.L / 2.0, self.W / 2.0)

                delta_angles_deg = [alpha_deg, -alpha_deg, -alpha_deg, alpha_deg]
                v_wheels = [-w_pt * radius, w_pt * radius, -w_pt * radius, w_pt * radius]

            # --- 4. CRAB WALK (full 360: axis[5] swings the wheels, yaw sets speed) ---
            elif mode == 'CRAB_WALK':
                self.crab_update_angle()

                # Forward stick gives direction (+ forward / - reverse), yaw scales speed
                speed = fwd * speed_scale

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