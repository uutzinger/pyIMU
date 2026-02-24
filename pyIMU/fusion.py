"""
Fusion-style AHRS based on Madgwick Chapter 7 concepts and xioTechnologies/Fusion behavior.

Earth convention: NED (x: North, y: East, z: Down)
"""

from copy import copy
import math

from pyIMU.quaternion import Quaternion, Vector3D
from pyIMU.utilities import accel2q, accelmag2q, clamp, clip
try:
    from pyIMU import _fcore
    _HAS_FCORE = True
except Exception:
    _fcore = None
    _HAS_FCORE = False

DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
GRAVITY = 9.80665
EPSILON = math.ldexp(1.0, -53)


class Fusion:
    """
    AHRS filter with gain ramp, gyro-bias estimation, and sensor rejection/recovery.

    Compatibility aliases:
    - `k_init`, `k_normal`, `t_init`
    - `fc_bias`, `omega_min`, `t_bias`
    - `g_deviation`, `mag_min`, `mag_max`
    """

    def __init__(self, **kwargs):
        # Sampling
        self.dt = float(kwargs.get("dt", 0.01))

        # Gains / initialisation ramp
        self.k_init = float(kwargs.get("k_init", 10.0))
        self.k_normal = float(kwargs.get("k_normal", kwargs.get("gain", 0.5)))
        self.t_init = float(kwargs.get("t_init", 3.0))
        self._gain_ramp_rate = (self.k_init - self.k_normal) / self.t_init if self.t_init > 0.0 else 0.0
        self.ramped_gain = self.k_init
        self.initialising = True

        # Gyroscope units/range for angular-rate recovery
        self.gyr_in_dps = bool(kwargs.get("gyr_in_dps", False))
        self.gyr_range = float(kwargs.get("gyr_range", 2000.0))
        self._gyr_range_rad = self.gyr_range * (DEG2RAD if self.gyr_in_dps else 1.0)

        # Bias estimator (stationary-gated LPF)
        self.fc_bias = float(kwargs.get("fc_bias", 0.05))
        self.omega_min = float(kwargs.get("omega_min", 0.35))
        self.t_bias = float(kwargs.get("t_bias", 1.0))
        self._stationary_time = 0.0

        # Sensor input units
        self.acc_in_g = bool(kwargs.get("acc_in_g", True))

        # Rejection thresholds
        self.g_deviation = float(kwargs.get("g_deviation", 0.10))
        self.acceleration_rejection = float(kwargs.get("acceleration_rejection", 12.0)) * DEG2RAD
        self.magnetic_rejection = float(kwargs.get("magnetic_rejection", 12.0)) * DEG2RAD
        self.mag_min = kwargs.get("mag_min", None)
        self.mag_max = kwargs.get("mag_max", None)

        # Recovery trigger periods (seconds)
        self.t_acc_reject = float(kwargs.get("t_acc_reject", 0.5))
        self.t_mag_reject = float(kwargs.get("t_mag_reject", 0.5))
        self.recovery_timeout_factor = int(kwargs.get("recovery_timeout_factor", 5))

        self._acc_trigger = 0
        self._acc_timeout = 0
        self._mag_trigger = 0
        self._mag_timeout = 0

        # State
        self.q = Quaternion(1.0, 0.0, 0.0, 0.0)
        self.gyro_bias = Vector3D(0.0, 0.0, 0.0)

        self.angular_rate_recovery = False
        self.accelerometer_ignored = False
        self.magnetometer_ignored = True
        self.acceleration_error = 0.0
        self.magnetic_error = 0.0

        # Optional outputs
        self.azero = Vector3D(0.0, 0.0, 0.0)   # acceleration with gravity removed (sensor frame)
        self.aglobal = Vector3D(0.0, 0.0, 0.0) # acceleration in Earth frame

        convention = kwargs.get("convention", "NED")
        if convention.upper() != "NED":
            raise ValueError("Fusion currently supports only NED convention.")

    def _to_acc_g(self, acc: Vector3D) -> Vector3D:
        if self.acc_in_g:
            return Vector3D(acc)
        return Vector3D(acc.x / GRAVITY, acc.y / GRAVITY, acc.z / GRAVITY)

    def _to_gyr_rads(self, gyr: Vector3D) -> Vector3D:
        if self.gyr_in_dps:
            return Vector3D(gyr.x * DEG2RAD, gyr.y * DEG2RAD, gyr.z * DEG2RAD)
        return Vector3D(gyr)

    def _half_gravity(self) -> Vector3D:
        q = self.q
        return Vector3D(
            q.x * q.z - q.w * q.y,
            q.w * q.x + q.y * q.z,
            q.w * q.w - 0.5 + q.z * q.z,
        )

    def _gravity_sensor(self) -> Vector3D:
        q = self.q
        return Vector3D(
            2.0 * (q.x * q.z - q.w * q.y),
            2.0 * (q.w * q.x + q.y * q.z),
            q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
        )

    def _update_bias(self, gyr: Vector3D, acc_mag_error: float, dt: float) -> None:
        is_stationary = (gyr.norm <= self.omega_min) and (acc_mag_error <= self.g_deviation)
        if is_stationary:
            self._stationary_time += dt
        else:
            self._stationary_time = 0.0

        if self._stationary_time >= self.t_bias:
            alpha = clamp(2.0 * math.pi * self.fc_bias * dt, 0.0, 1.0)
            self.gyro_bias.x += alpha * (gyr.x - self.gyro_bias.x)
            self.gyro_bias.y += alpha * (gyr.y - self.gyro_bias.y)
            self.gyro_bias.z += alpha * (gyr.z - self.gyro_bias.z)

    def _apply_recovery(self, accepted: bool, initialising: bool, trigger: int, timeout: int, trigger_period: int):
        timeout_period = self.recovery_timeout_factor * trigger_period

        if initialising or accepted:
            trigger -= 9
            ignored = False
        else:
            trigger += 1
            ignored = True

        if trigger > trigger_period:
            timeout = 0
            ignored = False
        elif timeout > timeout_period:
            trigger = 0
            ignored = False
        else:
            timeout += 1

        if not initialising:
            trigger = int(clip(trigger, trigger_period))
            timeout = int(clip(timeout, timeout_period))

        return ignored, trigger, timeout

    def update(self, gyr: Vector3D, acc: Vector3D, mag: Vector3D = None, dt: float = -1.0) -> Quaternion:
        """Compatibility wrapper: update state and return a copy of quaternion."""
        self.update_inplace(gyr=gyr, acc=acc, mag=mag, dt=dt)
        return copy(self.q)

    def update_inplace(self, gyr: Vector3D, acc: Vector3D, mag: Vector3D = None, dt: float = -1.0) -> Quaternion:
        """Update filter state in place and return the internal quaternion reference."""
        if dt <= 0.0:
            dt = self.dt

        gyr_rads = self._to_gyr_rads(gyr)
        acc_g = self._to_acc_g(acc)
        mag_present = mag is not None
        if mag_present:
            mx, my, mz = mag.x, mag.y, mag.z
        else:
            mx = my = mz = 0.0

        if self.q is None:
            if not mag_present:
                self.q = accel2q(acc_g)
            else:
                self.q = accelmag2q(acc_g, mag)

        if _HAS_FCORE:
            (
                self.q.w, self.q.x, self.q.y, self.q.z,
                self.gyro_bias.x, self.gyro_bias.y, self.gyro_bias.z,
                self._acc_trigger, self._acc_timeout, self._mag_trigger, self._mag_timeout,
                self._stationary_time, self.ramped_gain, initialising,
                angular_rate_recovery, accelerometer_ignored, magnetometer_ignored,
                self.acceleration_error, self.magnetic_error,
                self.azero.x, self.azero.y, self.azero.z,
                self.aglobal.x, self.aglobal.y, self.aglobal.z
            ) = _fcore.fusion_step(
                self.q.w, self.q.x, self.q.y, self.q.z,
                gyr_rads.x, gyr_rads.y, gyr_rads.z,
                acc_g.x, acc_g.y, acc_g.z,
                mag_present, mx, my, mz,
                self.gyro_bias.x, self.gyro_bias.y, self.gyro_bias.z,
                dt,
                self.k_normal,
                self.ramped_gain,
                self._gain_ramp_rate,
                self.initialising,
                self._gyr_range_rad,
                self.omega_min,
                self.fc_bias,
                self.t_bias,
                self.g_deviation,
                self.acceleration_rejection,
                self.magnetic_rejection,
                self.mag_min is not None,
                0.0 if self.mag_min is None else float(self.mag_min),
                self.mag_max is not None,
                0.0 if self.mag_max is None else float(self.mag_max),
                self.t_acc_reject,
                self.t_mag_reject,
                self.recovery_timeout_factor,
                self._acc_trigger,
                self._acc_timeout,
                self._mag_trigger,
                self._mag_timeout,
                self._stationary_time
            )
            self.initialising = bool(initialising)
            self.angular_rate_recovery = bool(angular_rate_recovery)
            self.accelerometer_ignored = bool(accelerometer_ignored)
            self.magnetometer_ignored = bool(magnetometer_ignored)
            return self.q

        if self.initialising and self._gain_ramp_rate > 0.0:
            self.ramped_gain -= self._gain_ramp_rate * dt
            if self.ramped_gain <= self.k_normal:
                self.ramped_gain = self.k_normal
                self.initialising = False

        rate_limit = 0.98 * self._gyr_range_rad
        self.angular_rate_recovery = (
            abs(gyr_rads.x) > rate_limit
            or abs(gyr_rads.y) > rate_limit
            or abs(gyr_rads.z) > rate_limit
        )

        feedback = Vector3D(0.0, 0.0, 0.0)
        half_gravity = self._half_gravity()

        acc_norm = acc_g.norm
        acc_mag_error = abs(acc_norm - 1.0)

        if acc_norm > EPSILON and not self.angular_rate_recovery:
            acc_unit = acc_g / acc_norm
            half_acc_feedback = acc_unit.cross(half_gravity)
            self.acceleration_error = math.asin(clamp(2.0 * half_acc_feedback.norm, -1.0, 1.0))

            acc_accepted = (acc_mag_error <= self.g_deviation) and (self.acceleration_error <= self.acceleration_rejection)
            self.accelerometer_ignored, self._acc_trigger, self._acc_timeout = self._apply_recovery(
                accepted=acc_accepted,
                initialising=self.initialising,
                trigger=self._acc_trigger,
                timeout=self._acc_timeout,
                trigger_period=max(1, int(round(self.t_acc_reject / dt))),
            )
            if not self.accelerometer_ignored:
                feedback = feedback + half_acc_feedback
        else:
            self.accelerometer_ignored = True
            self.acceleration_error = 0.0

        self.magnetometer_ignored = True
        self.magnetic_error = 0.0
        if mag_present and not self.angular_rate_recovery:
            mag_vec = Vector3D(mag)
            mag_norm_raw = mag_vec.norm
            if mag_norm_raw > EPSILON:
                mag_unit = mag_vec / mag_norm_raw

                # Estimate Earth-frame magnetic vector and project horizontal component to North axis.
                h = self.q.conjugate * Quaternion(mag_unit) * self.q
                bx = math.sqrt(h.x * h.x + h.y * h.y)
                bz = h.z
                ref_sensor = Vector3D(
                    2.0 * bx * (0.5 - self.q.y * self.q.y - self.q.z * self.q.z) + 2.0 * bz * (self.q.x * self.q.z - self.q.w * self.q.y),
                    2.0 * bx * (self.q.x * self.q.y - self.q.w * self.q.z) + 2.0 * bz * (self.q.w * self.q.x + self.q.y * self.q.z),
                    2.0 * bx * (self.q.w * self.q.y + self.q.x * self.q.z) + 2.0 * bz * (0.5 - self.q.x * self.q.x - self.q.y * self.q.y),
                )
                ref_norm = ref_sensor.norm
                if ref_norm > EPSILON:
                    ref_sensor = ref_sensor / ref_norm

                half_mag_feedback = mag_unit.cross(ref_sensor)
                self.magnetic_error = math.asin(clamp(2.0 * half_mag_feedback.norm, -1.0, 1.0))

                mag_strength_ok = True
                if self.mag_min is not None and mag_norm_raw < self.mag_min:
                    mag_strength_ok = False
                if self.mag_max is not None and mag_norm_raw > self.mag_max:
                    mag_strength_ok = False

                mag_accepted = mag_strength_ok and (self.magnetic_error <= self.magnetic_rejection)
                self.magnetometer_ignored, self._mag_trigger, self._mag_timeout = self._apply_recovery(
                    accepted=mag_accepted,
                    initialising=self.initialising,
                    trigger=self._mag_trigger,
                    timeout=self._mag_timeout,
                    trigger_period=max(1, int(round(self.t_mag_reject / dt))),
                )

                if not self.magnetometer_ignored:
                    feedback = feedback + half_mag_feedback

        if self.angular_rate_recovery:
            self.accelerometer_ignored = True
            self.magnetometer_ignored = True

        self._update_bias(gyr_rads, acc_mag_error, dt)

        gain = self.ramped_gain if self.initialising else self.k_normal

        if _HAS_FCORE:
            self.q.w, self.q.x, self.q.y, self.q.z = _fcore.integrate_quaternion_step(
                self.q.w, self.q.x, self.q.y, self.q.z,
                gyr_rads.x, gyr_rads.y, gyr_rads.z,
                self.gyro_bias.x, self.gyro_bias.y, self.gyro_bias.z,
                feedback.x, feedback.y, feedback.z,
                gain, dt
            )
        else:
            half_gyr = Vector3D(
                0.5 * (gyr_rads.x - self.gyro_bias.x) + gain * feedback.x,
                0.5 * (gyr_rads.y - self.gyro_bias.y) + gain * feedback.y,
                0.5 * (gyr_rads.z - self.gyro_bias.z) + gain * feedback.z,
            )

            qw0, qx0, qy0, qz0 = self.q.w, self.q.x, self.q.y, self.q.z
            qdw = (-qx0 * half_gyr.x - qy0 * half_gyr.y - qz0 * half_gyr.z)
            qdx = ( qw0 * half_gyr.x + qy0 * half_gyr.z - qz0 * half_gyr.y)
            qdy = ( qw0 * half_gyr.y - qx0 * half_gyr.z + qz0 * half_gyr.x)
            qdz = ( qw0 * half_gyr.z + qx0 * half_gyr.y - qy0 * half_gyr.x)

            self.q.w = qw0 + qdw * dt
            self.q.x = qx0 + qdx * dt
            self.q.y = qy0 + qdy * dt
            self.q.z = qz0 + qdz * dt
            self.q.normalize()

        # Optional acceleration outputs (in g)
        gravity_sensor = self._gravity_sensor()
        self.azero.x = acc_g.x - gravity_sensor.x
        self.azero.y = acc_g.y - gravity_sensor.y
        self.azero.z = acc_g.z - gravity_sensor.z
        aq = self.q.conjugate * Quaternion(self.azero) * self.q
        self.aglobal.x = aq.x
        self.aglobal.y = aq.y
        self.aglobal.z = aq.z

        return self.q
