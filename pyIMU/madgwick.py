"""
Pose and Motion Estimation with Madgwick Filter
Urs Utzinger, 2023
GPT 5.2, 2026 speed optimizations
"""

from   pyIMU.quaternion import Quaternion, Vector3D
from   pyIMU.utilities import accelmag2q, accel2q
from   copy import copy
import math

try:
    from pyIMU import _mcore
    _HAS_MCORE = True
except Exception:
    _mcore = None
    _HAS_MCORE = False

TWOPI               = 2.0 * math.pi
IDENTITY_QUATERNION = Quaternion(1.0, 0.0, 0.0, 0.0)
VECTOR_ZERO         = Vector3D(0.0, 0.0, 0.0)
DEG2RAD             = math.pi / 180.0
RAD2DEG             = 180.0 / math.pi
EPSILON             = math.ldexp(1.0, -53)
GRAVITY             = 9.80665

def updateIMU(q: Quaternion, gyr: Vector3D, acc: Vector3D, dt: float, gain: float) -> Quaternion:
    """
    Quaternion Estimation with a Gyroscope and Accelerometer.
    q   : A-priori quaternion.
    gyr : Vector3D of tri-axial Gyroscope in rad/s
    acc : Vector3D of tri-axial Accelerometer in m/s^2
    dt  : float, default: None, Time step, in seconds, between consecutive Quaternions.
    Returns
    q   : Estimated quaternion.
    """

    # Delegates to inplace implementation while preserving non-mutating API.
    q_new = copy(q)
    return updateIMU_inplace(q_new, gyr, acc, dt, gain)

def updateMARG(q: Quaternion, gyr: Vector3D, acc: Vector3D, mag: Vector3D, dt: float, gain: float) -> Quaternion:
    """
    Quaternion Estimation with a Gyroscope, Accelerometer and Magnetometer.
    q   : A-priori quaternion.
    gyr : Vector3D of tri-axial Gyroscope in rad/s
    acc : Vector3D of tri-axial Accelerometer in m/s^2
    mag : Vector3D of tri-axial Magnetometer in nT
    dt  : float, default: None, Time step, in seconds, between consecutive Quaternions.
    Returns
    q : Estimated quaternion.
    """
    # Delegates to inplace implementation while preserving non-mutating API.
    q_new = copy(q)
    return updateMARG_inplace(q_new, gyr, acc, mag, dt, gain)


def updateIMU_inplace(q: Quaternion, gyr: Vector3D, acc: Vector3D, dt: float, gain: float) -> Quaternion:
    """
    In-place variant of updateIMU. Mutates and returns `q`.
    """
    acc.normalize()                                            #          // 160-162
    # q.normalize() its normalized at the end

    # Estimated orientation change from gyroscope              # (eq. 12)
    # Objective function                                       # (eq. 25)
    # Update orientation change                                # (eq. 33)
    # Update orientation                                       # (eq. 13)
    if _HAS_MCORE:
        q.w, q.x, q.y, q.z = _mcore.update_imu_step(
            q.w, q.x, q.y, q.z,
            gyr.x, gyr.y, gyr.z,
            acc.x, acc.y, acc.z,
            dt, gain
        )
    else:
        q.w, q.x, q.y, q.z = _updateIMU_python(
            q.w, q.x, q.y, q.z,
            gyr.x, gyr.y, gyr.z,
            acc.x, acc.y, acc.z,
            dt, gain
        )
    return q


def updateMARG_inplace(q: Quaternion, gyr: Vector3D, acc: Vector3D, mag: Vector3D, dt: float, gain: float) -> Quaternion:
    """
    In-place variant of updateMARG. Mutates and returns `q`.
    """
    acc.normalize()
    mag.normalize()
    # q.normalize() its normalized at the end

    # Estimated orientation change from gyroscope              # (eq. 12)
    # Rotate normalized magnetometer measurements              # (eq. 45), (eq. 46)
    # Objective function                                       # (eq. 31)
    # Updated orientation change                               # (eq. 33)
    # Update orientation                                       # (eq. 13)
    if _HAS_MCORE:
        q.w, q.x, q.y, q.z = _mcore.update_marg_step(
            q.w, q.x, q.y, q.z,
            gyr.x, gyr.y, gyr.z,
            acc.x, acc.y, acc.z,
            mag.x, mag.y, mag.z,
            dt, gain
        )
    else:
        q.w, q.x, q.y, q.z = _updateMARG_python(
            q.w, q.x, q.y, q.z,
            gyr.x, gyr.y, gyr.z,
            acc.x, acc.y, acc.z,
            mag.x, mag.y, mag.z,
            dt, gain
        )
    return q


def _updateIMU_python(
    qw: float, qx: float, qy: float, qz: float,
    gx: float, gy: float, gz: float,
    ax: float, ay: float, az: float,
    dt: float, gain: float
):
    # Estimated orientation change from gyroscope              # (eq. 12)
    qDotw = 0.5 * (-qx * gx - qy * gy - qz * gz)
    qDotx = 0.5 * ( qw * gx + qy * gz - qz * gy)
    qDoty = 0.5 * ( qw * gy - qx * gz + qz * gx)
    qDotz = 0.5 * ( qw * gz + qx * gy - qy * gx)

    # Objective function                                       # (eq. 25)
    f0 = 2.0 * (qx * qz - qw * qy) - ax
    f1 = 2.0 * (qw * qx + qy * qz) - ay
    f2 = 2.0 * (0.5 - qx * qx - qy * qy) - az

    if math.sqrt(f0 * f0 + f1 * f1 + f2 * f2) > 0.0:
        # Sensitivity matrix: gradient = J.T @ f              # (eq. 34)
        g0 = (-2.0 * qy) * f0 + (2.0 * qx) * f1
        g1 = ( 2.0 * qz) * f0 + (2.0 * qw) * f1 + (-4.0 * qx) * f2
        g2 = (-2.0 * qw) * f0 + (2.0 * qz) * f1 + (-4.0 * qy) * f2
        g3 = ( 2.0 * qx) * f0 + (2.0 * qy) * f1

        # keep normalization behavior consistent with previous implementation
        gnorm = math.sqrt(g0 * g0 + g1 * g1 + g2 * g2)
        if gnorm > 0.0:
            inv_gnorm = 1.0 / gnorm
            g0 *= inv_gnorm
            g1 *= inv_gnorm
            g2 *= inv_gnorm
            g3 *= inv_gnorm

            # Update orientation change                        # (eq. 33)
            qDotw -= gain * g0
            qDotx -= gain * g1
            qDoty -= gain * g2
            qDotz -= gain * g3

    # Update orientation                                       # (eq. 13)
    qw += qDotw * dt
    qx += qDotx * dt
    qy += qDoty * dt
    qz += qDotz * dt

    qnorm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if qnorm > 0.0:
        inv_qnorm = 1.0 / qnorm
        qw *= inv_qnorm
        qx *= inv_qnorm
        qy *= inv_qnorm
        qz *= inv_qnorm
    return qw, qx, qy, qz


def _updateMARG_python(
    qw: float, qx: float, qy: float, qz: float,
    gx: float, gy: float, gz: float,
    ax: float, ay: float, az: float,
    mx: float, my: float, mz: float,
    dt: float, gain: float
):
    # Estimated orientation change from gyroscope              # (eq. 12)
    qDotw = 0.5 * (-qx * gx - qy * gy - qz * gz)
    qDotx = 0.5 * ( qw * gx + qy * gz - qz * gy)
    qDoty = 0.5 * ( qw * gy - qx * gz + qz * gx)
    qDotz = 0.5 * ( qw * gz + qx * gy - qy * gx)

    # Rotate normalized magnetometer measurements
    # h = q * mag * q.conjugate                                 # (eq. 45)
    mw = -(qx * mx + qy * my + qz * mz)
    mx1 = qw * mx + qy * mz - qz * my
    my1 = qw * my - qx * mz + qz * mx
    mz1 = qw * mz + qx * my - qy * mx

    hx = -mw * qx + mx1 * qw - my1 * qz + mz1 * qy
    hy = -mw * qy + mx1 * qz + my1 * qw - mz1 * qx
    hz = -mw * qz - mx1 * qy + my1 * qx + mz1 * qw

    bx = math.sqrt(hx * hx + hy * hy)                          # (eq. 46)
    bz = hz

    # Objective function                                        # (eq. 31)
    f0 = 2.0 * (qx * qz - qw * qy)                                             - ax
    f1 = 2.0 * (qw * qx + qy * qz)                                             - ay
    f2 = 2.0 * (0.5 - qx * qx - qy * qy)                                       - az
    f3 = 2.0 * bx * (0.5 - qy * qy - qz * qz) + 2.0 * bz * (qx * qz - qw * qy) - mx
    f4 = 2.0 * bx * (qx * qy - qw * qz)       + 2.0 * bz * (qw * qx + qy * qz) - my
    f5 = 2.0 * bx * (qw * qy + qx * qz)       + 2.0 * bz * (0.5 - qx * qx - qy * qy) - mz

    if math.sqrt(f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3 + f4 * f4 + f5 * f5) > 0.0:
        # Sensitivity matrix: gradient = J.T @ f               # (eq. 34)
        g0 = (-2.0 * qy) * f0 + (2.0 * qx) * f1 + (-2.0 * bz * qy) * f3 + (-2.0 * bx * qz + 2.0 * bz * qx) * f4 + (2.0 * bx * qy) * f5
        g1 = (2.0 * qz) * f0 + (2.0 * qw) * f1 + (-4.0 * qx) * f2 + (2.0 * bz * qz) * f3 + (2.0 * bx * qy + 2.0 * bz * qw) * f4 + (2.0 * bx * qz - 4.0 * bz * qx) * f5
        g2 = (-2.0 * qw) * f0 + (2.0 * qz) * f1 + (-4.0 * qy) * f2 + (-4.0 * bx * qy - 2.0 * bz * qw) * f3 + (2.0 * bx * qx + 2.0 * bz * qz) * f4 + (2.0 * bx * qw - 4.0 * bz * qy) * f5
        g3 = (2.0 * qx) * f0 + (2.0 * qy) * f1 + (-4.0 * bx * qz + 2.0 * bz * qx) * f3 + (-2.0 * bx * qw + 2.0 * bz * qy) * f4 + (2.0 * bx * qx) * f5

        gnorm = math.sqrt(g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3)
        if gnorm > 0.0:
            inv_gnorm = 1.0 / gnorm
            g0 *= inv_gnorm
            g1 *= inv_gnorm
            g2 *= inv_gnorm
            g3 *= inv_gnorm

            # Updated orientation change                        # (eq. 33)
            qDotw -= gain * g0
            qDotx -= gain * g1
            qDoty -= gain * g2
            qDotz -= gain * g3

    # Update orientation                                        # (eq. 13)
    qw += qDotw * dt
    qx += qDotx * dt
    qy += qDoty * dt
    qz += qDotz * dt

    qnorm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if qnorm > 0.0:
        inv_qnorm = 1.0 / qnorm
        qw *= inv_qnorm
        qx *= inv_qnorm
        qy *= inv_qnorm
        qz *= inv_qnorm
    return qw, qx, qy, qz

class Madgwick:
    """
    Madgwick's Gradient Descent Pose Filter
    Earth Axis Convention: NED (North East Down)

    Methods:
    - update: 
      Update the filter with new measurements, calls
      updateIMU when using only gyroscope and accelerometer data (IMU implementation),
      updateMARG when using gyroscope, accelerometer and magnetometer data (MARG implementation).

    Initialization:
      dt : float, default: 0.01;         Sampling step in seconds. Inverse of sampling frequency. Not required.
      gain : float,                      Filter gain. Defaults to 0.033 for IMU implementations, or to 0.041 for MARG implementations.
      gain_imu : float, default: 0.033;  Filter gain for IMU implementation.
      gain_marg : float, default: 0.041; Filter gain for MARG implementation.

    Example:
    >>> from pyIMU.madgwick import Madgwick
    >>> madgwick = Madgwick(dt=1.0/150.0, gain=0.033)
    >>> madgwick = Madgwick(dt=1/150.0, gain_imu=0.033)
    >>> type(madgwick.q)

    >>> madgwick.update(gyr=gyro_data, acc=acc_data, dt=0.01)
    >>> madgwick.update(gyr=gyro_data, acc=acc_data, mag=mag_data, dt=0.01)

    Disclaimer:
    This code is based on https://github.com/Mayitzin/ahrs/
    The original paper and formula references for the Madgwick algorithm are 
    - https://x-io.co.uk/downloads/madgwick_internal_report.pdf
    The peer reviewed publication for the Madgwick algorithum is
    - https://doi.org/10.1109/ICORR.2011.5975346
    The original C++ implementation is
    - https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/
    There is newer work by the same author at: https://github.com/xioTechnologies/Fusion
    including magnetic and acceleration rejection and C to Python API.
    
    Urs Utzinger 2023
    """
    
    def __init__(self, **kwargs):        
        self.q                     = None
        self.acc                   = None
        self.gyr                   = None
        self.mag                   = None
        self.azero                 = Vector3D(0.0, 0.0, 0.0)   # gravity-removed acceleration (sensor frame, g)
        self.aglobal               = Vector3D(0.0, 0.0, 0.0)   # gravity-removed acceleration (earth frame, g)
        self.dt: float             = kwargs.get('dt', 0.01)
        gain                       = kwargs.get('gain', None)
        self.gain_imu              = kwargs.get('gain_imu', gain if gain is not None else 0.033)
        self.gain_marg             = kwargs.get('gain_marg', gain if gain is not None else 0.041)
        self.acc_in_g              = kwargs.get('acc_in_g', True)
        self.gyr_in_dps            = kwargs.get('gyr_in_dps', False)
        self.convention            = str(kwargs.get('convention', 'NED')).upper()
        if self.convention != 'NED':
            raise ValueError("Madgwick currently supports only NED convention.")

    def _to_acc_g(self, acc: Vector3D) -> Vector3D:
        if self.acc_in_g:
            return Vector3D(acc)
        return Vector3D(acc.x / GRAVITY, acc.y / GRAVITY, acc.z / GRAVITY)

    def _to_gyr_rads(self, gyr: Vector3D) -> Vector3D:
        if self.gyr_in_dps:
            return Vector3D(gyr.x * DEG2RAD, gyr.y * DEG2RAD, gyr.z * DEG2RAD)
        return Vector3D(gyr)

    def _gravity_sensor(self) -> Vector3D:
        q = self.q
        return Vector3D(
            2.0 * (q.x * q.z - q.w * q.y),
            2.0 * (q.w * q.x + q.y * q.z),
            q.w * q.w - q.x * q.x - q.y * q.y + q.z * q.z,
        )

    def _update_acc_outputs(self) -> None:
        if self.q is None or self.acc is None:
            self.azero.x = self.azero.y = self.azero.z = 0.0
            self.aglobal.x = self.aglobal.y = self.aglobal.z = 0.0
            return

        gravity_sensor = self._gravity_sensor()
        self.azero.x = self.acc.x - gravity_sensor.x
        self.azero.y = self.acc.y - gravity_sensor.y
        self.azero.z = self.acc.z - gravity_sensor.z
        aq = self.q.conjugate * Quaternion(self.azero) * self.q
        self.aglobal.x = aq.x
        self.aglobal.y = aq.y
        self.aglobal.z = aq.z
        
    def update(self, gyr: Vector3D, acc: Vector3D, mag: Vector3D = None, dt: float = -1) -> Quaternion:        
        """
        Estimate the pose quaternion.
        gyr : Vector3D of tri-axial Gyroscope in rad/s (or deg/s if gyr_in_dps=True)
        acc : Vector3D of tri-axial Accelerometer in g (or m/s^2 if acc_in_g=False)
        mag : Vector3D of tri-axial Magnetometer in nT, optional
        dt  : float, default: None, Time step, in seconds, between consecutive function calls.
        """
        if dt <= 0:
            dt = self.dt

        if self.gyr is None:
            self.gyr = Vector3D(0.0, 0.0, 0.0)
        if self.acc is None:
            self.acc = Vector3D(0.0, 0.0, 0.0)

        if self.gyr_in_dps:
            self.gyr.x = gyr.x * DEG2RAD
            self.gyr.y = gyr.y * DEG2RAD
            self.gyr.z = gyr.z * DEG2RAD
        else:
            self.gyr.x = gyr.x
            self.gyr.y = gyr.y
            self.gyr.z = gyr.z

        if self.acc_in_g:
            self.acc.x = acc.x
            self.acc.y = acc.y
            self.acc.z = acc.z
        else:
            self.acc.x = acc.x / GRAVITY
            self.acc.y = acc.y / GRAVITY
            self.acc.z = acc.z / GRAVITY
             
        if mag is None:
            # Compute with IMU architecture
            if self.q is None:
                # We run this the first time. Estimate initial quaternion.
                #   Nake sure that you have stable readings from the senor before
                #   calling this function, otherwise it takes a while for the sensor to orient.
                self.q = accel2q(self.acc) # estimate initial orientation
                # self.q.normalize()
                # print('Init with acc only: ', q2rpy(self.q))
            else:
                updateIMU_inplace(self.q, self.gyr, self.acc, dt=dt, gain=self.gain_imu)

        else:
            # Compute with MARG architecture
            if self.mag is None:
                self.mag = Vector3D(0.0, 0.0, 0.0)
            self.mag.x = mag.x
            self.mag.y = mag.y
            self.mag.z = mag.z
            if self.q is None:
                # We run this the first time. Estimate initial quaternion.
                #   Nake sure that you have stable readings from the senor before
                #   calling this function, otherwise it will take a while for sensor to orient.        
                self.q = accelmag2q(self.acc, self.mag)
                # self.q.normalize()
                # print('Init with acc and mag: ', q2rpy(self.q))
            else:    
                updateMARG_inplace(self.q, self.gyr, self.acc, self.mag, dt=dt, gain=self.gain_marg)

        self._update_acc_outputs()
        return self.q
        
