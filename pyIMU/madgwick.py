"""
Pose and Motion Estimation with Madgwick Filter
Urs Utzinger, 2023
"""

import numpy as np
from   pyIMU.quaternion import Quaternion, Vector3D 
from   pyIMU.utilities import accelmag2q, accel2q
from   copy import copy
import math

TWOPI               = 2.0 * math.pi
IDENTITY_QUATERNION = Quaternion(1.0, 0.0, 0.0, 0.0)
VECTOR_ZERO         = Vector3D(0.0, 0.0, 0.0)
DEG2RAD             = math.pi / 180.0
RAD2DEG             = 180.0 / math.pi
EPSILON             = math.ldexp(1.0, -53)

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

    acc.normalize()                                            #          // 160-162
    # q.normalize() its normalized at the end 
    
    # Estimated orientation change from gyroscope
    qDot = 0.5 * (q * gyr)                                     # (eq. 12) // 150-153

    # Objective function                                       # (eq. 25)
    f = np.array([2.0*(q.x*q.z - q.w*q.y) - acc.x,
                  2.0*(q.w*q.x + q.y*q.z) - acc.y,
                  2.0*(0.5-q.x**2-q.y**2) - acc.z])

    if np.linalg.norm(f) > 0:
        # Jacobian                                             # (eq. 26)
        J = np.array([[-2.0*q.y,  2.0*q.z, -2.0*q.w, 2.0*q.x],
                      [ 2.0*q.x,  2.0*q.w,  2.0*q.z, 2.0*q.y],
                      [ 0.0,     -4.0*q.x, -4.0*q.y, 0.0    ]])
        
        # Sensitivity Matrix                                   # (eq. 34)
        gradient = J.T@f

        gradient = gradient / np.linalg.norm(gradient)         #           // 184-188

        # Update orientation change
        qDot = qDot - gain*gradient                            # (eq. 33) // 191-194
    
    # Update orientation
    q += qDot*dt                                               # (eq. 13) // 198-201
    q.normalize()                                              #          // 204-208 

    return q

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
    acc.normalize()
    mag.normalize()
    # q.normalize() its normalized at the end
    
    # Estimated orientation change from gyroscope
    qDot = 0.5 * (q * gyr)                                     # (eq. 12)
    
    # Rotate normalized magnetometer measurements
    h = q * mag * q.conjugate                                  # (eq. 45)
    bx = math.sqrt(h.x**2 + h.y**2)                            # (eq. 46)
    bz = h.z

    # Objective function                                       # (eq. 31)
    f = np.array([2.0*(q.x*q.z - q.w*q.y)                                         - acc.x,
                  2.0*(q.w*q.x + q.y*q.z)                                         - acc.y,
                  2.0*(0.5-q.x**2-q.y**2)                                         - acc.z,
                  2.0*bx*(0.5 - q.y**2 - q.z**2) + 2.0*bz*(q.x*q.z - q.w*q.y)     - mag.x,
                  2.0*bx*(q.x*q.y - q.w*q.z)     + 2.0*bz*(q.w*q.x + q.y*q.z)     - mag.y,
                  2.0*bx*(q.w*q.y + q.x*q.z)     + 2.0*bz*(0.5 - q.x**2 - q.y**2) - mag.z])

    if np.linalg.norm(f) > 0:
        # Jacobian                                             # eq. 32)
        J = np.array([[-2.0*q.y,               2.0*q.z,              -2.0*q.w,                2.0*q.x              ],
                      [ 2.0*q.x,               2.0*q.w,               2.0*q.z,                2.0*q.y              ],
                      [ 0.0,                  -4.0*q.x,              -4.0*q.y,                0.0                  ],
                      [-2.0*bz*q.y,            2.0*bz*q.z,           -4.0*bx*q.y-2.0*bz*q.w, -4.0*bx*q.z+2.0*bz*q.x],
                      [-2.0*bx*q.z+2.0*bz*q.x, 2.0*bx*q.y+2.0*bz*q.w, 2.0*bx*q.x+2.0*bz*q.z, -2.0*bx*q.w+2.0*bz*q.y],
                      [ 2.0*bx*q.y,            2.0*bx*q.z-4.0*bz*q.x, 2.0*bx*q.w-4.0*bz*q.y,  2.0*bx*q.x           ]])

        # Sensitivity Matrix
        gradient = J.T@f                                      # (eq. 34)
        gradient = gradient / np.linalg.norm(gradient)

        # Updated orientation change            
        qDot -= gain*gradient                                # (eq. 33)

    # Update orientation
    q += qDot*dt                                             # (eq. 13)
    q.normalize()
    return q

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
      frequency : float, default: 100.0; Sampling frequency in Hertz, or
      dt : float, default: 0.01;         Sampling step in seconds. Inverse of sampling frequency. Not required.
      gain : float,                      Filter gain. Defaults to 0.033 for IMU implementations, or to 0.041 for MARG implementations.
      gain_imu : float, default: 0.033;  Filter gain for IMU implementation.
      gain_marg : float, default: 0.041; Filter gain for MARG implementation.

    Example:
    >>> from pyIMU.madgwick import Madgwick
    >>> madgwick = Madgwick(frequency=150.0, gain=0.033)
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
        self.q: Quaternion         = IDENTITY_QUATERNION
        self.acc                   = None
        self.gyr                   = None
        self.mag                   = None
        self.frequency: float      = kwargs.get('frequency', 100.0)
        self.dt: float             = kwargs.get('dt', (1.0/self.frequency) if self.frequency else 0.01)
        self.gain_imu              = kwargs.get('gain_imu', 0.033)
        self.gain_marg             = kwargs.get('gain_marg', 0.041)
        
    def update(self, gyr: Vector3D, acc: Vector3D, mag: Vector3D = None, dt: float = -1) -> Quaternion:        
        """
        Estimate the pose quaternion.
        gyr : Vector3D of tri-axial Gyroscope in rad/s
        acc : Vector3D of tri-axial Accelerometer in m/s^2
        mag : Vector3D of tri-axial Magnetometer in nT, optional
        dt  : float, default: None, Time step, in seconds, between consecutive function calls.
        """
        self.gyr = copy(gyr)
        self.acc = copy(acc)
             
        if mag is None:
            # Compute with IMU architecture
            if (self.q is None) or (dt < 0):
                self.q = accel2q(self.acc) # estimate initial orientation
                self.q.normalize()
            else:
                self.q = updateIMU(self.q, self.gyr, self.acc, dt=dt, gain=self.gain_imu)

        else:
            # Compute with MARG architecture
            self.mag = copy(mag)
            if (self.q is None) or (dt < 0):
                self.q = accelmag2q(self.acc, self.mag)
                self.q.normalize()
            else:    
                self.q = updateMARG(self.q, self.gyr, self.acc, self.mag, dt=dt, gain=self.gain_marg)

        return self.q
        