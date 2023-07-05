from collections import deque
from pyIMU.quaternion import Vector3D, Quaternion, TWOPI, DEG2RAD
import numpy as np
import math
import struct
from copy import copy

###########################################################
# Utility Functions
###########################################################

def clip(val, largest):
    ''' 
    Clip val to [0,largest]
    '''
    return 0  if val < 0 else largest if val > largest else val 

def clamp(val, smallest, largest): 
    '''
    Clip val to [smallest, largest]
    '''
    if val < smallest: return smallest
    if val > largest: return largest
    return val

def asin(value: float):
    if value <=-1.0:
        return -math.pi/2
    elif value >= 1.0:
        return math.pi/2
    return math.asin(value)

def invSqrt(value: float) -> float:
    '''
    Fast inverse square root: 1/sqrt(x)
    Kahan algorithm does not improve speed on modern CPUs as they have sqrt in the instruction set.

    Use value**-0.5 instead
    '''

    # https://pizer.wordpress.com/2008/10/12/fast-inverse-square-root/
    threehalfs = float(1.5)
    x2 = value * float(0.5)
    y = value
    
    packed_y = struct.pack('f', y)       
    i = struct.unpack('i', packed_y)[0]  # treat float's bytes as int 
    i = 0x5f3759df - (i >> 1)            # arithmetic with magic number
    packed_i = struct.pack('i', i)
    y = struct.unpack('f', packed_i)[0]  # treat int's bytes as float
    
    y = y * (threehalfs - (x2 * y * y))  # Newton's method
    return y

class RunningAverage:
    '''
    Running Average with Variance
    with help form chat.openai.com
    '''
    
    def __init__(self, window_size):
        self.window_size = window_size
        if self.window_size <= 0: raise ValueError("Window size must be greater than zero.")
        self.window = deque(maxlen=window_size)
        self.sum = Vector3D(x=0.,y=0.,z=0.)
        self.squared_sum = Vector3D(x=0.,y=0.,z=0.)

    def update(self,value):
        self.len = len(self.window)
        if self.len < self.window_size:
            self.window.append(value)
            self.sum = self.sum + value
            # self.avg = self.sum / len(self.window)
            self.squared_sum = self.squared_sum + (value ** 2)
        else:
            old_value = self.window.popleft()
            self.window.append(value)
            self.sum = self.sum + value - old_value
            self.squared_sum = self.squared_sum + (value ** 2) - (old_value ** 2)
            
    @property
    def avg(self) -> float:
        return self.sum / self.len

    @property
    def var(self) -> float:
        return (self.squared_sum - (self.sum ** 2) / self.len) / self.len


def vector_angle2q(vec: Vector3D, angle: float = 0.0) -> Quaternion:
    '''Create quaternion based on rotation around vector'''
    sinHalfTheta = math.sin(angle * 0.5)
    return Quaternion(
        w = math.cos(angle * 0.5), 
        x = vec.x * sinHalfTheta, 
        y = vec.y * sinHalfTheta, 
        z = vec.z * sinHalfTheta)

def q2rpy(pose: Quaternion) -> Vector3D:
    '''quaternion to roll pitch yaw'''

    # roll (x-axis rotation)
    sinr_cosp =       2.0 * (pose.w * pose.x + pose.y * pose.z)
    cosr_cosp = 1.0 - 2.0 * (pose.x**2 + pose.y**2)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    # pitch (y-axis rotation)
    sinp      =       2.0 * (pose.w * pose.y - pose.x * pose.z)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2.0, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
    # yaw (z-axis rotation)
    siny_cosp =       2.0 * (pose.w * pose.z + pose.x * pose.y)
    cosy_cosp = 1.0 - 2.0 * (pose.y**2 + pose.z**2)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return Vector3D(x=roll, y=pitch, z=yaw)

def rpy2q(rpy) -> Quaternion:
    '''assume vector contains roll, pitch, yaw and convert to quaternion'''
    
    if isinstance(rpy, Vector3D):
        roll  = rpy.x 
        pitch = rpy.y
        yaw   = rpy.z
    elif isinstance(r, np.ndarray):
        if len(rpy) == 3:
            roll, pitch, yaw = r
    else:
        raise TypeError("Unsupported operand type for rpy2q: {}".format(type(r)))
    
    cosY2 = math.cos(yaw   * 0.5)
    sinY2 = math.sin(yaw   * 0.5)
    cosP2 = math.cos(pitch * 0.5)
    sinP2 = math.sin(pitch * 0.5)
    cosR2 = math.cos(roll  * 0.5)
    sinR2 = math.sin(roll  * 0.5)

    w = cosY2 * cosP2 * cosR2 + sinY2 * sinP2 * sinR2
    x = cosY2 * cosP2 * sinR2 - sinY2 * sinP2 * cosR2
    y = sinY2 * cosP2 * sinR2 + cosY2 * sinP2 * cosR2
    z = sinY2 * cosP2 * cosR2 - cosY2 * sinP2 * sinR2

    return Quaternion(w, x, y, z)

def accel2rpy(acc) -> Vector3D:

    if isinstance(acc, Vector3D):
        _acc = copy(acc)
    elif isinstance(acc, np.ndarray):
        if len(acc) == 3:
            _acc = Vector3D(acc)
    else:
        raise TypeError("Unsupported operand type for accel2rpy: {}".format(type(x)))

    _acc.normalize()

    roll  =  math.atan2(_acc.y, _acc.z)
    pitch = -math.atan2(_acc.x, math.sqrt(_acc.y * _acc.y + _acc.z * _acc.z))
    yaw   =  0.0
    return(Vector3D(roll, pitch, yaw))

def accel2q(acc) -> Quaternion:
    '''
    Converts Accelerometer to Quaternion assuming no motion
    Input accelerometer reading x,y,z
    Output quaternion w,x,y,z
    '''
    # # vec_z = Vector3D(0, 0, 1.0)
    # # angle = math.acos(z.dot(normAccel)) simplified to
    # angle = math.acos(_acc.z)
    # # vec = _acc.cross(vec_z) simplified to
    # vec = Vector3D(x=_acc.y, y=-_acc.x, z=0.)
    # return vec.angle2q(angle)
    if isinstance(acc, Vector3D):
        _acc = copy(acc) 
    elif isinstance(acc, np.ndarray):
        if len(acc) == 3:
            _acc = Vector3D(acc)
    else:
        raise TypeError("Unsupported operand type for accel2rpy: {}".format(type(acc)))

    _acc.normalize()
    roll  =  math.atan2(_acc.y, _acc.z)
    pitch = -math.atan2(_acc.x, math.sqrt(_acc.y * _acc.y + _acc.z * _acc.z))
    # yaw   =  0.0

    cosY2 = math.cos(pitch * 0.5)
    sinY2 = math.sin(pitch * 0.5)
    cosX2 = math.cos(roll  * 0.5)
    sinX2 = math.sin(roll  * 0.5)

    w =  cosY2 * cosX2 
    x =  cosY2 * sinX2
    y =  sinY2 * cosX2
    z = -sinY2 * sinX2

    return Quaternion(w=w, x=x, y=y, z=z)
    
def accelmag2q(acc, mag) -> Quaternion:
    '''
    Estimate Pose Vector from Accelerometer and Compass
    1) Acceleration to Roll Pitch Yaw=0.0
    2) Convert RPY to estimated Pose Quaternion
    3) Rotate Compass to World from estimated Pose
    4) Update estimated Yaw in Euler and return result
    '''

    if isinstance(acc, Vector3D):
        _acc = copy(acc)
    elif isinstance(acc, np.ndarray):
        if len(acc) == 3:
            _acc = Vector3D(acc)
    else:
        raise TypeError("Unsupported operand type for accel2rpy: {}".format(type(acc)))

    if isinstance(mag, Vector3D):
        _mag = copy(mag) 
    elif isinstance(mag, np.ndarray):
        if len(mag) == 3:
            _mag = Vector3D(mag)
    else:
        raise TypeError("Unsupported operand type for accel2rpy: {}".format(type(mag)))
            
    rpy = accel2rpy(_acc) # resutls rpy.z = 0

    # rpy to quaternion, simplified because rpy.z is zero
    cosP2 = math.cos(rpy.y * 0.5) # P
    sinP2 = math.sin(rpy.y * 0.5) # P
    cosR2 = math.cos(rpy.x * 0.5) # R
    sinR2 = math.sin(rpy.x * 0.5) # R
    q = Quaternion(
        w =  cosP2 * cosR2,
        x =  cosP2 * sinR2, 
        y =  sinP2 * cosR2,
        z = -sinP2 * sinR2
    )
    
    # rotate Magnetometer to Q pose
    m = Quaternion(
        w = 0.,
        x = _mag.x,
        y = _mag.y,
        z = _mag.z
    )
    m = q * m * q.conjugate # conversion from sensor frame to world frame

    # Update Yaw in RPY from Magnetometer
    rpy.z = -math.atan2(m.y, m.x)
    
    # Convert RPY to Quaternion
    q = rpy2q(rpy)

    return q

def heading(pose:Quaternion, mag, declination=0.0) -> float:
    '''
    Tilt compensated heading from compass
    Corrected for local magnetic declination
    Input:
      pose:       Quaternion
      mag:        Vector3D
      declination float
    Output:
      heading:    float
    '''
    if isinstance(mag, np.ndarray):
        _mag = Vector3D(mag)
    elif isinstance(mag, Vector3D):
        _mag = copy(mag)
    else:
        raise TypeError("Unsupported operand type for mag: {}".format(type(mag)))

    _mag.normalize()

    _mag_conjugate = pose * _mag * pose.conjugate
    heading = math.atan2(_mag_conjugate.y,_mag_conjugate.x) + declination

    # # Convert pose quaternion to RPY
    # rpy = q2rpy(pose)

    # cos_roll  = math.cos(rpy.x)
    # sin_roll  = math.sin(rpy.x)
    # cos_pitch = math.cos(rpy.y)
    # sin_pitch = math.sin(rpy.y)

    # # Tilt compensated magnetic field X component:
    # head_x = _mag.x*cos_pitch + _mag.y*sin_roll*sin_pitch + _mag.z*cos_roll*sin_pitch
    # # Tilt compensated magnetic field Y component:
    # head_y = _mag.y*cos_roll - _mag.z*sin_roll;
    # # Magnetic Heading
    # heading = -math.atan2(head_y,head_x) - declination

    return heading if heading > 0 else TWOPI + heading

def q2gravity(pose: Quaternion) -> Vector3D:
    '''
    Creates unit Gravity vector from pose quaternion.

    North East Down (Gravity is positive in Z direction pointing down)
    
    3x3 rotation matrix from quaternion
    multiply with dot product
    
    Usually its Matrix times Vector

    dot (
        np.array([
            [1 - 2 * (yy + zz),     2 * (xy - zw),     2 * (xz + yw)],
            [    2 * (xy + zw), 1 - 2 * (xx + zz),     2 * (yz - xw)],
            [    2 * (xz - yw),     2 * (yz + xw), 1 - 2 * (xx + yy)]
        ]).T,
        np.array([0,0,1])
    )
    
    Results in the bottom row of the rotation matrix.

    gx =  2*(xz - wy)
    gy =  2*(yz + wx) 
    gz =  1 - 2(xx + yy) 
    
    '''
    x =  2.0 * (pose.x * pose.z - pose.w * pose.y)
    y =  2.0 * (pose.y * pose.z + pose.w * pose.x)
    z =  1 - 2.0 * (pose.x**2 + pose.y**2) 
    
    return Vector3D(x, y, z)

def sensorAcc(acc: Vector3D, q: Quaternion, g: float) -> Vector3D:
    """
    compute residual acceleration in sensor frame
    """
    return (acc - g*q2gravity(q))

def earthAcc(acc: Vector3D, q: Quaternion, g: float) -> Vector3D:
    """
    compute residual acceleration in earth frame
    """
    acc_r = acc.rot(q.r33.T)
    acc_r.z -= g # subtract gravity
    return (acc_r)

def gravity(latitude: float, altitude: float) -> float:
    '''
    Gravity on Ellipsoid Surface
    
    from https://github.com/Mayitzin/ahrs/blob/master/ahrs/utils/wgs84.py
    '''
    a  = 6_378_137.0                        # EARTH_EQUATOR_RADIUS
    f  = 1./298.257223563                   # EARTH_FLATTENING
    gm = 3.986004418e14                     # EARTH_GM  
    w  = 7.292115e-5                        # EARTH_ROTATION
    b  = a*(1-f)                            # EARTH_POLE_RADIUS
    ge = 9.78032533590406                   # EARTH_EQUATOR_GRAVITY
    gp = 9.832184937863065                  # EARTH_POLE_GRAVITY
    e2 = 0.0066943799901413165              # EARTH_ECCENTRICITY_SQUARED
    lat = latitude * DEG2RAD                # LATITUDE in radians
    k  = (b*gp)/(a*ge)-1
    sin2 = math.sin(lat)**2
    gravity = ge*(1+k*sin2)/math.sqrt(1-e2*sin2)  # Gravity on Ellipsoid Surface
    if altitude != 0.0:
        m = w**2*a**2*b/gm                  # Gravity constant
        gravity *= 1.-2.*altitude*(1.+f+m-2*f*sin2)/a + 3.*altitude**2/a**2   # Gravity Above Ellipsoid

    return gravity

