from collections import deque
from pyIMU.quaternion import Vector3D, Quaternion, r33toq
from pyIMU.quaternion import TWOPI, DEG2RAD, EPSILON
import numpy as np
import math
import struct
import numbers
from copy import copy

IDENTITY_QUATERNION = Quaternion(1.0, 0.0, 0.0, 0.0)
VECTOR_ZERO         = Vector3D(0.0, 0.0, 0.0)

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
            self.squared_sum = self.squared_sum + (value * value)
        else:
            old_value = self.window.popleft()
            self.window.append(value)
            self.sum = self.sum + value - old_value
            self.squared_sum = self.squared_sum + (value * value) - (old_value * old_value)
            
    @property
    def avg(self) -> float:
        return self.sum / self.len

    @property
    def var(self) -> float:
        return (self.squared_sum - (self.sum * self.sum) / self.len) / self.len


def vector_angle2q(vec: Vector3D, angle: float = 0.0) -> Quaternion:
    '''Create quaternion based on rotation around vector'''
    _vec = copy(vec)
    _vec.normalize()
    halfAngle = angle * 0.5
    sinHalfAngle = math.sin(halfAngle)
    return Quaternion(
        w = math.cos(halfAngle), 
        x = vec.x * sinHalfAngle, 
        y = vec.y * sinHalfAngle, 
        z = vec.z * sinHalfAngle)

def q2rpy(q: Quaternion) -> Vector3D:
    '''
    quaternion to roll pitch yaw
    chat.openai.com
    '''
    
    wx = q.w * q.x
    yz = q.y * q.z
    xx = q.x * q.x
    yy = q.y * q.y
    zz = q.z * q.z
    wy = q.w * q.y
    xz = q.x * q.z
    wz = q.w * q.z
    xy = q.x * q.y

    # roll (x-axis rotation)
    sinr_cosp =       2.*(wx + yz)
    cosr_cosp = 1.0 - 2.*(xx + yy)
    roll = math.atan2(sinr_cosp, cosr_cosp)
            
    # pitch (y-axis rotation)
    sinp      =       2.*(wy - xz)
    if abs(sinp) >= 1.:
        pitch = math.copysign(math.pi / 2.0, sinp)  # use 90 degrees if out of range
    else:
        pitch = math.asin(sinp)
        
    # yaw (z-axis rotation)
    siny_cosp =       2.*(wz + xy)
    cosy_cosp = 1.0 - 2.*(yy + zz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return Vector3D(x=roll, y=pitch, z=yaw)

def rpy2q(r, p:float = 0., y: float = 0.) -> Quaternion:
    '''
    assume vector contains roll, pitch, yaw and convert to quaternion
    chat.openai.com 
    accel2q is q.y and q.z have wrong sign
    '''
    
    if isinstance(r, Vector3D):
        roll  = r.x 
        pitch = r.y
        yaw   = r.z
    elif isinstance(r, np.ndarray):
        if len(r) == 3:
            roll, pitch, yaw = r
    elif isinstance(r, numbers.Number):
        roll  = r
        pitch = p
        yaw   = y    
    else:
        raise TypeError("Unsupported operand type for rpy2q: {}".format(type(r)))
    
    cy2 = math.cos(yaw   * 0.5)
    sy2 = math.sin(yaw   * 0.5)
    cp2 = math.cos(pitch * 0.5)
    sp2 = math.sin(pitch * 0.5)
    cr2 = math.cos(roll  * 0.5)
    sr2 = math.sin(roll  * 0.5)

    w = cy2 * cp2 * cr2 + sy2 * sp2 * sr2
    x = cy2 * cp2 * sr2 - sy2 * sp2 * cr2
    y = sy2 * cp2 * sr2 + cy2 * sp2 * cr2
    z = sy2 * cp2 * cr2 - cy2 * sp2 * sr2

    return Quaternion(w, x, y, z)

def accel2rpy(acc) -> Vector3D:
    ''' 
    gravity to roll pitch yaw
    when X forward, Y right, Z down
    '''
    if isinstance(acc, Vector3D):
        _acc = copy(acc)
    elif isinstance(acc, np.ndarray):
        if len(acc) == 3:
            _acc = Vector3D(acc)
    else:
        raise TypeError("Unsupported operand type for accel2rpy: {}".format(type(x)))

    _acc.normalize()

    roll  = math.atan2(_acc.y, _acc.z)
    pitch = math.atan2(-_acc.x, math.sqrt(_acc.y*_acc.y + _acc.z* _acc.z))
    yaw   =  0.0
    return(Vector3D(roll, pitch, yaw))

def accel2q(acc) -> Quaternion:
    '''
    Converts Accelerometer to Quaternion assuming no motion
    '''

    if isinstance(acc, Vector3D):
        _acc = copy(acc) 
    elif isinstance(acc, np.ndarray):
        if len(acc) == 3:
            _acc = Vector3D(acc)
    else:
        raise TypeError("Unsupported operand type for accel2rpy: {}".format(type(acc)))

    _acc.normalize()
    # Calculate roll and pitch angles
    roll  =  math.atan2(_acc.y, _acc.z)
    # verify the negative sign on the pitch
    pitch = -math.atan2(-_acc.x, math.sqrt(_acc.y*_acc.y + _acc.z*_acc.z))
    # yaw   =  0.0

    cp2 = math.cos(pitch * 0.5)
    sp2 = math.sin(pitch * 0.5)
    cr2 = math.cos(roll  * 0.5)
    sr2 = math.sin(roll  * 0.5)
    # cy2 = 1.
    # sy2 = 0.

    w =  cr2 * cp2
    x =  sr2 * cp2
    y = -cr2 * sp2
    z =  sr2 * sp2

    q = Quaternion(w, x, y, z)
    q.normalize()

    return q

def accelmag2rpy(acc, mag) -> Quaternion:
    '''
    Estimate Pose Vector from Accelerometer and Compass
    Assuming X forward, Y right, Z down
    1) Acceleration to Roll Pitch Yaw=0.0
    2) Update estimated Yaw in Euler and return result

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
    
    
    # 1) calculate roll and pitch from acceleration (gravity) vector
    _acc.normalize()
    roll  = math.atan2(_acc.y, _acc.z)
    pitch = math.atan2(-_acc.x, math.sqrt(_acc.y*_acc.y + _acc.z*_acc.z))

    # 2) calculate yaw from magnetometer (rotation around z-axis)
    _mag.normalize()
    mag_x = _mag.x * math.cos(pitch) + _mag.y * math.sin(pitch) * math.sin(roll) + _mag.z * math.sin(pitch) * math.cos(roll)
    mag_y = _mag.y * math.cos(roll)  - _mag.z * math.sin(roll)
    yaw = math.atan2(-mag_y, mag_x)
    
    return Vector3D(roll, pitch, yaw)

def accelmag2q(acc, mag) -> Quaternion:
    '''
    Estimate Pose Vector from Accelerometer and Compass
    Assuming X forward, Y right, Z down

    Using following approach:
    1) accel mag to rpy
    2) rpy to quaternion

    Compared to other approaches:

    - pypi AHRS
      R = am2DCM(a, m, frame=NED)
      q = dcm2quat(R)
    
        if frame.upper() not in ['ENU', 'NED']:
            raise ValueError("Wrong coordinate frame. Try 'ENU' or 'NED'")
            a = np.array(a)
            m = np.array(m)
            H = np.cross(m, a)
            H /= np.linalg.norm(H)
            a /= np.linalg.norm(a)
            M = np.cross(a, H)
            if frame.upper() == 'ENU':
                return np.array([[H[0], M[0], a[0]],
                                [H[1], M[1], a[1]],
                                [H[2], M[2], a[2]]])
                                
            return np.array([[M[0], H[0], -a[0]],
                            [M[1], H[1], -a[1]],
                            [M[2], H[2], -a[2]]])
    
        if R.shape[0] != R.shape[1]:
            raise ValueError('Input is not a square matrix')
        if R.shape[0] != 3:
            raise ValueError('Input needs to be a 3x3 array or matrix')
        q = np.array([1., 0., 0., 0.])
        q[0] = 0.5*np.sqrt(1.0 + R.trace())
        q[1] = (R[1, 2] - R[2, 1]) / q[0]
        q[2] = (R[2, 0] - R[0, 2]) / q[0]
        q[3] = (R[0, 1] - R[1, 0]) / q[0]
        q[1:] /= 4.0
        return q / np.linalg.norm(q)    

    - chat.openai.com
        Could not make work
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

    # 1) calculate roll and pitch from acceleration (gravity) vector
    _acc.normalize()
    roll  = math.atan2(_acc.y, _acc.z)
    pitch = math.atan2(-_acc.x, math.sqrt(_acc.y*_acc.y + _acc.z*_acc.z))

    # 2) calculate yaw from magnetometer (rotation around z-axis)
    _mag.normalize()
    mag_x = _mag.x * math.cos(pitch) + _mag.y * math.sin(pitch) * math.sin(roll) + _mag.z * math.sin(pitch) * math.cos(roll)
    mag_y = _mag.y * math.cos(roll)  - _mag.z * math.sin(roll)
    yaw = math.atan2(-mag_y, mag_x)

    # 3) convert roll pitch yaw to quaternion
    cy2 = math.cos(yaw   * 0.5)
    sy2 = math.sin(yaw   * 0.5)
    cp2 = math.cos(pitch * 0.5)
    sp2 = math.sin(pitch * 0.5)
    cr2 = math.cos(roll  * 0.5)
    sr2 = math.sin(roll  * 0.5)

    w = cy2 * cp2 * cr2 + sy2 * sp2 * sr2
    x = cy2 * cp2 * sr2 - sy2 * sp2 * cr2
    y = sy2 * cp2 * sr2 + cy2 * sp2 * cr2
    z = sy2 * cp2 * cr2 - cy2 * sp2 * sr2

    q = Quaternion(w=w, x=x, y=y, z=z)

    # Method of using North, East, Down
    # to create rotation matrix and then
    # converting rotation matrix to quaternion.
    # Not working at this time.
    #    
    # # Calculate the auxiliary vectors
    # # East is cross product of gravity and magnetic field
    # east  = _acc.cross(_mag)
    # east.normalize()

    # # North is cross product of east and gravity
    # _acc.normalize()        
    # north = east.cross(_acc)
    
    # # Assign the rotation matrix
    # # "Each column or row gives the direction of one of the transformed axes."
    # r33 = np.empty((3,3))
    # r33[:, 0] = north.v
    # r33[:, 1] = east.v
    # r33[:, 2] = acc.v
    # q = r33toq(r33, check=True)
            
    return q

def heading(q:Quaternion, mag, declination=0.0) -> float:
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

    # _mag_rot = pose * _mag * pose.conjugate
    _mag_rot = q * _mag * q.conjugate

    heading = math.atan2(_mag_rot.y,_mag_rot.x) + declination

    return heading if heading > -EPSILON else TWOPI + heading

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
    z =  1 - 2.0 * (pose.x*pose.x + pose.y*pose.y) 
    
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
    acc_r = q * acc * q.conjugate
    acc_r.z = acc_r.z - g # subtract gravity
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

