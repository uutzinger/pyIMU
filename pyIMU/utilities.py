from collections import deque
from pyIMU.quaternion import Vector3D, Quaternion, TWOPI, DEG2RAD
import numpy as np
import math
import struct

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

def rpy2q(r=0.0, p=0.0, y=0.0) -> Quaternion:
    '''assume vector contains roll, pitch, yaw and convert to quaternion'''
    
    if np.isscalar(r):
        roll  = r
        pitch = p
        yaw   = y
    elif isinstance(r, Vector3D):
        roll  = r.x 
        pitch = r.y
        yaw   = r.z
    elif isinstance(r, np.ndarray):
        if len(r) == 3:
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

def accel2rpy(x=0., y=0., z=0.) -> Vector3D:

    if np.isscalar(x):
        _x = x
        _y = y
        _z = z
    elif isinstance(x, Vector3D):
        _x = x.x 
        _y = x.y
        _z = x.z
    elif isinstance(x, np.ndarray):
        if len(x) == 3:
            _x, _y, _z = x
    else:
        raise TypeError("Unsupported operand type for accel2rpy: {}".format(type(x)))

    normAccel = Vector3D(_x, _y, _z)
    normAccel.normalize()

    roll  =  math.atan2(normAccel.y, normAccel.z)
    pitch = -math.atan2(normAccel.x, math.sqrt(normAccel.y * normAccel.y + normAccel.z * normAccel.z))
    yaw   =  0.0
    return(Vector3D(roll, pitch, yaw))

def accel2q(x=0.,y=0.,z=0.) -> Quaternion:
    '''
    Converts Accelerometer to Quaternion assuming no motion
    Input accelerometer reading x,y,z
    Output quaternion w,x,y,z
    '''
    # # vec_z = Vector3D(0, 0, 1.0)
    # # angle = math.acos(z.dot(normAccel)) simplified to
    # angle = math.acos(normAccel.z)
    # # vec = normAccel.cross(vec_z) simplified to
    # vec = Vector3D(x=normAccel.y, y=-normAccel.x, z=0.)
    # return vec.angle2q(angle)

    if np.isscalar(x):
        _x = x
        _y = y
        _z = z
    elif isinstance(x, Vector3D):
        _x = x.x 
        _y = x.y
        _z = x.z
    elif isinstance(x, np.ndarray):
        if len(x) == 3:
            _x, _y, _z = x
    else:
        raise TypeError("Unsupported operand type for accel2rpy: {}".format(type(x)))

    normAccel = Vector3D(_x, _y, _z)
    normAccel.normalize()

    roll  =  math.atan2(normAccel.y, normAccel.z)
    pitch = -math.atan2(normAccel.x, math.sqrt(normAccel.y * normAccel.y + normAccel.z * normAccel.z))
    # yaw   =  0.0

    cosY2 = math.cos(pitch * 0.5)
    sinY2 = math.sin(pitch * 0.5)
    cosX2 = math.cos(roll  * 0.5)
    sinX2 = math.sin(roll  * 0.5)

    w =  cosY2 * cosX2 
    x =  cosY2 * sinX2
    y =  sinY2 * cosX2
    z = -sinY2 * sinX2

    return Quaternion(w, x, y, z)
    
def accelmag2q(accel, mag) -> Quaternion:
    '''
    Estimate Pose Vector from Accelerometer and Compass
    1) Acceleration to Roll Pitch Yaw=0.0
    2) Convert RPY to estimated Pose Quaternion
    3) Rotate Compass to World from estimated Pose
    4) Update estimated Yaw in Euler and return result
    '''
    if isinstance(accel, np.ndarray):
        if len(accel) == 3:
            accel = Vector3D(accel)
            
    rpy = accel2rpy(accel.x, accel.y, accel.z) # rpy.z = 0

    # rpy to quaternion, simplified because rpy.z is zero
    cosP2 = math.cos(rpy.p * 0.5)
    sinP2 = math.sin(rpy.p * 0.5)
    cosR2 = math.cos(rpy.r * 0.5)
    sinR2 = math.sin(rpy.r * 0.5)
    q = Quaternion(
        w =  cosP2 * cosR2,
        x =  cosP2 * sinR2, 
        y =  sinP2 * cosR2,
        z = -sinP2 * sinR2
    )
    
    # rotate Magnetometer to Q pose
    m = Quaternion(
        w = 0.,
        x = mag.x,
        y = mag.y,
        z = mag.z
    )
    m = q * m * q.conjugate # conversion from sensor frame to world frame

    # Update Yaw in RPY from Magnetometer
    rpy.z = -math.atan2(m.y, m.x)
    
    # Convert RPY to Quaternion
    return rpy2q(rpy)

def heading(pose: Quaternion, mag, declination=0.0) -> float:
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

    # NOT TESTED YET
    
    if isinstance(mag, np.ndarray):
        mag=Vector3D(mag)
    elif isinstance(mag, Vector3D):
        pass
    else:
        raise TypeError("Unsupported operand type for mag: {}".format(type(mag)))

    # Convert pose quaternion to RPY
    rpy = q2rpy(pose)
    
    cos_roll  = math.cos(rpy.x())
    sin_roll  = math.sin(rpy.x())
    cos_pitch = math.cos(rpy.y())
    sin_pitch = math.sin(rpy.y())

    # Tilt compensated magnetic field X component:
    head_x = mag.x*cos_pitch + mag.y*sin_roll*sin_pitch + mag.z*cos_roll*sin_pitch
    # Tilt compensated magnetic field Y component:
    head_y = mag.y*cos_roll - mag.z*sin_roll;
    # Magnetic Heading
    heading = -math.atan2(head_y,head_x) - declination

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

def ellipsoid_fit(X, **kwargs):
    '''
    Fit an ellispoid/sphere to a set of xyz data points:

      [center, radii, evecs, pars ] = ellipsoid_fit( X )
      [center, radii, evecs, pars ] = ellipsoid_fit( np.array([x y z]) );
      [center, radii, evecs, pars ] = ellipsoid_fit( X, 1 );
      [center, radii, evecs, pars ] = ellipsoid_fit( X, 2, 'xz' );
      [center, radii, evecs, pars ] = ellipsoid_fit( X, 3 );

    Parameters:
    * X=[x y z]    - Cartesian data, n x 3 matrix or three n x 1 vectors
    * optionrrrrrrrrrrrrr       - 0 fits an arbitrary ellipsoid (default),
                   - 1 fits an ellipsoid with its axes along [x y z] axes
                   - 2 followed by, say, 'xy' fits as 1 but also x_rad = y_rad
                   - 3 fits a sphere

    Output:
    * center    -  ellipsoid center coordinates [xc; yc; zc]
    * ax        -  ellipsoid radii [a; b; c]
    * evecs     -  ellipsoid radii directions as columns of the 3x3 matrix
    * v         -  the 9 parameters describing the ellipsoid algebraically: 
                   Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1

    This needs: numpy
    
    Author:
    Yury Petrov, Northeastern University, Boston, MA
    '''

    option  = kwargs.get('option', 0)
    equals  = kwargs.g9et('equals', 'xy')
    
    if X.shape[1] != 3:
        raise ValueError('Input data must have three columns!')
    else:
        x = X[:, 0]
        y = X[:, 1]
        z = X[:, 2]
    
    # Need nine or more data points
    if len(x) < 9 and option == 0:
        raise ValueError('Must have at least 9 points to fit a unique ellipsoid')
    if len(x) < 6 and option == 1:
        raise ValueError('Must have at least 6 points to fit a unique oriented ellipsoid')
    if len(x) < 5 and option == 2:
        raise ValueError('Must have at least 5 points to fit a unique oriented ellipsoid with two axes equal')
    if len(x) < 3 and option == 3:
        raise ValueError('Must have at least 4 points to fit a unique sphere')
    
    if   option == 0: # Fit ellipsoid in the form Ax^2 + By^2 + Cz^2 + 2Dxy + 2Exz + 2Fyz + 2Gx + 2Hy + 2Iz = 1
        D = np.column_stack((x*x, y*y, z*z, 2*x*y, 2*x*z, 2*y*z, 2*x, 2*y, 2*z))
    elif option == 1: # Fit ellipsoid in the form Ax^2 + By^2 + Cz^2 + 2Gx + 2Hy + 2Iz = 1
        D = np.column_stack((x*x, y*y, z*z, 2*x, 2*y, 2*z))
    elif option == 2: # Fit ellipsoid in the form Ax^2 + By^2 + Cz^2 + 2Gx + 2Hy + 2Iz = 1, where A = B or B = C or A = C
        if equals == 'yz' or equals == 'zy':
            D = np.column_stack((y*y+z*z, x*x, 2*x, 2*y, 2*z))
        elif equals == 'xz' or equals == 'zx':
            D = np.column_stack((x*x+z*z, y*y, 2*x, 2*y, 2*z))
        else:
            D = np.column_stack((x*x+y*y, z*z, 2*x, 2*y, 2*z))
    else:
        # Fit sphere in the form A(x^2 + y^2 + z^2) + 2Gx + 2Hy + 2Iz = 1
        D = np.column_stack((x*x+y*y+z*z, 2*x, 2*y, 2*z))
    
    # Solve the normal system of equations
    v = np.linalg.lstsq(D.T @ D, D.T @ np.ones(len(x)), rcond=None)[0]
    
    # Find the ellipsoid parameters
    if option == 0:
        # Form the algebraic form of the ellipsoid
        A = np.array([[v[0], v[3], v[4], v[6]],
                      [v[3], v[1], v[5], v[7]],
                      [v[4], v[5], v[2], v[8]],
                      [v[6], v[7], v[8],  -1]])
        
        # Find the center of the ellipsoid
        center = -np.linalg.inv(A[:3, :3]) @ np.array([v[6], v[7], v[8]])
        
        # Form the corresponding translation matrix
        T = np.eye(4)
        T[:3, 3] = center
        
        # Translate to the center
        R = T @ A @ T.T
        
        # Solve the Eigen problem
        evals, evecs = np.linalg.eig(-R[:3, :3] / R[3, 3])
        radii = np.sqrt(1. / np.diag(evals))
    else:
        if option == 1:
            v = np.array([v[0], v[1], v[2], 0, 0, 0, v[3], v[4], v[5]])
        elif option == 2:
            if equals == 'xz' or equals == 'zx':
                v = np.array([v[0], v[1], v[0], 0, 0, 0, v[2], v[3], v[4]])
            elif equals == 'yz' or equals == 'zy':
                v = np.array([v[1], v[0], v[0], 0, 0, 0, v[2], v[3], v[4]])
            else:  # xy
                v = np.array([v[0], v[0], v[1], 0, 0, 0, v[2], v[3], v[4]])
        else:
            v = np.array([v[0], v[0], v[0], 0, 0, 0, v[1], v[2], v[3]])
        
        center = -v[6:9] / v[0:3]
        gam = 1 + (v[6] ** 2 / v[0] + v[7] ** 2 / v[1] + v[8] ** 2 / v[2])
        radii = np.sqrt(gam / v[0:3])
        evecs = np.eye(3)
    
    return center, radii, evecs, v
