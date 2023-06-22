###########################################################
# Quaternion and Vector3D data types and functions
# Includes gravity and acceleration projections to IMU or world frame
#
# Urs Utzinger, Spring 2023
# chat.openai.com
###########################################################

import numpy as np
import math

###########################################################
# Constants
###########################################################

TWOPI   = 2.0 * math.pi
DEG2RAD = math.pi / 180.0
RAD2DEG = 180.0 / math.pi
EPSILON = 2.0*math.ldexp(1.0, -53)

class Quaternion():
    '''
    Quaternion Class
    q1 = Quaternion(1., 2., 3., 4.)
    q2 = Quaternion(w=5., x=6., y=7., z=8.)
    q3 = Quaternion(np.array([9,10,11,12]))

    q5 = q1 + q2
    q6 = q1 * q2
    q7 = 2 * q1
    
    q1.conjugate
    q1.inverse
    q1.normalize()
    q1.r33 (cosine matrix)
    q1.norm: length of quaternion
    q1.v: vector part of quaternion as Vector3D
    q1.q: quaternion as np.array
    '''
    def __init__(self, w=0.0, x=0.0, y=0.0, z=0.0, v=None):
        # allows any possible combination to be passed in
        if v is None:
            if np.isscalar(w):
                self.w = w
                self.x = x
                self.y = y
                self.z = z
            elif isinstance(w, np.ndarray):
                if len(w) == 3:
                    self.w=w[0]
                    self.x=w[1]
                    self.y=w[2]
                    self.z=w[3]
                elif len(w)==4:
                    self.w = 0.
                    self.x = w[0]
                    self.y = w[1]
                    self.z = w[2]                                    
            elif isinstance(w, Vector3D):
                self.w = 0.
                self.x = w.x
                self.y = w.y
                self.z = w.z
            elif isinstance(w, Quaternion):
                self.w = w.w
                self.x = w.x
                self.y = w.y
                self.z = w.z
        elif isinstance(v, Vector3D):
            self.w=0.
            self.x=v.x
            self.y=v.y
            self.z=v.z
        elif isinstance(v, np.ndarray):
            if len(v) == 3:
                self.w=0.
                self.x=v[0]
                self.y=v[1]
                self.z=v[2]
            elif len(v)==4:
                self.w=v[0]
                self.x=v[1]
                self.y=v[2]
                self.z=v[3]
        elif isinstance(v, Quaternion):
                self.w=v.w
                self.x=v.x
                self.y=v.y
                self.z=v.z
                                       
    def __str__(self):
        return f"Quaternion({self.w}, {self.x}, {self.y}, {self.z})"
    
    def __repr__(self):
        return str(self)
    
    def __add__(self, other):
        '''add two quaternions or quaternion and scalar'''
        if isinstance(other, Quaternion):
            return Quaternion(self.w+other.w, self.x+other.x, self.y+other.y, self.z+other.z)
        elif isinstance(other, np.ndarray):
            return Quaternion(self.w+other[0], self.x+other[1], self.y+other[2], self.z+other[3])
        elif np.isscalar(other):
            return Quaternion(self.w+other, self.x+other, self.y+other, self.z+other)
        else:
            raise TypeError("Unsupported operand type for +: Quaternion and {}".format(type(other)))
                
    def __sub__(self, other):
        '''subtract two quaternions or quaternion and scalar'''
        if isinstance(other, Quaternion):
            return Quaternion(self.w-other.w, self.x-other.x, self.y-other.y, self.z-other.z)
        elif isinstance(other, np.ndarray):
            return Quaternion(self.w-other[0], self.x-other[1], self.y-other[2], self.z-other[3])
        elif np.isscalar(other):
            return Quaternion(self.w-other, self.x-other, self.y-other, self.z-other)
        else:
            raise TypeError("Unsupported operand type for -: Quaternion and {}".format(type(other)))

    def __mul__(self, other):
        '''multiply two quaternions or quaternion and vector'''
        if isinstance(other, Quaternion):
            w = (self.w * other.w) - (self.x * other.x) - (self.y * other.y) - (self.z * other.z)
            x = (self.w * other.x) + (self.x * other.w) + (self.y * other.z) - (self.z * other.y)
            y = (self.w * other.y) - (self.x * other.z) + (self.y * other.w) + (self.z * other.x)
            z = (self.w * other.z) + (self.x * other.y) - (self.y * other.x) + (self.z * other.w)
            return Quaternion(w, x, y, z)
        elif isinstance(other, Vector3D):
            '''
            multiply quaternion with vector
            vector is converted to quaternion with [0,vector]
            the computed the same as above with other.w=0
            '''
            w = - (self.x * other.x) - (self.y * other.y) - (self.z * other.z)
            x =    self.w * other.x  +  self.y * other.z  -  self.z * other.y
            y =    self.w * other.y  -  self.x * other.z  +  self.z * other.x
            z =    self.w * other.z  +  self.x * other.y  -  self.y * other.x
            return Quaternion(w, x, y, z)
        elif isinstance(other, (int, float)):
            '''multiply with scalar'''
            return Quaternion(self.w*other,self.x*other,self.y*other,self.z*other)
        else:
            raise TypeError("Unsupported operand type")
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Quaternion(self.w/other,self.x/other,self.y/other,self.z/other)
        else:
            raise TypeError("Unsupported operand type")
    
    def __eq__(self, other):
        '''are the two quaternions equal'''
        return (self.w==other.w and self.x==other.x and self.y==other.y and self.z==other.z)

    def normalize(self) -> float:
        mag = self.norm
        if mag != 0:
            self.w /= mag
            self.x /= mag
            self.y /= mag
            self.z /= mag
    
    @property
    def v(self) -> np.ndarray:
        '''extract the vector component of the quaternion'''
        # return np.array([self.x,self.y,self.z])
        return Vector3D(self.x,self.y,self.z)

    @property
    def q(self) -> np.ndarray:
        '''extract the vector component of the quaternion'''
        # return np.array([self.x,self.y,self.z])
        return np.array([self.w,self.x,self.y,self.z])

    @property
    def conjugate(self):
        '''conjugate of quaternion'''
        return Quaternion(self.w, -self.x, -self.y, -self.z)
    
    @property
    def norm(self) -> float:
        '''length of quaternion'''
        return math.sqrt(self.w**2  + self.x**2 + self.y**2 + self.z**2)
    
    @property
    def inverse(self):
        '''inverse of quaternion'''
        return self.conjugate / self.norm

    @property
    def r33(self) -> np.ndarray:
        '''
        quaternion to 3x3 rotation matrix
        q * P * q.conjugate
        [ 1-2zz-2yy,   -2wz+2xy,    2wy+2xz
            2xy+2wz,  1-2zz-2xx,    2yz-2wx
            2xz-2wy,    2wx+2yz,  1-2yy-2xx ]
            
        simplifications because ww+xx+yy+zz = 1
        
        - https://www.euclideanspace.com/maths/geometry/rotations/conversons/quaternionToMatrix/index.htm
        - chat.openai.com
        '''
        
        # Normalize quaternion
        self.normalize()
        
        # Compute rotation matrix elements
        # (not needed) ww = self.w**2
        xx = self.x**2
        xy = self.x * self.y
        xz = self.x * self.z
        xw = self.x * self.w
        yy = self.y**2
        yz = self.y * self.z
        yw = self.y * self.w
        zz = self.z**2
        zw = self.z * self.w

        # Construct the rotation matrix

        # Point Rotation:
        return np.array([
            [1 - 2 * (yy + zz),     2 * (xy - zw),     2 * (xz + yw)],
            [    2 * (xy + zw), 1 - 2 * (xx + zz),     2 * (yz - xw)],
            [    2 * (xz - yw),     2 * (yz + xw), 1 - 2 * (xx + yy)]
        ])
        # Frame Rotation would be transpose of the above

        # Same as above
        # return np.array([
        #     [ 2 * (ww - 0.5 * xx),       2 * (xy - zw),      2 * (xz + yw)],
        #     [       2 * (xy + zw), 2 * (ww - 0.5 + yy),      2 * (yz - xw)],
        #     [       2 * (xz - yw),       2 * (yz + xw), 2 * (ww - 0.5 +zz)]
        # ])

    @property
    def isZero(self) -> bool:
        return (abs(self.w) <= EPSILON and abs(self.x) <= EPSILON and abs(self.y) <= EPSILON and abs(self.z) <= EPSILON)
        
    
class Vector3D():
    '''
    3D Vector Class
    v1 = Vector3D(1., 2., 3.)
    v2 = Vector3D(x=4., y=5., z=6.)
    v3 = Vector3D(np.array([7,8,9]))

    v4 = v1 + v2
    v5 = v1 * v2
    v6 = 2. * v1
    
    v1.dot(v2)
    v1.cross(v2)
    v1.normalize()
    v1.norm: length of vector
    v1.rotate(np.array[3x3])
    v1.v: vector as np.array
    v1.q: vector as quaternion with w=0.
    '''
    def __init__(self, x=0.0, y=0.0, z=0.0):
        if np.isscalar(x):
            self.x = x
            self.y = y
            self.z = z
        elif isinstance(x, np.ndarray):
            if len(x) == 3:
                self.x = x[0]
                self.y = x[1]
                self.z = x[2]
        elif isinstance(x, Vector3D):
            self.x = x.x
            self.y = x.y
            self.z = x.z
 
    def __str__(self):
        return f"Vector3D({self.x}, {self.y}, {self.z})"

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        if isinstance(other, Vector3D):
            return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
        elif np.isscalar(other):
            return Vector3D(self.x + other, self.y + other, self.z + other)
        else:
            raise TypeError("Unsupported operand type for +: Vector3D and {}".format(type(other)))

    def __sub__(self, other):
        if isinstance(other, Vector3D):
            return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
        elif isinstance(other, (int, float)):
            return Vector3D(self.x - other, self.y - other, self.z - other)
        else:
            raise TypeError("Unsupported operand type for -: Vector3D and {}".format(type(other)))

    def __mul__(self, other):
        if np.isscalar(other):
            return Vector3D(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, Vector3D):
            return Vector3D(self.x * other.x, self.y * other.y, self.z * other.z)            
        elif isinstance(other, np.ndarray):
            if len(other) == 3:
                return Vector3D(self.x * other[0], self.y * other[1], self.z * other[2])
            elif len(other) == 4:
                '''
                other is quaternion of w,x,y,z 
                convert vector to quaternion [0,x,y,z]
                '''
                # x1, y1, z1     = self.v # w1 is 0, dont use
                other_w, other_x, other_y, other_z = other
                w = - self.x * other_x - self.y * other_y - self.z * other_z
                x =   self.x * other_w + self.y * other_z - self.z * other_y
                y = - self.x * other_z + self.y * other_w + self.z * other_x
                z =   self.x * other_y - self.y * other_x + self.z * other_w
                return Quaternion(w, x, y, z)
        elif isinstance(other, Quaternion):
                w = - self.x * other.x - self.y * other.y - self.z * other.z
                x =   self.x * other.w + self.y * other.z - self.z * other.y
                y = - self.x * other.z + self.y * other.w + self.z * other.x
                z =   self.x * other.y - self.y * other.x + self.z * other.w
                return Quaternion(w, x, y, z)
        else:
            raise TypeError("Unsupported operand type for *: Vector3D and {}".format(type(other)))

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __div__(self, other):
        if isinstance(other, (int, float)) and other != 0:
            return Vector3D(self.x / other, self.y / other, self.z / other)
        elif isinstance(other, Vector3D):
            return Vector3D(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            raise ValueError("Unsupported operand value for /: {}".format(other))

    def __pow__(self, other):
        '''potentiate'''
        if isinstance(other, (int, float)):
            return Vector3D(self.x ** other, self.y ** other, self.z ** other)
        elif isinstance(other, Vector3D):
            return Vector3D(self.x ** other.x, self.y ** other.y, self.z ** other.z)
        else:
            raise TypeError("Unsupported operand type for **: Vector3D and {}".format(type(other)))

    def __eq__(self, other):
        '''are the two vectors equal'''
        return (self.x==other.x and self.y==other.y and self.z==other.z)

    def __lt__(self, other):
        '''is vector smaller than other'''
        return Vector3D(x=self.x < other.x, y=self.y<other.y, z=self.z<other.z)

    def min(self, other):
        '''smaller component of the two vectors'''        
        return Vector3D(x=min(self.x, other.x), y=min(self.y, other.y), z=min(self.z, other.z))

    def max(self, other):
        '''larger component of the two vectors'''        
        return Vector3D(x=max(self.x, other.x), y=max(self.y, other.y), z=max(self.z, other.z))

    def abs(self):
        '''absolute'''        
        return Vector3D(x=math.abs(self.x), y=math.abs(self.y), z=math.abs(self.z))

    def normalize(self):
        mag = self.norm
        if mag != 0:
            self.x /= mag
            self.y /= mag
            self.z /= mag

    def sum(self):
        return self.x  + self.y + self.z

    def dot(self, other) -> float:
        if isinstance(other, Vector3D):
            return self.x * other.x + self.y * other.y + self.z * other.z
        if isinstance(other, np.ndarray):
            if len(other) == 3:
                return self.x * other[0] + self.y * other[1] + self.z * other[2]
            else:
                raise TypeError("Unsupported operand type for dot product: nd.array length {}".format(len(other)))
        else:
            raise TypeError("Unsupported operand type for dot product: Vector3D and {}".format(type(other)))

    def cross(self, other):
        if isinstance(other, Vector3D):
            x = self.y * other.z - self.z * other.y
            y = self.z * other.x - self.x * other.z
            z = self.x * other.y - self.y * other.x
            return Vector3D(x, y, z)
        else:
            raise TypeError("Unsupported operand type for cross product: Vector3D and {}".format(type(other)))

    def rotate(self, other):
        if isinstance(other, np.ndarray):
            if other.shape == (3,3):
                # rotated_vector = np.dot(other.T, np.array([self.x,self.y,self.z]))
                rotated_vector = np.dot(np.array([self.x,self.y,self.z]),other)
                return(Vector3D(x=rotated_vector[0], y=rotated_vector[1], z=rotated_vector[2]))
            else:
                raise TypeError("Unsupported operand type for cross product: Vector3D and nd.array of shape {}".format(other.shape))
        else:
            raise TypeError("Unsupported operand type for cross product: Vector3D and {}".format(type(other)))

    @property
    def q(self):
        '''return np array with rotation 0 and vector x,y,z'''
        return Quaternion(w=0, x=self.x, y=self.y, z=self.z)

    @property
    def v(self) -> np.ndarray:
        '''returns np array of vector'''
        return np.array([self.x,self.y,self.z])
    @v.setter
    def v(self, val):
        '''set vector'''
        if isinstance(val, (list, tuple, np.ndarray)):
            self.x = val[0]
            self.y = val[1]
            self.z = val[2]
        elif isinstance(val, (int,float)):
            self.x = val
            self.y = val
            self.z = val            
        else:
            raise TypeError("Unsupported operand type for cross product: Vector3D and {}".format(type(val)))
    
    @property
    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    @property
    def isZero(self) -> bool:
        return (abs(self.x) <= EPSILON and abs(self.y) <= EPSILON and abs(self.z) <= EPSILON)

