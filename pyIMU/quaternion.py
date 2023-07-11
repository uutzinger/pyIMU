###########################################################
# Quaternion and Vector3D data types and functions
# Includes gravity and acceleration projections to IMU or world frame
#
# Urs Utzinger, Spring 2023
# chat.openai.com
###########################################################

import numpy as np
import math
import numbers

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
            if isinstance(w, numbers.Number):
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
    
    def __copy__(self):
        return Quaternion(self.w, self.x, self.y, self.z)
    
    def __bool__(self):
        return not self.isZero
    
    def __abs__(self):
        return Quaternion(abs(self.w), abs(self.x), abs(self.y), abs(self.z))
    
    def __neg__(self):
        return Quaternion(-self.w, -self.x, -self.y, -self.z)

    def __len__(self):
        return 4
                     
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
        elif isinstance(other, numbers.Number):
            return Quaternion(self.w+other, self.x+other, self.y+other, self.z+other)
        else:
            raise TypeError("Unsupported operand type for +: Quaternion and {}".format(type(other)))
                
    def __sub__(self, other):
        '''subtract two quaternions or quaternion and scalar'''
        if isinstance(other, Quaternion):
            return Quaternion(self.w-other.w, self.x-other.x, self.y-other.y, self.z-other.z)
        elif isinstance(other, np.ndarray):
            return Quaternion(self.w-other[0], self.x-other[1], self.y-other[2], self.z-other[3])
        elif isinstance(other, numbers.Number):
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
            then computed the same as above with other.w=0
            '''
            w = - (self.x * other.x) - (self.y * other.y) - (self.z * other.z)
            x =    self.w * other.x  +  self.y * other.z  -  self.z * other.y
            y =    self.w * other.y  -  self.x * other.z  +  self.z * other.x
            z =    self.w * other.z  +  self.x * other.y  -  self.y * other.x
            return Quaternion(w, x, y, z)
        elif isinstance(other, numbers.Number):
            '''multiply with scalar'''
            return Quaternion(self.w*other,self.x*other,self.y*other,self.z*other)
        else:
            raise TypeError("Unsupported operand type")
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return Quaternion(self.w/other,self.x/other,self.y/other,self.z/other)
        else:
            raise TypeError("Unsupported operand type")

    def __floordiv__(self, other):
        if isinstance(other, numbers.Number):
            return Quaternion(self.w//other,self.x//other,self.y//other,self.z//other)
        else:
            raise TypeError("Unsupported operand type")

    def __eq__(self, other):
        '''are the two quaternions equal'''
        return (self.w==other.w and self.x==other.x and self.y==other.y and self.z==other.z)

    def normalize(self):
        mag = self.norm
        if mag != 0:
            self.w = self.w/mag
            self.x = self.x/mag
            self.y = self.y/mag
            self.z = self.z/mag
    
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
        return math.sqrt(self.w*self.w  + self.x*self.x + self.y*self.y + self.z*self.z)
    
    @property
    def inverse(self):
        '''inverse of quaternion'''
        return self.conjugate / self.norm

    @property
    def r33(self) -> np.ndarray:
        '''
        quaternion to 3x3 rotation matrix            
        simplifications because ww+xx+yy+zz = 1
        
        - https://www.euclideanspace.com/maths/geometry/rotations/conversons/quaternionToMatrix/index.htm
        - pypi.org AHRS
        - pypi.org quaternionic
        - chat.openai.com

        Assuming the quaternion R rotates a vector v according to

            v' = R * v * R⁻¹,

        we can also express this rotation in terms of a 3x3 matrix ℛ such that

            v' = ℛ * v.

        This function returns that matrix.
        '''

        # Normalize quaternion
        # self.normalize()
        
        # Compute rotation matrix elements
        xx = self.x * self.x
        xy = self.x * self.y
        xz = self.x * self.z
        xw = self.x * self.w
        yy = self.y * self.y
        yz = self.y * self.z
        yw = self.y * self.w
        zz = self.z * self.z
        zw = self.z * self.w

        # Construct the rotation matrix
        return np.array([
            [1.0 - 2.*(yy + zz),          2.*(xy - zw),          2.*(xz + yw)],
            [      2.*(xy + zw),    1.0 - 2.*(xx + zz),          2.*(yz - xw)],
            [      2.*(xz - yw),          2.*(yz + xw),    1.0 - 2.*(xx + yy)]
        ])        
    
    @property
    def isZero(self) -> bool:
        return (abs(self.w) <= EPSILON and abs(self.x) <= EPSILON and abs(self.y) <= EPSILON and abs(self.z) <= EPSILON)
 
def r33toq(r33, check= False) -> Quaternion:
    '''
    Rotation Matrix to Quaternion
    chat.openai.com
    https://github.com/blender/blender/blob/756538b4a117cb51a15e848fa6170143b6aafcd8/source/blender/blenlib/intern/math_rotation.c#L272

    pypi.org quaternionic:
        
    Assuming an orthogonal 3x3 matrix ℛ rotates a vector v such that

        v' = ℛ * v,

    we can also express this rotation in terms of a unit quaternion R such that

        v' = R * v * R⁻¹,

    where v and v' are now considered pure-vector quaternions.  This function
    returns that quaternion.  If `rot` is not orthogonal, the "closest" orthogonal
    matrix is used; see Notes below.
    '''

    if isinstance(r33, np.ndarray):
        if r33.shape == (3,3):

            if check: 
                det = np.linalg.det(r33)
                if not np.isfinite(det): r33 = np.eye(3)  # Set to identity matrix if determinant is not finite
                elif det < 0.0:          r33 = -r33       # Negate matrix if determinant is negative
                isOrthogonal = math.isclose(abs(det), 1.0, atol=EPSILON)
            else:
                isOrthogonal = True

            # NON ORTHOGONAL OPTION

            if not isOrthogonal:

                K3 = np.array([
                    [(r33[0, 0] - r33[1, 1] - r33[2, 2]) / 3, 
                        (r33[1, 0] + r33[0, 1]) / 3,             
                        (r33[2, 0] + r33[0, 2]) / 3,
                        (r33[1, 2] - r33[2, 1]) / 3
                    ],
                    [ (r33[1, 0] + r33[0, 1]) / 3,             
                        (r33[1, 1] - r33[0, 0] - r33[2, 2]) / 3, 
                        (r33[2, 1] + r33[1, 2]) / 3,
                        (r33[2, 0] - r33[0, 2]) / 3
                    ],
                    [(r33[2, 0] + r33[0, 2]) / 3,
                        (r33[2, 1] + r33[1, 2]) / 3,
                        (r33[2, 2] - r33[0, 0] - r33[1, 1]) / 3,
                        (r33[0, 1] - r33[1, 0]) / 3
                    ],
                    [(r33[1, 2] - r33[2, 1]) / 3,
                        (r33[2, 0] - r33[0, 2]) / 3,
                        (r33[0, 1] - r33[1, 0]) / 3,
                        (r33[0, 0] + r33[1, 1] + r33[2, 2]) / 3
                    ]
                ])

                eigvecs = np.linalg.eigh(K3.T)[1]
                res = eigvecs[:,-1]
                q = Quaternion(w=res[3], x=res[0], y=res[1], z=res[2])

            else:
                
                # ORTHONORMAL OPTION
                
                diagonals = np.array([
                    r33[0, 0], 
                    r33[1, 1], 
                    r33[2, 2], 
                    r33[0, 0] + r33[1, 1] + r33[2, 2]
                ])

                indices = np.argmax(diagonals, axis=-1)

                if indices == 3:
                    qw = 1 + r33[0,0] + r33[1,1] + r33[2,2]
                    qx =     r33[2,1] - r33[1,2]
                    qy =     r33[0,2] - r33[2,0]
                    qz =     r33[1,0] - r33[0,1]
                elif indices == 0:
                    qw =     r33[2,1] - r33[1,2]
                    qx = 1 + r33[0,0] - r33[1,1] - r33[2,2]
                    qy =     r33[0,1] + r33[1,0]
                    qz =     r33[0,2] + r33[2,0]
                elif indices == 1:
                    qw =     r33[0,2] - r33[2,0]
                    qx =     r33[1,0] + r33[0,1]
                    qy = 1 - r33[0,0] + r33[1,1] - r33[2,2]
                    qz =     r33[1,2] + r33[2,1]
                elif indices == 2:
                    qw =     r33[1,0] - r33[0,1]
                    qx =     r33[2,0] + r33[0,2]
                    qy =     r33[2,1] + r33[1,2]
                    qz = 1 - r33[0,0] - r33[1,1] + r33[2,2]

                q = Quaternion(qw, qx, qy, qz)

        else:
            raise TypeError("Unsupported operand type for m33toq: {}".format(type(r33)))

    q.normalize()

    return q

###############################################################################################

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
        if isinstance(x, numbers.Number):
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
 
    def __copy__(self):
        return Vector3D(self.x, self.y, self.z)
    
    def __bool__(self):
        return not self.isZero
    
    def __abs__(self):
        return Vector3D(abs(self.x), abs(self.y), abs(self.z))
    
    def __neg__(self):
        return Vector3D(-self.x, -self.y, -self.z)
    
    def __len__(self):
        return 3
            
    def __str__(self):
        return f"Vector3D({self.x}, {self.y}, {self.z})"

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        if isinstance(other, Vector3D):
            return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
        elif isinstance(other, numbers.Number):
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
        if isinstance(other, numbers.Number):
            return Vector3D(self.x * other, self.y * other, self.z * other)
        elif isinstance(other, Vector3D):
            return Vector3D(self.x * other.x, self.y * other.y, self.z * other.z)            
        elif isinstance(other, np.ndarray):
            shape = other.shape
            if len(shape) == 1:
                if shape[0] == 3:
                    return Vector3D(self.x * other[0], self.y * other[1], self.z * other[2])
                elif shape[0] == 4:
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
            elif shape == (3,3):
                '''Matrix Multiplication'''
                rotated_vector = np.dot(np.array([self.x,self.y,self.z]),other)
                return(Vector3D(x=rotated_vector[0], y=rotated_vector[1], z=rotated_vector[2]))
                
        elif isinstance(other, Quaternion):
                w = - self.x * other.x - self.y * other.y - self.z * other.z
                x =   self.x * other.w + self.y * other.z - self.z * other.y
                y = - self.x * other.z + self.y * other.w + self.z * other.x
                z =   self.x * other.y - self.y * other.x + self.z * other.w
                return Quaternion(w, x, y, z)
        else:
            raise TypeError("Unsupported operand type for *: Vector3D and {}".format(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if isinstance(other, numbers.Number):
            return Vector3D(self.x / other, self.y / other, self.z / other)
        elif isinstance(other, Vector3D):
            return Vector3D(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            raise ValueError("Unsupported operand type for /: {}".format(other))

    def __floordiv__(self, other):
        if isinstance(other, numbers.Number):
            return Vector3D(self.x // other, self.y // other, self.z // other)
        elif isinstance(other, Vector3D):
            return Vector3D(self.x // other.x, self.y // other.y, self.z // other.z)
        else:
            raise ValueError("Unsupported operand type for //: {}".format(other))

    def __pow__(self, other):
        '''potentiate'''
        if isinstance(other, numbers.Number):
            return Vector3D(self.x ** other, self.y ** other, self.z ** other)
        elif isinstance(other, Vector3D):
            return Vector3D(self.x ** other.x, self.y ** other.y, self.z ** other.z)
        else:
            raise TypeError("Unsupported operand type for **: Vector3D and {}".format(type(other)))

    def __eq__(self, other):
        '''are the two vectors equal'''
        if isinstance(other, numbers.Number):
            return Vector3D(self.x == other, self.y == other, self.z == other)
        elif isinstance(other, Vector3D):
            return Vector3D(self.x==other.x, self.y==other.y, self.z==other.z)
        else:
            raise TypeError("Unsupported operand type for ==: Vector3D and {}".format(type(other)))

    def __lt__(self, other):
        '''is vector smaller than other'''
        if isinstance(other, numbers.Number):
            return Vector3D(self.x < other, self.y < other, self.z < other)
        elif isinstance(other, Vector3D):
            return Vector3D(x=self.x < other.x, y=self.y<other.y, z=self.z<other.z)
        else:
            raise TypeError("Unsupported operand type for <=: Vector3D and {}".format(type(other)))

    def min(self, other):
        '''smaller component of the two vectors'''        
        if isinstance(other, numbers.Number):
            return Vector3D(min(self.x, other), min(self.y, other), min(self.z < other))
        elif isinstance(other, Vector3D):
            return Vector3D(x=min(self.x, other.x), y=min(self.y, other.y), z=min(self.z, other.z))
        else:
            raise TypeError("Unsupported operand type for min: Vector3D and {}".format(type(other)))

    def max(self, other):
        '''larger component of the two vectors'''        
        if isinstance(other, numbers.Number):
            return Vector3D(max(self.x, other), max(self.y, other), max(self.z < other))
        elif isinstance(other, Vector3D):
            return Vector3D(x=max(self.x, other.x), y=max(self.y, other.y), z=max(self.z, other.z))
        else:
            raise TypeError("Unsupported operand type for max: Vector3D and {}".format(type(other)))

    def abs(self):
        '''absolute'''        
        return Vector3D(x=abs(self.x), y=abs(self.y), z=abs(self.z))

    def normalize(self):
        mag = self.norm
        if mag != 0:
            self.x = self.x/mag
            self.y = self.y/mag
            self.z = self.z/mag

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
        '''
        u × v = [u2v3 - u3v2, u3v1 - u1v3, u1v2 - u2v1]
        x = u2v3 - u3v2
        y = u3v1 - u1v3
        z = u1v2 - u2v1
        '''
        if isinstance(other, Vector3D):
            x = (self.y * other.z) - (self.z * other.y)
            y = (self.z * other.x) - (self.x * other.z)
            z = (self.x * other.y) - (self.y * other.x)
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
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)

    @property
    def isZero(self) -> bool:
        return (abs(self.x) <= EPSILON and abs(self.y) <= EPSILON and abs(self.z) <= EPSILON)

