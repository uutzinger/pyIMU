# cython: language_level=3

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt


cpdef double vector_norm(double x, double y, double z):
    return sqrt(x*x + y*y + z*z)


cpdef tuple normalize_vector(double x, double y, double z):
    cdef double mag = sqrt(x*x + y*y + z*z)
    if mag == 0.0:
        return (x, y, z)
    return (x/mag, y/mag, z/mag)


cpdef double vector_dot(
    double ax, double ay, double az,
    double bx, double by, double bz
):
    return ax*bx + ay*by + az*bz


cpdef tuple vector_cross(
    double ax, double ay, double az,
    double bx, double by, double bz
):
    cdef double x = (ay * bz) - (az * by)
    cdef double y = (az * bx) - (ax * bz)
    cdef double z = (ax * by) - (ay * bx)
    return (x, y, z)


cpdef tuple rotate_vector_r33(
    double vx, double vy, double vz,
    cnp.ndarray[cnp.double_t, ndim=2] r33
):
    if r33.shape[0] != 3 or r33.shape[1] != 3:
        raise ValueError("Rotation matrix must have shape (3, 3).")

    cdef double x = r33[0, 0] * vx + r33[0, 1] * vy + r33[0, 2] * vz
    cdef double y = r33[1, 0] * vx + r33[1, 1] * vy + r33[1, 2] * vz
    cdef double z = r33[2, 0] * vx + r33[2, 1] * vy + r33[2, 2] * vz
    return (x, y, z)

