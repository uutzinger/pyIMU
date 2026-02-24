# cython: language_level=3

import numpy as np
cimport numpy as cnp
from libc.math cimport sqrt


cpdef tuple quaternion_multiply(
    double aw, double ax, double ay, double az,
    double bw, double bx, double by, double bz
):
    cdef double w = (aw * bw) - (ax * bx) - (ay * by) - (az * bz)
    cdef double x = (aw * bx) + (ax * bw) + (ay * bz) - (az * by)
    cdef double y = (aw * by) - (ax * bz) + (ay * bw) + (az * bx)
    cdef double z = (aw * bz) + (ax * by) - (ay * bx) + (az * bw)
    return (w, x, y, z)


cpdef tuple quaternion_times_vector(
    double qw, double qx, double qy, double qz,
    double vx, double vy, double vz
):
    cdef double w = -(qx * vx) - (qy * vy) - (qz * vz)
    cdef double x =  (qw * vx) + (qy * vz) - (qz * vy)
    cdef double y =  (qw * vy) - (qx * vz) + (qz * vx)
    cdef double z =  (qw * vz) + (qx * vy) - (qy * vx)
    return (w, x, y, z)


cpdef tuple normalize_quaternion(double w, double x, double y, double z):
    cdef double mag = sqrt(w*w + x*x + y*y + z*z)
    if mag == 0.0:
        return (w, x, y, z)
    return (w/mag, x/mag, y/mag, z/mag)


cpdef cnp.ndarray quaternion_to_r33(double w, double x, double y, double z):
    cdef double xx = x * x
    cdef double xy = x * y
    cdef double xz = x * z
    cdef double xw = x * w
    cdef double yy = y * y
    cdef double yz = y * z
    cdef double yw = y * w
    cdef double zz = z * z
    cdef double zw = z * w

    cdef cnp.ndarray[cnp.double_t, ndim=2] out = np.empty((3, 3), dtype=np.float64)
    out[0, 0] = 1.0 - 2.0 * (yy + zz)
    out[0, 1] =       2.0 * (xy - zw)
    out[0, 2] =       2.0 * (xz + yw)
    out[1, 0] =       2.0 * (xy + zw)
    out[1, 1] = 1.0 - 2.0 * (xx + zz)
    out[1, 2] =       2.0 * (yz - xw)
    out[2, 0] =       2.0 * (xz - yw)
    out[2, 1] =       2.0 * (yz + xw)
    out[2, 2] = 1.0 - 2.0 * (xx + yy)
    return out


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

