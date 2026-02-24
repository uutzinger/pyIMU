# cython: language_level=3

cpdef tuple q2gravity_components(double qw, double qx, double qy, double qz):
    cdef double gx, gy, gz

    gx = 2.0 * (qx * qz - qw * qy)
    gy = 2.0 * (qy * qz + qw * qx)
    gz = 1.0 - 2.0 * (qx * qx + qy * qy)

    return (gx, gy, gz)
