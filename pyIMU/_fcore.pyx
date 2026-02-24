# cython: language_level=3

from libc.math cimport sqrt


cpdef tuple integrate_quaternion_step(
    double qw, double qx, double qy, double qz,
    double gx, double gy, double gz,
    double bx, double by, double bz,
    double fx, double fy, double fz,
    double gain,
    double dt
):
    """
    Integrate quaternion using Fusion-style corrected gyro term:
      half_gyr = 0.5 * (gyro - bias) + gain * feedback
      q_dot = q * [0, half_gyr]
    """
    cdef double hgx, hgy, hgz
    cdef double qdw, qdx, qdy, qdz
    cdef double qnorm, inv_qnorm

    hgx = 0.5 * (gx - bx) + gain * fx
    hgy = 0.5 * (gy - by) + gain * fy
    hgz = 0.5 * (gz - bz) + gain * fz

    qdw = -qx * hgx - qy * hgy - qz * hgz
    qdx =  qw * hgx + qy * hgz - qz * hgy
    qdy =  qw * hgy - qx * hgz + qz * hgx
    qdz =  qw * hgz + qx * hgy - qy * hgx

    qw += qdw * dt
    qx += qdx * dt
    qy += qdy * dt
    qz += qdz * dt

    qnorm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if qnorm > 0.0:
        inv_qnorm = 1.0 / qnorm
        qw *= inv_qnorm
        qx *= inv_qnorm
        qy *= inv_qnorm
        qz *= inv_qnorm

    return (qw, qx, qy, qz)
