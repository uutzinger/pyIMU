# cython: language_level=3

from libc.math cimport sqrt


cpdef tuple update_imu_step(
    double qw, double qx, double qy, double qz,
    double gx, double gy, double gz,
    double ax, double ay, double az,
    double dt, double gain
):
    cdef double qDotw, qDotx, qDoty, qDotz
    cdef double f0, f1, f2
    cdef double g0, g1, g2, g3
    cdef double gnorm, inv_gnorm
    cdef double qnorm, inv_qnorm

    # Estimated orientation change from gyroscope              # (eq. 12)
    qDotw = 0.5 * (-qx * gx - qy * gy - qz * gz)
    qDotx = 0.5 * ( qw * gx + qy * gz - qz * gy)
    qDoty = 0.5 * ( qw * gy - qx * gz + qz * gx)
    qDotz = 0.5 * ( qw * gz + qx * gy - qy * gx)

    # Objective function                                       # (eq. 25)
    f0 = 2.0 * (qx * qz - qw * qy) - ax
    f1 = 2.0 * (qw * qx + qy * qz) - ay
    f2 = 2.0 * (0.5 - qx * qx - qy * qy) - az

    if sqrt(f0 * f0 + f1 * f1 + f2 * f2) > 0.0:
        # Sensitivity matrix: gradient = J.T @ f              # (eq. 34)
        g0 = (-2.0 * qy) * f0 + (2.0 * qx) * f1
        g1 = ( 2.0 * qz) * f0 + (2.0 * qw) * f1 + (-4.0 * qx) * f2
        g2 = (-2.0 * qw) * f0 + (2.0 * qz) * f1 + (-4.0 * qy) * f2
        g3 = ( 2.0 * qx) * f0 + (2.0 * qy) * f1

        # keep normalization behavior consistent with previous implementation
        gnorm = sqrt(g0 * g0 + g1 * g1 + g2 * g2)
        if gnorm > 0.0:
            inv_gnorm = 1.0 / gnorm
            g0 *= inv_gnorm
            g1 *= inv_gnorm
            g2 *= inv_gnorm
            g3 *= inv_gnorm

            # Update orientation change                        # (eq. 33)
            qDotw -= gain * g0
            qDotx -= gain * g1
            qDoty -= gain * g2
            qDotz -= gain * g3

    # Update orientation                                       # (eq. 13)
    qw += qDotw * dt
    qx += qDotx * dt
    qy += qDoty * dt
    qz += qDotz * dt

    qnorm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if qnorm > 0.0:
        inv_qnorm = 1.0 / qnorm
        qw *= inv_qnorm
        qx *= inv_qnorm
        qy *= inv_qnorm
        qz *= inv_qnorm

    return (qw, qx, qy, qz)


cpdef tuple update_marg_step(
    double qw, double qx, double qy, double qz,
    double gx, double gy, double gz,
    double ax, double ay, double az,
    double mx, double my, double mz,
    double dt, double gain
):
    cdef double qDotw, qDotx, qDoty, qDotz
    cdef double mw, mx1, my1, mz1
    cdef double hx, hy, hz, bx, bz
    cdef double f0, f1, f2, f3, f4, f5
    cdef double g0, g1, g2, g3
    cdef double gnorm, inv_gnorm
    cdef double qnorm, inv_qnorm

    # Estimated orientation change from gyroscope              # (eq. 12)
    qDotw = 0.5 * (-qx * gx - qy * gy - qz * gz)
    qDotx = 0.5 * ( qw * gx + qy * gz - qz * gy)
    qDoty = 0.5 * ( qw * gy - qx * gz + qz * gx)
    qDotz = 0.5 * ( qw * gz + qx * gy - qy * gx)

    # Rotate normalized magnetometer measurements
    # h = q * mag * q.conjugate                                 # (eq. 45)
    mw = -(qx * mx + qy * my + qz * mz)
    mx1 = qw * mx + qy * mz - qz * my
    my1 = qw * my - qx * mz + qz * mx
    mz1 = qw * mz + qx * my - qy * mx

    hx = -mw * qx + mx1 * qw - my1 * qz + mz1 * qy
    hy = -mw * qy + mx1 * qz + my1 * qw - mz1 * qx
    hz = -mw * qz - mx1 * qy + my1 * qx + mz1 * qw

    bx = sqrt(hx * hx + hy * hy)                               # (eq. 46)
    bz = hz

    # Objective function                                        # (eq. 31)
    f0 = 2.0 * (qx * qz - qw * qy)                                             - ax
    f1 = 2.0 * (qw * qx + qy * qz)                                             - ay
    f2 = 2.0 * (0.5 - qx * qx - qy * qy)                                       - az
    f3 = 2.0 * bx * (0.5 - qy * qy - qz * qz) + 2.0 * bz * (qx * qz - qw * qy) - mx
    f4 = 2.0 * bx * (qx * qy - qw * qz)       + 2.0 * bz * (qw * qx + qy * qz) - my
    f5 = 2.0 * bx * (qw * qy + qx * qz)       + 2.0 * bz * (0.5 - qx * qx - qy * qy) - mz

    if sqrt(f0 * f0 + f1 * f1 + f2 * f2 + f3 * f3 + f4 * f4 + f5 * f5) > 0.0:
        # Sensitivity matrix: gradient = J.T @ f               # (eq. 34)
        g0 = (-2.0 * qy) * f0 + (2.0 * qx) * f1 + (-2.0 * bz * qy) * f3 + (-2.0 * bx * qz + 2.0 * bz * qx) * f4 + (2.0 * bx * qy) * f5
        g1 = (2.0 * qz) * f0 + (2.0 * qw) * f1 + (-4.0 * qx) * f2 + (2.0 * bz * qz) * f3 + (2.0 * bx * qy + 2.0 * bz * qw) * f4 + (2.0 * bx * qz - 4.0 * bz * qx) * f5
        g2 = (-2.0 * qw) * f0 + (2.0 * qz) * f1 + (-4.0 * qy) * f2 + (-4.0 * bx * qy - 2.0 * bz * qw) * f3 + (2.0 * bx * qx + 2.0 * bz * qz) * f4 + (2.0 * bx * qw - 4.0 * bz * qy) * f5
        g3 = (2.0 * qx) * f0 + (2.0 * qy) * f1 + (-4.0 * bx * qz + 2.0 * bz * qx) * f3 + (-2.0 * bx * qw + 2.0 * bz * qy) * f4 + (2.0 * bx * qx) * f5

        gnorm = sqrt(g0 * g0 + g1 * g1 + g2 * g2 + g3 * g3)
        if gnorm > 0.0:
            inv_gnorm = 1.0 / gnorm
            g0 *= inv_gnorm
            g1 *= inv_gnorm
            g2 *= inv_gnorm
            g3 *= inv_gnorm

            # Updated orientation change                        # (eq. 33)
            qDotw -= gain * g0
            qDotx -= gain * g1
            qDoty -= gain * g2
            qDotz -= gain * g3

    # Update orientation                                        # (eq. 13)
    qw += qDotw * dt
    qx += qDotx * dt
    qy += qDoty * dt
    qz += qDotz * dt

    qnorm = sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if qnorm > 0.0:
        inv_qnorm = 1.0 / qnorm
        qw *= inv_qnorm
        qx *= inv_qnorm
        qy *= inv_qnorm
        qz *= inv_qnorm

    return (qw, qx, qy, qz)

