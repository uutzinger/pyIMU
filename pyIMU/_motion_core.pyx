# cython: language_level=3

cpdef tuple motion_step(
    double accx, double accy, double accz,
    double qw, double qx, double qy, double qz,
    double r00, double r01, double r02,
    double r10, double r11, double r12,
    double r20, double r21, double r22,
    double local_gravity,
    double bias_x, double bias_y, double bias_z,
    double wr_prev_x, double wr_prev_y, double wr_prev_z,
    double wv_x, double wv_y, double wv_z,
    double wv_prev_x, double wv_prev_y, double wv_prev_z,
    double wp_prev_x, double wp_prev_y, double wp_prev_z,
    double wv_drift_x, double wv_drift_y, double wv_drift_z,
    double drift_alpha,
    double dt,
    double dtmotion,
    bint moving,
    bint motion_ended,
    double min_motion_time
):
    cdef double gx, gy, gz
    cdef double residual_x, residual_y, residual_z
    cdef double wr_x, wr_y, wr_z
    cdef double wp_x, wp_y, wp_z
    cdef double one_minus_alpha = 1.0 - drift_alpha
    cdef double scale

    # q2gravity + sensor residual on sensor frame
    gx = 2.0 * (qx * qz - qw * qy)
    gy = 2.0 * (qy * qz + qw * qx)
    gz = 1.0 - 2.0 * (qx * qx + qy * qy)

    residual_x = accx - (local_gravity * gx) - bias_x
    residual_y = accy - (local_gravity * gy) - bias_y
    residual_z = accz - (local_gravity * gz) - bias_z

    # residuals in world frame: r33 * residual
    wr_x = r00 * residual_x + r01 * residual_y + r02 * residual_z
    wr_y = r10 * residual_x + r11 * residual_y + r12 * residual_z
    wr_z = r20 * residual_x + r21 * residual_y + r22 * residual_z

    if moving:
        # trapezoidal integrate acceleration -> velocity
        scale = 0.5 * dt
        wv_x = wv_prev_x + (wr_x + wr_prev_x) * scale
        wv_y = wv_prev_y + (wr_y + wr_prev_y) * scale
        wv_z = wv_prev_z + (wr_z + wr_prev_z) * scale

        dtmotion += dt

        # subtract learned drift
        wv_x -= wv_drift_x * dt
        wv_y -= wv_drift_y * dt
        wv_z -= wv_drift_z * dt

        # trapezoidal integrate velocity -> position
        wp_x = wp_prev_x + (wv_x + wv_prev_x) * scale
        wp_y = wp_prev_y + (wv_y + wv_prev_y) * scale
        wp_z = wp_prev_z + (wv_z + wv_prev_z) * scale

        wr_prev_x = wr_x
        wr_prev_y = wr_y
        wr_prev_z = wr_z

        wv_prev_x = wv_x
        wv_prev_y = wv_y
        wv_prev_z = wv_z

        wp_prev_x = wp_x
        wp_prev_y = wp_y
        wp_prev_z = wp_z
    else:
        # update drift estimate on motion stop
        if motion_ended and dtmotion > min_motion_time:
            scale = drift_alpha / dtmotion
            wv_drift_x = wv_drift_x * one_minus_alpha + wv_x * scale
            wv_drift_y = wv_drift_y * one_minus_alpha + wv_y * scale
            wv_drift_z = wv_drift_z * one_minus_alpha + wv_z * scale
            dtmotion = 0.0

        wr_prev_x = 0.0
        wr_prev_y = 0.0
        wr_prev_z = 0.0

        wv_x = 0.0
        wv_y = 0.0
        wv_z = 0.0
        wv_prev_x = 0.0
        wv_prev_y = 0.0
        wv_prev_z = 0.0

        # update residual bias (expected residual ~0 when stationary)
        bias_x = bias_x * one_minus_alpha + residual_x * drift_alpha
        bias_y = bias_y * one_minus_alpha + residual_y * drift_alpha
        bias_z = bias_z * one_minus_alpha + residual_z * drift_alpha

    return (
        residual_x, residual_y, residual_z,
        wr_x, wr_y, wr_z,
        wr_prev_x, wr_prev_y, wr_prev_z,
        wv_x, wv_y, wv_z,
        wv_prev_x, wv_prev_y, wv_prev_z,
        wp_prev_x, wp_prev_y, wp_prev_z,
        wv_drift_x, wv_drift_y, wv_drift_z,
        bias_x, bias_y, bias_z,
        dtmotion,
    )


cpdef tuple motion_step_quat(
    double accx, double accy, double accz,
    double qw, double qx, double qy, double qz,
    double local_gravity,
    double bias_x, double bias_y, double bias_z,
    double wr_prev_x, double wr_prev_y, double wr_prev_z,
    double wv_x, double wv_y, double wv_z,
    double wv_prev_x, double wv_prev_y, double wv_prev_z,
    double wp_prev_x, double wp_prev_y, double wp_prev_z,
    double wv_drift_x, double wv_drift_y, double wv_drift_z,
    double drift_alpha,
    double dt,
    double dtmotion,
    bint moving,
    bint motion_ended,
    double min_motion_time
):
    cdef double xx, xy, xz, xw, yy, yz, yw, zz, zw
    cdef double r00, r01, r02
    cdef double r10, r11, r12
    cdef double r20, r21, r22

    # Quaternion -> rotation matrix (sensor to world)
    xx = qx * qx
    xy = qx * qy
    xz = qx * qz
    xw = qx * qw
    yy = qy * qy
    yz = qy * qz
    yw = qy * qw
    zz = qz * qz
    zw = qz * qw

    r00 = 1.0 - 2.0 * (yy + zz)
    r01 = 2.0 * (xy - zw)
    r02 = 2.0 * (xz + yw)
    r10 = 2.0 * (xy + zw)
    r11 = 1.0 - 2.0 * (xx + zz)
    r12 = 2.0 * (yz - xw)
    r20 = 2.0 * (xz - yw)
    r21 = 2.0 * (yz + xw)
    r22 = 1.0 - 2.0 * (xx + yy)

    return motion_step(
        accx, accy, accz,
        qw, qx, qy, qz,
        r00, r01, r02,
        r10, r11, r12,
        r20, r21, r22,
        local_gravity,
        bias_x, bias_y, bias_z,
        wr_prev_x, wr_prev_y, wr_prev_z,
        wv_x, wv_y, wv_z,
        wv_prev_x, wv_prev_y, wv_prev_z,
        wp_prev_x, wp_prev_y, wp_prev_z,
        wv_drift_x, wv_drift_y, wv_drift_z,
        drift_alpha,
        dt,
        dtmotion,
        moving,
        motion_ended,
        min_motion_time
    )
