# cython: language_level=3

from libc.math cimport sqrt, asin, sin, fabs, round

cdef double EPSILON = 1e-15
cdef double TWO_PI = 6.283185307179586476925286766559


cdef inline double clampd(double v, double lo, double hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


cdef inline int clipi(int v, int hi):
    if v < 0:
        return 0
    if v > hi:
        return hi
    return v


cpdef tuple integrate_quaternion_step(
    double qw, double qx, double qy, double qz,
    double gx, double gy, double gz,
    double bx, double by, double bz,
    double fx, double fy, double fz,
    double gain,
    double dt
):
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


cpdef tuple fusion_step(
    double qw, double qx, double qy, double qz,
    double gx, double gy, double gz,
    double ax, double ay, double az,
    bint mag_present,
    double mx, double my, double mz,
    double bias_x, double bias_y, double bias_z,
    double dt,
    double k_normal,
    double ramped_gain,
    double gain_ramp_rate,
    bint initialising,
    double gyr_range_rad,
    double omega_min,
    double fc_bias,
    double t_bias,
    double g_deviation,
    double acceleration_rejection,
    double magnetic_rejection,
    bint mag_min_enabled,
    double mag_min,
    bint mag_max_enabled,
    double mag_max,
    double t_acc_reject,
    double t_mag_reject,
    int recovery_timeout_factor,
    int acc_trigger,
    int acc_timeout,
    int mag_trigger,
    int mag_timeout,
    double stationary_time
):
    cdef double rate_limit
    cdef bint angular_rate_recovery
    cdef double fbx = 0.0
    cdef double fby = 0.0
    cdef double fbz = 0.0

    cdef double hgx, hgy, hgz
    cdef double acc_norm, acc_mag_error
    cdef double acc_ux, acc_uy, acc_uz
    cdef double half_acc_x, half_acc_y, half_acc_z
    cdef double half_acc_norm
    cdef double acceleration_error = 0.0
    cdef bint accelerometer_ignored = True
    cdef bint acc_accepted
    cdef int trigger_period, timeout_period

    cdef double mag_norm_raw
    cdef double mag_ux, mag_uy, mag_uz
    cdef double qcw, qcx, qcy, qcz
    cdef double tw, tx, ty, tz
    cdef double hw, hx, hy, hz
    cdef double bx, bz
    cdef double refx, refy, refz, ref_norm
    cdef double half_mag_x, half_mag_y, half_mag_z
    cdef double half_mag_norm
    cdef double magnetic_error = 0.0
    cdef bint magnetometer_ignored = True
    cdef bint mag_strength_ok = True
    cdef bint mag_accepted

    cdef double gyr_norm
    cdef bint is_stationary
    cdef double alpha

    cdef double gain
    cdef double cgx, cgy, cgz
    cdef double qdw, qdx, qdy, qdz
    cdef double qnorm, inv_qnorm

    cdef double gsx, gsy, gsz
    cdef double azx, azy, azz
    cdef double agx, agy, agz

    # initialisation ramp
    if initialising and gain_ramp_rate > 0.0:
        ramped_gain -= gain_ramp_rate * dt
        if ramped_gain <= k_normal:
            ramped_gain = k_normal
            initialising = False

    # angular-rate recovery
    rate_limit = 0.98 * gyr_range_rad
    angular_rate_recovery = (fabs(gx) > rate_limit) or (fabs(gy) > rate_limit) or (fabs(gz) > rate_limit)

    # half gravity from quaternion
    hgx = qx * qz - qw * qy
    hgy = qw * qx + qy * qz
    hgz = qw * qw - 0.5 + qz * qz

    # accelerometer rejection/recovery
    acc_norm = sqrt(ax * ax + ay * ay + az * az)
    acc_mag_error = fabs(acc_norm - 1.0)
    if acc_norm > EPSILON and not angular_rate_recovery:
        acc_ux = ax / acc_norm
        acc_uy = ay / acc_norm
        acc_uz = az / acc_norm

        half_acc_x = acc_uy * hgz - acc_uz * hgy
        half_acc_y = acc_uz * hgx - acc_ux * hgz
        half_acc_z = acc_ux * hgy - acc_uy * hgx

        half_acc_norm = sqrt(half_acc_x * half_acc_x + half_acc_y * half_acc_y + half_acc_z * half_acc_z)
        acceleration_error = asin(clampd(2.0 * half_acc_norm, -1.0, 1.0))

        acc_accepted = (acc_mag_error <= g_deviation) and (acceleration_error <= acceleration_rejection)
        trigger_period = <int>round(t_acc_reject / dt)
        if trigger_period < 1:
            trigger_period = 1
        timeout_period = recovery_timeout_factor * trigger_period

        if initialising or acc_accepted:
            acc_trigger -= 9
            accelerometer_ignored = False
        else:
            acc_trigger += 1
            accelerometer_ignored = True

        if acc_trigger > trigger_period:
            acc_timeout = 0
            accelerometer_ignored = False
        elif acc_timeout > timeout_period:
            acc_trigger = 0
            accelerometer_ignored = False
        else:
            acc_timeout += 1

        if not initialising:
            acc_trigger = clipi(acc_trigger, trigger_period)
            acc_timeout = clipi(acc_timeout, timeout_period)

        if not accelerometer_ignored:
            fbx += half_acc_x
            fby += half_acc_y
            fbz += half_acc_z
    else:
        accelerometer_ignored = True
        acceleration_error = 0.0

    # magnetometer rejection/recovery
    if mag_present and not angular_rate_recovery:
        mag_norm_raw = sqrt(mx * mx + my * my + mz * mz)
        if mag_norm_raw > EPSILON:
            mag_ux = mx / mag_norm_raw
            mag_uy = my / mag_norm_raw
            mag_uz = mz / mag_norm_raw

            # h = q.conjugate * [0,mag] * q
            qcw = qw
            qcx = -qx
            qcy = -qy
            qcz = -qz

            tw = -(qcx * mag_ux + qcy * mag_uy + qcz * mag_uz)
            tx = qcw * mag_ux + qcy * mag_uz - qcz * mag_uy
            ty = qcw * mag_uy - qcx * mag_uz + qcz * mag_ux
            tz = qcw * mag_uz + qcx * mag_uy - qcy * mag_ux

            hw = tw * qw - tx * qx - ty * qy - tz * qz
            hx = tw * qx + tx * qw + ty * qz - tz * qy
            hy = tw * qy - tx * qz + ty * qw + tz * qx
            hz = tw * qz + tx * qy - ty * qx + tz * qw

            bx = sqrt(hx * hx + hy * hy)
            bz = hz

            refx = 2.0 * bx * (0.5 - qy * qy - qz * qz) + 2.0 * bz * (qx * qz - qw * qy)
            refy = 2.0 * bx * (qx * qy - qw * qz) + 2.0 * bz * (qw * qx + qy * qz)
            refz = 2.0 * bx * (qw * qy + qx * qz) + 2.0 * bz * (0.5 - qx * qx - qy * qy)

            ref_norm = sqrt(refx * refx + refy * refy + refz * refz)
            if ref_norm > EPSILON:
                refx /= ref_norm
                refy /= ref_norm
                refz /= ref_norm

            half_mag_x = mag_uy * refz - mag_uz * refy
            half_mag_y = mag_uz * refx - mag_ux * refz
            half_mag_z = mag_ux * refy - mag_uy * refx

            half_mag_norm = sqrt(half_mag_x * half_mag_x + half_mag_y * half_mag_y + half_mag_z * half_mag_z)
            magnetic_error = asin(clampd(2.0 * half_mag_norm, -1.0, 1.0))

            mag_strength_ok = True
            if mag_min_enabled and mag_norm_raw < mag_min:
                mag_strength_ok = False
            if mag_max_enabled and mag_norm_raw > mag_max:
                mag_strength_ok = False

            mag_accepted = mag_strength_ok and (magnetic_error <= magnetic_rejection)
            trigger_period = <int>round(t_mag_reject / dt)
            if trigger_period < 1:
                trigger_period = 1
            timeout_period = recovery_timeout_factor * trigger_period

            if initialising or mag_accepted:
                mag_trigger -= 9
                magnetometer_ignored = False
            else:
                mag_trigger += 1
                magnetometer_ignored = True

            if mag_trigger > trigger_period:
                mag_timeout = 0
                magnetometer_ignored = False
            elif mag_timeout > timeout_period:
                mag_trigger = 0
                magnetometer_ignored = False
            else:
                mag_timeout += 1

            if not initialising:
                mag_trigger = clipi(mag_trigger, trigger_period)
                mag_timeout = clipi(mag_timeout, timeout_period)

            if not magnetometer_ignored:
                fbx += half_mag_x
                fby += half_mag_y
                fbz += half_mag_z

    if angular_rate_recovery:
        accelerometer_ignored = True
        magnetometer_ignored = True

    # bias update
    gyr_norm = sqrt(gx * gx + gy * gy + gz * gz)
    is_stationary = (gyr_norm <= omega_min) and (acc_mag_error <= g_deviation)
    if is_stationary:
        stationary_time += dt
    else:
        stationary_time = 0.0

    if stationary_time >= t_bias:
        alpha = clampd(TWO_PI * fc_bias * dt, 0.0, 1.0)
        bias_x += alpha * (gx - bias_x)
        bias_y += alpha * (gy - bias_y)
        bias_z += alpha * (gz - bias_z)

    gain = ramped_gain if initialising else k_normal

    # integrate quaternion
    cgx = 0.5 * (gx - bias_x) + gain * fbx
    cgy = 0.5 * (gy - bias_y) + gain * fby
    cgz = 0.5 * (gz - bias_z) + gain * fbz

    qdw = -qx * cgx - qy * cgy - qz * cgz
    qdx =  qw * cgx + qy * cgz - qz * cgy
    qdy =  qw * cgy - qx * cgz + qz * cgx
    qdz =  qw * cgz + qx * cgy - qy * cgx

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

    # optional acceleration outputs
    gsx = 2.0 * (qx * qz - qw * qy)
    gsy = 2.0 * (qw * qx + qy * qz)
    gsz = qw * qw - qx * qx - qy * qy + qz * qz

    azx = ax - gsx
    azy = ay - gsy
    azz = az - gsz

    # aglobal = q.conjugate * [0,azero] * q
    qcw = qw
    qcx = -qx
    qcy = -qy
    qcz = -qz

    tw = -(qcx * azx + qcy * azy + qcz * azz)
    tx = qcw * azx + qcy * azz - qcz * azy
    ty = qcw * azy - qcx * azz + qcz * azx
    tz = qcw * azz + qcx * azy - qcy * azx

    agx = tw * qx + tx * qw + ty * qz - tz * qy
    agy = tw * qy - tx * qz + ty * qw + tz * qx
    agz = tw * qz + tx * qy - ty * qx + tz * qw

    return (
        qw, qx, qy, qz,
        bias_x, bias_y, bias_z,
        acc_trigger, acc_timeout, mag_trigger, mag_timeout,
        stationary_time, ramped_gain, <int>initialising,
        <int>angular_rate_recovery, <int>accelerometer_ignored, <int>magnetometer_ignored,
        acceleration_error, magnetic_error,
        azx, azy, azz, agx, agy, agz
    )
