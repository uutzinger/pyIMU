# Fusion Filter API

Module: `pyIMU.fusion`

## Class
`Fusion(**kwargs)`

Fusion-style AHRS with gain ramp, gyro bias estimation, and rejection/recovery logic (NED convention).

Initialization Parameters
- `dt`: `float`, default `0.01`
  - Default time step when `update(..., dt<=0)`.
- `k_init`: `float`, default `10.0`
  - Initial high gain during startup.
- `k_normal`: `float`, default `0.5` (or `gain` if provided)
  - Steady-state gain.
- `gain`: `float`, optional alias for `k_normal`.
- `t_init`: `float`, default `3.0`
  - Ramp duration from `k_init` to `k_normal`.
- `gyr_in_dps`: `bool`, default `False`
  - If `True`, gyro input is `deg/s`; converted to `rad/s`.
- `gyr_range`: `float`, default `2000.0`
  - Full-scale gyro range used for angular-rate recovery threshold.
- `fc_bias`: `float`, default `0.05`
  - Bias low-pass filter cutoff factor.
- `omega_min`: `float`, default `0.35`
  - Stationary gyro threshold for bias learning (`rad/s`).
- `t_bias`: `float`, default `1.0`
  - Required stationary duration before bias update.
- `acc_in_g`: `bool`, default `True`
  - If `False`, accel input is `m/s^2`; converted to `g`.
- `g_deviation`: `float`, default `0.10`
  - Acceptable deviation from `1 g` for accel acceptance.
- `acceleration_rejection`: `float`, default `12.0` (degrees)
  - Angular error threshold for accel rejection.
- `magnetic_rejection`: `float`, default `12.0` (degrees)
  - Angular error threshold for mag rejection.
- `mag_min`: `float | None`, default `None`
  - Optional lower bound on magnetometer norm.
- `mag_max`: `float | None`, default `None`
  - Optional upper bound on magnetometer norm.
- `t_acc_reject`: `float`, default `0.5`
  - Accel rejection trigger period (seconds).
- `t_mag_reject`: `float`, default `0.5`
  - Mag rejection trigger period (seconds).
- `recovery_timeout_factor`: `int`, default `5`
  - Timeout multiplier for rejection recovery.
- `convention`: `str`, default `"NED"`
  - Only `NED` is currently supported.

Methods
- `update(gyr, acc, mag=None, dt=-1.0) -> Quaternion`
  - Updates internal state and returns a copy of quaternion.
- `update_inplace(gyr, acc, mag=None, dt=-1.0) -> Quaternion`
  - Updates internal state and returns internal quaternion reference.

Update Parameters
- `gyr`: `Vector3D` gyroscope sample.
- `acc`: `Vector3D` accelerometer sample.
- `mag`: optional `Vector3D` magnetometer sample.
- `dt`: seconds; if `<=0`, uses instance `dt`.

Output State Fields
- `q`: current orientation quaternion.
- `gyro_bias`: estimated gyro bias.
- `azero`: acceleration with gravity removed (sensor frame, in `g`).
- `aglobal`: gravity-free acceleration in Earth frame (in `g`).
- `accelerometer_ignored`, `magnetometer_ignored`, `angular_rate_recovery`: status flags.
