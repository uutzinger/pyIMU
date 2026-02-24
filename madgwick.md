# Madgwick Filter API

Module: `pyIMU.madgwick`

## Class
`Madgwick(**kwargs)`

Madgwick gradient-descent AHRS filter (NED convention).

Initialization Parameters
- `dt`: `float`, default `0.01`
  - Default time step used when `update(..., dt<=0)`.
- `gain`: `float`, optional
  - Convenience gain applied to both IMU and MARG unless overridden.
- `gain_imu`: `float`, default `0.033` (or `gain` if provided)
  - Gain used for IMU updates (`gyr+acc`).
- `gain_marg`: `float`, default `0.041` (or `gain` if provided)
  - Gain used for MARG updates (`gyr+acc+mag`).
- `acc_in_g`: `bool`, default `True`
  - `True`: accelerometer input expected in `g`.
  - `False`: input treated as `m/s^2` and converted to `g`.
- `gyr_in_dps`: `bool`, default `False`
  - `True`: gyroscope input expected in `deg/s` and converted to `rad/s`.
- `convention`: `str`, default `"NED"`
  - Only `NED` is currently supported.

Methods
- `update(gyr, acc, mag=None, dt=-1) -> Quaternion`
  - `gyr`: `Vector3D` gyro sample.
  - `acc`: `Vector3D` accel sample.
  - `mag`: optional `Vector3D` magnetometer sample.
  - `dt`: time step in seconds. If `<=0`, uses instance `dt`.

Behavior:
- First call initializes orientation from accel (`accel2q`) or accel+mag (`accelmag2q`).
- Later calls run IMU or MARG update equations.

## Functional API
- `updateIMU(q, gyr, acc, dt, gain) -> Quaternion`
- `updateMARG(q, gyr, acc, mag, dt, gain) -> Quaternion`
- `updateIMU_inplace(...) -> Quaternion`
- `updateMARG_inplace(...) -> Quaternion`

Use inplace variants to mutate and reuse an existing quaternion object.
