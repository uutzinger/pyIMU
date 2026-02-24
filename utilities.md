# Utilities API

Module: `pyIMU.utilities`

## Scalar Helpers
- `clip(val, largest)`
  - Clips scalar to interval `[0, largest]`.
- `clamp(val, smallest, largest)`
  - Clips scalar to interval `[smallest, largest]`.
- `asin(value)`
  - Safe arcsin with saturation at `[-pi/2, pi/2]`.
- `invSqrt(value)`
  - Fast inverse square root approximation (kept for compatibility).

## RunningAverage
Class: `RunningAverage(window_size)`

Parameters:
- `window_size`: `int > 0`
  - Maximum number of samples kept in the moving window.

Methods and Properties:
- `update(value)`
  - `value`: scalar or `Vector3D`.
- `avg`
  - Mean over current window.
- `var`
  - Variance over current window.

## Orientation Conversion
- `vector_angle2q(vec, angle=0.0) -> Quaternion`
  - `vec`: rotation axis (`Vector3D`).
  - `angle`: angle in radians.
- `q2rpy(q) -> Vector3D`
  - Quaternion to roll/pitch/yaw (radians).
- `rpy2q(r, p=0.0, y=0.0) -> Quaternion`
  - Inputs can be `Vector3D`, length-3 array-like, or scalar triplet.

## Initial Pose Estimation
- `accel2rpy(acc) -> Vector3D`
- `accel2q(acc) -> Quaternion`
- `accelmag2rpy(acc, mag) -> Vector3D`
- `accelmag2q(acc, mag) -> Quaternion`

Parameter expectations:
- `acc`, `mag`: `Vector3D` or length-3 array-like.

## Heading Functions
- `rpymag2h(rpy, mag, declination=0.0) -> float`
- `qmag2h(q, mag, declination=0.0) -> float`

Parameters:
- `rpy`: roll/pitch/yaw (`Vector3D`, radians).
- `q`: orientation quaternion.
- `mag`: magnetometer vector (`Vector3D` or length-3 array-like).
- `declination`: magnetic declination in radians.

## Gravity and Acceleration Projection
- `q2gravity(pose) -> Vector3D`
  - Gravity direction in sensor frame from quaternion.
- `sensorAcc(acc, q, g) -> Vector3D`
  - Sensor-frame linear acceleration (`acc - gravity`).
- `earthAcc(acc, q, g) -> Vector3D`
  - Earth-frame linear acceleration.
- `gravity(latitude, altitude) -> float`
  - Local gravity magnitude in `m/s^2`.

Parameters:
- `acc`: accelerometer sample (`Vector3D`).
- `q` / `pose`: orientation quaternion.
- `g`: gravity magnitude.
- `latitude`: degrees.
- `altitude`: meters.
