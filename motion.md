# Motion API

Module: `pyIMU.motion`

## Class
`Motion(**kwargs)`

Estimates gravity-compensated acceleration, velocity, and position in world frame from orientation + accelerometer input.

## Initialization Parameters
- `convention`: `str`, default `"NED"`
  - Coordinate convention. Only `NED` is currently supported.
- `latitude`: `float`, default `32.253460`
  - Latitude in decimal degrees.
  - Used to compute local gravity magnitude.
- `altitude`: `float`, default `730`
  - Altitude above sea level in meters.
  - Used to compute local gravity magnitude.

## Methods
- `update(q, acc, moving, timestamp)`
  - Compatibility wrapper that calls `update_inplace(...)`.
- `update_inplace(q, acc, moving, timestamp)`
  - Updates internal state in place.
- `updateAverageHeading(heading) -> float`
  - Running-average heading with wraparound handling.
- `reset()`
  - Resets motion-related state (bias, residuals, velocity, position drift terms).
- `resetPosition()`
  - Resets world position only.

## `update` / `update_inplace` Parameters
- `q`: `Quaternion`
  - Current orientation (NED convention).
- `acc`: `Vector3D`
  - Accelerometer sample (expected in `m/s^2`).
- `moving`: `bool`
  - Motion detector output (`True` when moving).
- `timestamp`: `float`
  - Monotonic time in seconds.

## Outputs / State Fields
After each update, these fields are updated:
- `residuals`: sensor-frame linear acceleration after gravity and bias removal.
- `residuals_bias`: learned sensor-frame acceleration bias.
- `worldResiduals`: gravity-compensated acceleration in world frame.
- `worldVelocity`: integrated world-frame velocity.
- `worldPosition`: integrated world-frame position.
- `worldVelocity_drift`: learned drift term used for velocity correction.
- `dt`: last integration step.
- `dtmotion`: accumulated duration of current motion segment.

## Notes
- Motion integration is drift-prone without external aiding (e.g., GPS, vision, ZUPT logic).
- `moving` quality strongly affects bias/drift learning.
- Optional kernel `pyIMU._motion_core` is used automatically when available; pure Python fallback is used otherwise.
