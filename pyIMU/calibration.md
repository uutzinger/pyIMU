# Calibration API

Module: `pyIMU.calibration`

## InertialCalibration
Class: `InertialCalibration(offset=(0,0,0), sensitivity=(1,1,1), misalignment=I)`

Calibration model for accelerometer or gyroscope:
`calibrated = misalignment @ (diag(sensitivity) @ (raw - offset))`

Initialization Parameters
- `offset`: `Vector3D` or length-3 array-like, default `(0,0,0)`.
- `sensitivity`: `Vector3D` or length-3 array-like, default `(1,1,1)`.
- `misalignment`: `3x3` array-like, default identity.

Methods
- `apply(raw) -> Vector3D`
  - `raw`: `Vector3D` or length-3 array-like raw sample.

## MagnetometerCalibration
Class: `MagnetometerCalibration(hard_iron=(0,0,0), soft_iron=I)`

Calibration model:
`calibrated = soft_iron @ (raw - hard_iron)`

Initialization Parameters
- `hard_iron`: `Vector3D` or length-3 array-like, default `(0,0,0)`.
- `soft_iron`: `3x3` array-like, default identity.

Methods
- `apply(raw) -> Vector3D`
  - `raw`: `Vector3D` or length-3 array-like raw sample.

## One-Shot Helpers
- `calibrate_inertial(raw, offset=None, sensitivity=None, misalignment=None) -> Vector3D`
  - Defaults: zero offset, unit sensitivity, identity misalignment.
- `calibrate_magnetic(raw, hard_iron=None, soft_iron=None) -> Vector3D`
  - Defaults: zero hard-iron, identity soft-iron.

## Notes
- All defaults are identity calibration, so inputs pass through unchanged.
- These helpers return `Vector3D`.
