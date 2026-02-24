# Quaternion API

## Class
`Quaternion(w=0.0, x=0.0, y=0.0, z=0.0, v=None)`

Represents a quaternion in scalar-first form: `[w, x, y, z]`.

### Initialization Parameters
- `w`: `float | Quaternion | Vector3D | array-like`
  - If `v` is `None` and `w` is numeric, `w` is scalar part.
  - If `w` is `Quaternion`, copies values.
  - If `w` is `Vector3D`, creates pure quaternion `[0, x, y, z]`.
  - If `w` is length-4 array, interpreted as `[w, x, y, z]`.
  - If `w` is length-3 array, interpreted as `[0, x, y, z]`.
- `x`: `float`, default `0.0`.
- `y`: `float`, default `0.0`.
- `z`: `float`, default `0.0`.
- `v`: optional alternate input source (`Quaternion`, `Vector3D`, length-3/4 array-like).

## Common Operations
- `q1 + q2`, `q1 - q2`
- `q1 * q2` (Hamilton product)
- `q * Vector3D` (vector promoted to pure quaternion)
- `q * scalar`, `scalar * q`
- `q / scalar`

## Methods
- `normalize()`
  - In-place normalization to unit quaternion.

## Properties
- `q.v -> Vector3D`: vector part `(x, y, z)`.
- `q.q -> np.ndarray`: array `[w, x, y, z]`.
- `q.conjugate -> Quaternion`: `[w, -x, -y, -z]`.
- `q.inverse -> Quaternion`: `conjugate / ||q||^2`.
- `q.norm -> float`: quaternion magnitude.
- `q.r33 -> np.ndarray`: `3x3` rotation matrix.
- `q.isZero -> bool`: near-zero check using module epsilon.

## Notes
- Many hot paths are optionally accelerated by `pyIMU._qcore`.
- Rotation convention in this project is NED.
