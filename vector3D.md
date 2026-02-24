# Vector3D API

## Class
`Vector3D(x=0.0, y=0.0, z=0.0)`

Represents a 3D vector.

### Initialization Parameters
- `x`: `float | Vector3D | array-like`
  - numeric: sets `x` value
  - `Vector3D`: copy constructor
  - length-3 array-like: interpreted as `[x, y, z]`
- `y`: `float`, default `0.0` (used when `x` is numeric)
- `z`: `float`, default `0.0` (used when `x` is numeric)

## Common Operations
- `v1 + v2`, `v1 - v2`
- `v * scalar`, `scalar * v`
- element-wise `v1 * v2`
- `v / scalar`
- `v ** scalar` or element-wise with another `Vector3D`

## Methods
- `normalize()`
  - In-place normalization to unit length when norm is non-zero.
- `dot(other)`
  - Dot product with `Vector3D` or length-3 `np.ndarray`.
- `cross(other)`
  - Cross product with `Vector3D`.
- `rotate(r33)`
  - Rotate using a `3x3` rotation matrix.
- `sum()`
  - Returns `x + y + z`.
- `min(other)`, `max(other)`
  - Componentwise min/max with scalar or `Vector3D`.

## Properties
- `v.q -> Quaternion`: pure quaternion `[0, x, y, z]`.
- `v.v -> np.ndarray`: array `[x, y, z]`.
- `v.norm -> float`: vector magnitude.
- `v.isZero -> bool`: near-zero check.

## Notes
- Vector kernels can be accelerated by optional `pyIMU._vcore`.
