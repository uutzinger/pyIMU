# pyIMU

Python implementation of quaternion/vector math for Attitude and Heading Reference Systems (AHRS), plus motion estimation from IMU data (accelerometer, gyroscope, optional magnetometer).

AHRS based on Madgwick filters.

## Authors
Urs Utzinger 2023-2026
Gpt 5.3 efficiency, cythonization and codereview, 2026

## Coordinate Convention
`pyIMU` uses a pilot-style NED frame:
- `x`: forward (North)
- `y`: right (East)
- `z`: down (Down)

Positive rotations follow this convention:
- roll: right wing down
- pitch: nose up
- yaw: nose right

## Units
Default units in filters:
- gyroscope: `rad/s` (or `deg/s` if `gyr_in_dps=True`)
- accelerometer: `g` (or `m/s^2` if `acc_in_g=False`)
- magnetometer: any consistent unit (direction is normalized internally)

## Installation
Install in editable mode:

```bash
pip3 install -e .
```

Or standard install:

```bash
pip3 install .
```

## Optional Cython Acceleration

Optional compiled kernel modules for hot functions:
- `pyIMU._qcore` for quaternion
- `pyIMU._vcore` for vector3D
- `pyIMU._mcore` for madgwick
- `pyIMU._motion_core` for motion
- `pyIMU._fcore` for imporoved madgwick

Build in place:

```bash
python3 setup.py build_ext --inplace
```

If C extensions are unavailable, `pyIMU` falls back to pure Python.

## Quick Start

### Madgwick

```python
from pyIMU.madgwick import Madgwick
from pyIMU.quaternion import Vector3D

f = Madgwick(dt=1.0/150.0, gain=0.033, convention="NED")
q = f.update(
    gyr=Vector3D(0.01, 0.02, 0.00),
    acc=Vector3D(0.0, 0.0, 1.0),
    mag=Vector3D(0.3, 0.0, 0.4),
    dt=1.0 / 150.0,
)
```


### Fusion (improved Madgwick)

```python
from pyIMU.fusion import Fusion
from pyIMU.quaternion import Vector3D

f = Fusion(k_init=10.0, k_normal=0.5, convention="NED")
q = f.update(
    gyr=Vector3D(0.01, 0.02, 0.00),
    acc=Vector3D(0.0, 0.0, 1.0),
    mag=Vector3D(0.3, 0.0, 0.4),
    dt=0.01,
)
```

### Motion

```python
import time
from pyIMU.motion import Motion
from pyIMU.quaternion import Quaternion, Vector3D

m = Motion(latitude=32.253460, altitude=730, convention="NED")

q = Quaternion(1.0, 0.0, 0.0, 0.0)
acc = Vector3D(0.0, 0.0, 1.0)
moving = False

m.update(q=q, acc=acc, moving=moving, timestamp=time.time())
```

### Calibration

```python
import numpy as np
from pyIMU.quaternion import Vector3D
from pyIMU.calibration import InertialCalibration, MagnetometerCalibration

acc_cal = InertialCalibration(
    offset=Vector3D(0.0, 0.0, 0.0),
    sensitivity=Vector3D(1.0, 1.0, 1.0),
    misalignment=np.eye(3),
)

mag_cal = MagnetometerCalibration(
    hard_iron=Vector3D(0.0, 0.0, 0.0),
    soft_iron=np.eye(3),
)

acc = acc_cal.apply(acc_raw)
mag = mag_cal.apply(mag_raw)

```

## Modules

### `pyIMU.quaternion`

- `Quaternion` math [primitives](quaternion.md)
- `Vector3D` math [primitives](vector3D.md)

### `pyIMU.madgwick`

Madgwick gradient-descent [AHRS implementation](madgwick.md).

References:
- [internal report](https://x-io.co.uk/downloads/madgwick_internal_report.pdf)
- [IEEE](https://doi.org/10.1109/ICORR.2011.5975346)
- [Thesis](https://x-io.co.uk/downloads/madgwick-phd-thesis.pdf)
- [x-io website](https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/)

### `pyIMU.fusion`

[Fusion-style AHRS](fusion.md) inspired by Chapter 7 of Madgwick's thesis and x-io Fusion behavior:

- gain ramp initialization
- gyroscope bias compensation
- acceleration rejection/recovery
- magnetic rejection/recovery
- angular-rate recovery

References:
- [Thesis](https://x-io.co.uk/downloads/madgwick-phd-thesis.pdf) chapter 7
- [x-io website](https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/)


### `pyIMU.motion`

[IMU motion](motion.md) integration (acceleration/velocity/position) with drift handling.

Note: drift is expected without external aiding.

### `pyIMU.calibration`

[Calibration helpers](calibration.md):

InertialCalibration: 
$$i_c=M_s(i_u-b)$$
`misalignment @ (diag(sensitivity) @ (raw - offset))`

MagnetometerCalibration: 
$$m_c=S(m_u-h)$$
`soft_iron @ (raw - hard_iron)`

one-shot helpers: 
`calibrate_inertial`, `calibrate_magnetic`

Defaults are identity calibration (`offset=0`, `scale=1`, identity matrices).

### `pyIMU.utilities`

[General helpers](utilities.md) and conversion utilities (`clip`, `clamp`, `q2rpy`, `rpy2q`, `accel2q` (when still), `accelmag2q` (when still), gravity and heading helpers, `RunningAverage`).

## Release Helper Script

Use `scripts/release.sh` to build and optionally install/upload/tag.

Examples:

- build only: `scripts/release.sh --clean`
- build + install wheel: `scripts/release.sh --clean --install`
- build + commit + tag: `scripts/release.sh --clean --version 1.0.1 --commit --tag`
- build + commit + tag + push: `scripts/release.sh --clean --version 1.0.1 --commit --tag --push`
- build + upload TestPyPI: `scripts/release.sh --clean --upload-testpypi`
