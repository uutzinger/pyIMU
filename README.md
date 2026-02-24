# pyIMU

Python implementation of **Quaternion** and **Vector** math for Attitude and Heading Reference System (AHRS) as well as **motion** (acceleration, speed, position) estimation based on a Inertial Measurement Unit (IMU) consisting of an accelerometer, gyroscope and optional magnetometer.

The geometry conventions used in this implementation are from a pilots point of view:
- x points forward (North), positive **roll** turns plane clockwise
- y points right (East), positive **pitch** points nose up
- z points down (Down), positive **yaw** turns nose right

The motion class has not yet been tested with hardware. 

## Install
Download the library and ```pip3 install -e .``` or omit the -e switch to install into python's site packages.

### Optional Cython acceleration
`pyIMU` can build optional Cython extensions (`pyIMU._qcore`, `pyIMU._vcore`, `pyIMU._mcore`, `pyIMU._motion_core`) for quaternion/vector/Madgwick/motion hot paths.

- Build in place: `python3 setup.py build_ext --inplace`
- Editable install with build: `pip3 install -e .`

If compilation is not available, `pyIMU` automatically falls back to pure Python behavior.

### Release Helper Script
Use `scripts/release.sh` to build artifacts and optionally install/upload/commit/tag.

Examples:
- Build only: `scripts/release.sh --clean`
- Build + install wheel: `scripts/release.sh --clean --install`
- Build + commit + tag: `scripts/release.sh --clean --version 1.0.1 --commit --tag`
- Build + upload TestPyPI: `scripts/release.sh --clean --upload-testpypi`

## pyIMU.madgwick
Contains pose sensor fusion based on Sebastian Madgwick [dissertation work](https://x-io.co.uk/downloads/madgwick_internal_report.pdf) and work by [Mario Garcia](https://pypi.org/project/AHRS/) as well as work by Richardson Tech (RTIMU).

There is newer [implementation](https://pypi.org/project/imufusion/) based on an alternative approach in the Madgwick [thesis](https://ethos.bl.uk/OrderDetails.do?uin=uk.bl.ethos.681552) which is implemented in C with a Python API.

Original Code can be found here: https://x-io.co.uk/open-source-imu-and-ahrs-algorithms/

## pyIMU.fusion
Contains a revised AHRS `Fusion` class inspired by Chapter 7 of Madgwick's PhD thesis:
- complementary filter using cross-product error terms
- gain ramp initialisation
- gyroscope bias compensation with stationary gating
- magnetic distortion rejection
- linear acceleration rejection
- zero-g and global acceleration outputs
- NED orientation convention (x: North, y: East, z: Down)

### Magdwick Class
Incremental sensor fusion to compute pose quaternion from accelerometer, gyroscope and optional magnetometer.

Example:

```
from pyIMU.madgwick import Madgwick
madgwick = Madgwick(frequency=150.0, gain=0.033)
madgwick = Madgwick(dt=1/150.0, gain_imu=0.033)
madgwick = Madgwick()
type(madgwick.q)

# provide time increment dt based on time expired between each sensor reading
madgwick.update(gyr=gyro_data, acc=acc_data, dt=0.01)
madgwick.update(gyr=gyro_data, acc=acc_data, mag=mag_data, dt=0.01)
# access the quaternion
madgwick.q
# or take the return value of the update function
```

## pyIMU.motion

### Motion Class
Motion estimation based on IMU data. The implementation is inspired by [Gate tracking](https://github.com/xioTechnologies/Gait-Tracking/)

Example:
```
from pyIMU.motion import motion
estimator = motion(declination=9.27, latitude=32.253460, altitude=730, magfield=47392.3)

...
timestamp = time.time()
estimator.update(acc=acc, gyr=gyr, mag=mag, timestamp=timestamp)
```
## pyIMU.quaternion
Contains Quaternion and Vector3d.

### Quaternion Class
There are many implementations for quaternion math in python. This one is simple.

```q = [w,x,y,z]```

It supports operations with **quaternion**, **ndarray**, **scalar**, **Vector3D**

The operations supported are:
- add 
- subtract
- multiply
- equal 
- normalize 
- conjugate
- inverse 
- isZero
- extract vector or ndarray from quaternion
- convert to 3x3 rotation matrix

### Vector3D Class
Vector class to support quaternions, rotations and manipulating sensor readings.

The class supports
- add 
- subtract 
- multiply 
- potentiation 
- equal 
- less than 
- min 
- max 
- abs 
- normalize 
- sum 
- dot product 
- cross product 
- rotate with 3x3
- conversion vector to quaternion, ndarray

## pyIMU.utilities

### General Utility
- clip
- clamp
- invSqrt (Do not use)

### RunningAverage
Running average filter providing average and variance.

### Vector and Quaternion Support Routines
Conversions:
- vector_angle2q: create quaternion based on rotation around vector
- q2rpy: Quaternion to Roll Pitch Yaw
- rpy2q: Roll Pitch Yaw to Quaternion

Estimated quaternion based on gravity and magnetometer readings and assumption sensor is not moving.
- accel2rpy convert accelerometer reading to roll pitch yaw
- accel2q convert accelerometer reading to quaternion
- accelmag2q: accelerometer and magnetometer reading to quaternion

Acceleration:
- q2gravity: creates gravity vector on sensor frame based on quaternion
- sensorAcc: computes acceleration residuals (gravity subtracted acceleration) on sensor frame
- earthAcc: compute residuals on earth frame

Gravity
- gravity: computes gravity constant based on latitude and altitude above sea level

Heading
- heading: tilt compensated heading from quaternion and magnetometer (not tested)

## Calibration
See freeIMUCal repo

### Dependencies

