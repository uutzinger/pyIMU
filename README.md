# pyIMU

Python implementation of **Quaternion** and **Vector** math for Attitude and Heading Reference System (AHRS) as well as **motion** (acceleration, speed, position) estimation based on a Inertial Measurement Unit (IMU) (accelerometer, gyroscope and optional magnetometer).

The geometry conventions used in this implementation are from a pilots point of view:
- x points forward (North), positive **roll** turns plane clockwise
- y points right (East), positive **pitch** points nose up
- z points down (Down), positive **yaw** turns nose right

The AHRS and motion class have not yet been tested with real hardware.

## Install
Download the library and ```pip3 install -e .``` or omit the -e switch to install into python's site packages.

## pyIMU.madgwick
Contains pose sensor fusion based on Sebastian Madgwick [Dissertation](https://x-io.co.uk/downloads/madgwick_internal_report.pdf)

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
```

## pyIMU.motion

### Motion Class
Motion estimation based on IMU data.

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
- invSqrt (Do not use. Check also exercise in tests folder where fast inverse sqrt is implemented in C and pybind11. It's not fast on common desktop CPUs)

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

Calibration
- ellipsoid_fit: fits ellipsoid to [x,y,z] data, needed for IMU calibration
