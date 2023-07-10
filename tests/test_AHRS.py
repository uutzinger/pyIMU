###################################
# To run profiler on this script:
# python -m cProfile -o profile_AHRS.prof tests\test_AHRS.py
# snakeviz profile_AHRS.prof
###################################
import numpy as np
import random

from pyIMU.madgwick import Madgwick
from pyIMU.motion import Motion
from pyIMU.quaternion import Quaternion, Vector3D
from pyIMU.utilities import q2rpy, heading

FUZZY_ACCEL_ZERO        = 10.0
FUZZY_DELTA_ACCEL_ZERO  = 0.04
FUZZY_GYRO_ZERO         = 0.08
FUZZY_DELTA_GYRO_ZERO   = 0.003
DECLINATION             = 9.27

AHRS = Madgwick(frequency=150.0, gain=0.033)

acc_offset          = Vector3D(0.,0.,0.)
gyr_offset          = Vector3D(0.,0.,0.)
mag_offset          = Vector3D(0.,0.,0.)
acc_crosscorr       = np.array(([1.,0.,0.], [0.,1.,0.], [0.,0.,1.]))
gyr_crosscorr       = np.array(([1.,0.,0.], [0.,1.,0.], [0.,0.,1.]))
mag_crosscorr       = np.array(([1.,0.,0.], [0.,1.,0.], [0.,0.,1.]))

def calibrate(data:Vector3D, offset:Vector3D, crosscorr=None):
    data = data-offset
    if crosscorr is not None:
        data = data.rotate(crosscorr)
    return data

def detectMotion(acc: float, gyr: float, acc_avg: float, gyr_avg:float) -> bool:
    # ACCELEROMETER
    acc_test       = abs(acc)           > FUZZY_ACCEL_ZERO
    acc_delta_test = abs(acc_avg - acc) > FUZZY_DELTA_ACCEL_ZERO
    # GYROSCOPE
    gyr_test       = abs(gyr)           > FUZZY_GYRO_ZERO
    gyr_delta_test = abs(gyr_avg - gyr) > FUZZY_DELTA_GYRO_ZERO
    return (acc_test or acc_delta_test or gyr_test or gyr_delta_test)

acc = Vector3D(0,0,1)
mag = Vector3D(30,0,30)
gyr = Vector3D(0,0,0)
dt = 1/150.
gyr_average = Vector3D(0,0,0)
acc_average = Vector3D(0,0,1)
    
for i in range(1000):

    # synthetic data
    rnd = random.uniform(0,0.2)    
    acc = acc + Vector3D(rnd,rnd,rnd)
    rnd = random.uniform(0,1)    
    mag = gyr + Vector3D(rnd,rnd,rnd)
    rnd = random.uniform(0,0.2)    
    gyr = gyr + Vector3D(rnd,rnd,rnd)
        
    _acc = calibrate(data=acc, offset=acc_offset, crosscorr=acc_crosscorr)
    _mag = calibrate(data=mag, offset=mag_offset, crosscorr=mag_crosscorr)
    _gyr = calibrate(data=gyr, offset=gyr_offset, crosscorr=gyr_crosscorr)

    gyr_average = 0.99*gyr_average + 0.01*gyr
    acc_average = 0.99*acc_average + 0.01*acc
    moving = detectMotion(acc.norm, gyr.norm, acc_average.norm, gyr_average.norm)

    if not moving:
        gyr_offset = 0.99*gyr_offset + 0.01*gyr
        gyr_offset_updated = True

    q = AHRS.update(acc=acc,gyr=gyr,mag=mag,dt=dt)
    h = heading(q=q, mag=mag, declination=DECLINATION)
    rpy=q2rpy(q=q)
