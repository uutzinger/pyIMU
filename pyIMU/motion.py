from pyIMU.quaternion import Quaternion, Vector3D, TWOPI
from pyIMU.utilities import gravity, RunningAverage, sensorAcc
from pyIMU.madgwick import Madgwick
import math
import time

# https://www.researchgate.net/publication/258817923_FreeIMU_An_Open_Hardware_Framework_for_Orientation_and_Motion_Sensing
# https://web.archive.org/web/20210308200933/https://launchpad.net/freeimu/

class Motion:
    '''
    IMU Motion Estimation
    '''
    def __init__(self, **kwargs):

        # Default values are for Tucson, Arizona, USA
        self.declination      = kwargs.get('declination', 9.27)    # decimal degrees 
        self.latitude         = kwargs.get('latitude', 32.253460)  # decimal degrees
        self.altitude         = kwargs.get('altitude', 730)        # meter
        self.magfield         = kwargs.get('magfield', 47392.3)    # nT

        ACCEL_AVG_HISTORY           =  5                      # size of moving average filter
        GYR_AVG_HISTORY             =  5                      # size of moving average filter
        HEADING_AVG_HISTORY         =  5 

        self.m_acc                  = RunningAverage(ACCEL_AVG_HISTORY)
        self.m_gyr                  = RunningAverage(GYR_AVG_HISTORY)
        self.m_heading_X            = RunningAverage(HEADING_AVG_HISTORY)
        self.m_heading_Y            = RunningAverage(HEADING_AVG_HISTORY)
        
        self.residuals_bias         = Vector3D(0,0,0)
        
        self.worldResiduals         = Vector3D(0,0,0)
        self.worldResiduals_previous= Vector3D(0,0,0)

        self.worldVelocity          = Vector3D(0,0,0)
        self.worldVelocity_drift    = Vector3D(0,0,0)
        self.worldVelocity_previous = Vector3D(0,0,0)
        
        self.worldPosition          = Vector3D(0,0,0)

        self.velocityDriftLearningAlpha = 0.2 # Poorman's low pass filter
        
        self.heading_X_avg          = 0.
        self.heading_Y_avg          = 0.
        self.heading                = 0.

        self.motion                 = False
        self.motion_previous        = False
        self.motionStart_time       = time.perf_counter()

        # These values need to be tuned for each IMU
        self.FUZZY_ACCEL_ZERO       = 0.1  # if more than 0.1m/s^2 acceleration, we are moving
        self.FUZZY_ACCEL_DELTA_ZERO = 0.0  # 2-4 times the sqrt of acceleration variance
        self.FUZZY_GYRO_ZERO        = 0.07 # absolute value of real gyration
        self.FUZZY_DELTA_GYRO_ZERO  = 0.01 # 2-4 times the sqrt of gyration variance
          
        self.timestamp_previous = time.perf_counter()   # provided by caller
        self.dtmotion           = 0.0                   # no motion has occurred yet
    
        self.gravity = gravity(latitude=self.latitude, altitude=self.altitude)    # Gravity on Earth's (ellipsoid) Surface
    
        self.madgwick = Madgwick()                      # AHRS filter
    
    def reset(self):
        self.residuals_bias         = Vector3D(0,0,0)
        self.worldVelocity          = Vector3D(0,0,0)
        self.worldPosition          = Vector3D(0,0,0)
        self.worldVelocity_previous = Vector3D(0,0,0)
        self.worldVelocity_drift    = Vector3D(0,0,0)
        self.motion                 = False
    
    def resetPosition(self):
        self.worldPosition = Vector3D(0,0,0)

    def updateAverageHeading(self, heading) -> float:
        ## this needs two component because of 0 - 360 jump at North 
        self.m_heading_X.update(math.cos(heading))
        self.m_heading_y.update(math.sin(heading))
        self.heading = math.atan2(self.m_heading_Y.avg,self.m_heading_X.avg)
        if (self.heading < 0) : self.heading += TWOPI
        return self.heading

    def detectMotion(self, acc: float, gyr: float) -> bool:
        # Three Stage Motion Detection
        # Original Code is from FreeIMU Processing example
        # Some modifications and tuning
        #
        # 0. Acceleration Activity
        # 1. Change in Acceleration
        # 2. Gyration Activity
        # 2. Change in Gyration
        
        # ACCELEROMETER
        # Absolute value
        acc_test       = math.abs(acc)                  > self.FUZZY_ACCEL_ZERO
        # Sudden changes
        acc_delta_test = math.abs(self.m_acc.avg - acc) > self.FUZZY_DELTA_ACCEL_ZERO

        # GYROSCOPE
        # Absolute value
        gyr_test       = math.abs(gyr)                  > self.FUZZY_GYRO_ZERO
        # Sudden changes
        gyr_delta_test = math.abs(self.m_gyr.avg - gyr) > self.FUZZY_DELTA_GYRO_ZERO
             
        # Combine acceleration test, acceleration deviation test and gyro test
        return (acc_test or acc_delta_test or gyr_test or gyr_delta_test)
	
    def update(self, acc:Vector3D, gyr:Vector3D, mag: None, timestamp: float):
        # Input:
        #  Acceleration
        #  Gyration
        #  Magnetometer
        #  Timestamp
        # Calculates:
        #  Acceleration in world coordinate system
        #  Velocity in world coordinate system
        #  Position in world coordinate system

        # Update moving average filters
        self.m_acc.update(acc)
        self.m_gyr.update(gyr)

        # Tune the motion detector   
        print("Value:    Accel: {}, Gyration: {}".format(acc, gyr))
        print("Average:  Accel: {}, Gyration: {}".format(self.m_acc.avg, self.m_gyr.avg))
        print("Variance: Accel: {}, Gyration: {}".format(self.m_acc.var, self.m_gyr.var))

        # Integration Time Step
        dt = timestamp - self.timestamp_previous
        self.timestamp_previous = timestamp

        # Pose Estimation
        q = self.madgwick.update(acc=acc, gyr=gyr, mag=mag, dt=dt)
    
        # Acceleration residuals
        self.residuals      = sensorAcc(acc=acc, q=q, g=self.gravity)
        self.residuals     -= self.residuals_bias
        # Acceleration residuals in world coordinate system
        self.worldResiduals = self.residuals.rot(q.r33.T)
    
        # Motion Status, ongoing, no motion, ended?
        self.motion = self.detectMotion(self, acc=acc.norm, gyr=gyr.norm)   
        motion_ended = False 
        if (self.motion_previous == False):
            if (self.motion == True):
                # Motion Started
                self.motionStart_time = timestamp
        else:
            if (self.motion == False):
                # Motion Ended
                self.dtmotion = timestamp - self.motionStart_time
                motion_ended = True       
        # Keep track of previous status
        self.motion_previous = self.motion

        # Update Velocity and Position when moving
        # Estimate Drift when not moving
        if self.motion: 
            # Integrate acceleration and add to velocity (uses trapezoidal integration technique
            self.worldVelocity = self.worldVelocity_previous + ((self.worldResiduals + self.worldResiduals_previous)*0.5 * dt)

            # Update Velocity
            self.worldVelocity -= self.worldVelocity_drift * dt

            # Integrate velocity and add to position
            self.worldPosition += ((self.worldVelocity + self.worldVelocity_previous)*0.5) * dt

            # keep history of previous values
            self.worldResiduals_previous = self.worldResiduals
            self.worldVelocity_previous  = self.worldVelocity

        else: # no Motion
            # Update Velocity Bias
            # When motion ends, velocity should be zero
            if ((motion_ended == True) and (self.dtmotion > 0.5)): # update velocity bias if we had at least half of second motion
                self.worldVelocity_drift = ( (self.worldVelocity_drift * (1.0 - self.velocityDriftLearningAlpha)) + ((self.worldVelocity / self.dtmotion) * self.velocityDriftLearningAlpha ) )

            # Reset Velocity
            self.worldVelocity = Vector3D(x=0.,y=0.,z=0.)    # minimize error propagation

            #  Update acceleration bias
            self.residuals_bias = ( (self.residuals_bias * (1.0 - self.velocityDriftLearningAlpha)) + (self.residuals * self.velocityDriftLearningAlpha ) )

        print("World:  Accel    {}".format(self.worldResiduals))
        print("World:  Velocity {}".format(self.worldVelocity))
        print("World:  Position {}".format(self.worldPosition))
        print("World:  Velocity Drift{}".format(self.worldVelocity_drift))
        print("Sensor: Acc Bias      {}".format(self.residuals_bias))
        print("Time: {}".format(timestamp))
        print("Motion Time: {}, Ended: {}", self.dtmotion, motion_ended); 
