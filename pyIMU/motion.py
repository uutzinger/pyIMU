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
        
        self.residuals_bias         = Vector3D(0,0,0) # on  the device frame coordinates
        
        self.worldResiduals         = Vector3D(0,0,0)
        self.worldResiduals_previous= Vector3D(0,0,0)

        self.worldVelocity          = Vector3D(0,0,0)
        self.worldVelocity_drift    = Vector3D(0,0,0) # in world corrdinate system
        self.worldVelocity_previous = Vector3D(0,0,0)
        
        self.worldPosition          = Vector3D(0,0,0)
        self.worldPosition_previous = Vector3D(0,0,0)

        self.driftLearningAlpha     = 0.2 # Poorman's low pass filter
        
        self.heading_X_avg          = 0.
        self.heading_Y_avg          = 0.
        self.heading                = 0.

        self.motion                 = False
        self.motion_previous        = False
        self.motionStart_time       = time.perf_counter()
          
        self.timestamp_previous = time.perf_counter()   # provided by caller
        self.dtmotion           = 0.0                   # no motion has occurred yet
    
        self.gravity = gravity(latitude=self.latitude, altitude=self.altitude)    # Gravity on Earth's (ellipsoid) Surface
        
    def reset(self):
        self.residuals_bias         = Vector3D(0,0,0)
        self.worldVelocity          = Vector3D(0,0,0)
        self.worldPosition          = Vector3D(0,0,0)
        self.worldVelocity_previous = Vector3D(0,0,0)
        self.worldVelocity_drift    = Vector3D(0,0,0)
        self.motion                 = False
    
    def resetPosition(self):
        self.worldPosition          = Vector3D(0,0,0)
        self.worldPosition_previous = Vector3D(0,0,0)

    def updateAverageHeading(self, heading) -> float:
        ## this needs two component because of 0 - 360 jump at North 
        self.m_heading_X.update(math.cos(heading))
        self.m_heading_y.update(math.sin(heading))
        self.heading = math.atan2(self.m_heading_Y.avg,self.m_heading_X.avg)
        if (self.heading < 0) : self.heading += TWOPI
        return self.heading
	
    def update(self, q:Quaternion, acc:Vector3D, motion: bool, timestamp: float):
        # Input:
        #  Quaternion
        #  Timestamp
        # Calculates:
        #  Acceleration in world coordinate system
        #  Velocity in world coordinate system
        #  Position in world coordinate system

        # Integration Time Step
        dt = timestamp - self.timestamp_previous
        self.timestamp_previous = timestamp
    
        # Acceleration residuals on the sensor
        self.residuals      = sensorAcc(acc=acc, q=q, g=self.gravity)
        self.residuals      = self.residuals - self.residuals_bias
        
        # Acceleration residuals in world coordinate system
        self.worldResiduals = q * self.residuals * q.conjugate
    
        # Motion Status, ongoing, no motion, ended?
        self.motion = motion
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
        if self.motion: 
            # Integrate acceleration and add to velocity (uses trapezoidal integration technique
            self.worldVelocity = self.worldVelocity_previous + ((self.worldResiduals + self.worldResiduals_previous)*0.5 * dt)

            # Update Velocity
            self.worldVelocity = self.worldVelocity - self.worldVelocity_drift * dt

            # Integrate velocity and add to position
            self.worldPosition = self.worldPosition_previous + (self.worldVelocity + self.worldVelocity_previous) * 0.5 * dt

            # keep history of previous values
            self.worldResiduals_previous = self.worldResiduals
            self.worldVelocity_previous  = self.worldVelocity
            self.worldPosition_previous  = self.worldPosition

        else: # no Motion
            # Estimate Velocity Bias when not moving
            # When motion ends, velocity should be zero
            if ((motion_ended == True) and (self.dtmotion > 0.5)): # update velocity bias if we had at least half of second motion
                self.worldVelocity_drift = ( (self.worldVelocity_drift * (1.0 - self.driftLearningAlpha)) + ((self.worldVelocity / self.dtmotion) * self.driftLearningAlpha ) )

            # Reset Velocity
            self.worldVelocity = Vector3D(x=0.,y=0.,z=0.)    # minimize error propagation

            #  Update acceleration bias
            self.residuals_bias = ( (self.residuals_bias * (1.0 - self.driftLearningAlpha)) + (self.residuals * self.driftLearningAlpha ) )
