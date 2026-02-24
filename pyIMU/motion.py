###########################################################
# Motion calculation from IMU data
# After sensor fusion, the acceleration residuals can be
#  calculated in the world coordinate system
# Integrating acceleration residuals gives velocity and
# integrating velocity gives position
# Velocity and position are sensitive to drift and usually
#  not very accurate
#
# Urs Utzinger, Spring 2023
# GPT 5.2, 2026 speed optimizations
###########################################################
# https://www.researchgate.net/publication/258817923_FreeIMU_An_Open_Hardware_Framework_for_Orientation_and_Motion_Sensing
# https://web.archive.org/web/20210308200933/https://launchpad.net/freeimu/
###########################################################

from pyIMU.quaternion import Quaternion, Vector3D, TWOPI
from pyIMU.utilities import gravity, RunningAverage
import math, time

try:
    from pyIMU import _motion_core
    _HAS_MOTION_CORE = True
except Exception:
    _motion_core = None
    _HAS_MOTION_CORE = False

MINMOTIONTIME               = 0.5 # seconds, need to have had at least half of second motion to update velocity bias
HEADING_AVG_HISTORY         =  5

class Motion:
    '''
    IMU Motion Estimation
    '''

    def __init__(self, **kwargs):
        self.convention = str(kwargs.get('convention', 'NED')).upper()
        if self.convention != 'NED':
            raise ValueError("Motion currently supports only NED convention.")

        # Default values are for Tucson, Arizona, USA
        # This will is needed to compute magnitude of gravity
        #   which is subtracted from measured acceleration
        self.latitude         = kwargs.get('latitude', 32.253460)  # decimal degrees
        self.altitude         = kwargs.get('altitude', 730)        # meter

        self.m_heading_X            = RunningAverage(HEADING_AVG_HISTORY)
        self.m_heading_Y            = RunningAverage(HEADING_AVG_HISTORY)

        self.residuals_bias         = Vector3D(0,0,0) # on  the device frame coordinates
        self.residuals              = Vector3D(0,0,0)

        self.worldResiduals         = Vector3D(0,0,0)
        self.worldResiduals_previous= Vector3D(0,0,0)

        self.worldVelocity          = Vector3D(0,0,0)
        self.worldVelocity_drift    = Vector3D(0,0,0) # in world corrdinate system
        self.worldVelocity_previous = Vector3D(0,0,0)

        self.worldPosition          = Vector3D(0,0,0)
        self.worldPosition_previous = Vector3D(0,0,0)

        self.driftLearningAlpha     = 0.01                  # Poorman's low pass filter

        self.heading_X_avg          = 0.
        self.heading_Y_avg          = 0.
        self.m_heading              = 0.                    # average heading in radians

        self.motion                 = False                 # is device moving?
        self.motion_previous        = False
        self.motionStart_time       = time.perf_counter()

        self.timestamp_previous     = -1.
        self.dtmotion               = 0.0                   # no motion has occurred yet, length of motion period in seconds
        self.dt                     = 0.0

        self.localGravity = gravity(latitude=self.latitude, altitude=self.altitude)    # Gravity on Earth's (ellipsoid) Surface

    def reset(self):
        self.residuals_bias         = Vector3D(0,0,0)
        self.residuals              = Vector3D(0,0,0)
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
        self.m_heading_Y.update(math.sin(heading))
        self.m_heading = math.atan2(self.m_heading_Y.avg,self.m_heading_X.avg)
        if (self.m_heading < 0) : self.m_heading += TWOPI
        return self.m_heading

    def update(self, q:Quaternion, acc:Vector3D, moving: bool, timestamp: float):
        # Compatibility wrapper. Motion state is always updated in-place.
        self.update_inplace(q, acc, moving, timestamp)

    def update_inplace(self, q:Quaternion, acc:Vector3D, moving: bool, timestamp: float):
        # Input:
        #  Quaternion
        #  Timestamp
        # Calculates:
        #  Acceleration in world coordinate system
        #  Velocity in world coordinate system
        #  Position in world coordinate system
        # If there is no motion
        #  Acceleration bias is updated
        #  Velocity bias is updated

        # Integration Time Step
        if self.timestamp_previous < 0.0:
            dt = 0.0
        else:
            dt = timestamp - self.timestamp_previous
            if dt > 1.0:
                # we had reconnection, avoid updating with large dt
                dt = 0.0
        self.timestamp_previous = timestamp

        # Motion Status, ongoing, no motion, ended?
        motion_ended = False
        if (self.motion_previous == False):
            if (moving == True):
                # Motion Started
                self.motionStart_time = timestamp
        else:
            if (moving == False):
                # Motion Ended
                motion_ended = True
        # Keep track of previous status
        self.motion_previous = moving

        r33 = q.r33
        if _HAS_MOTION_CORE:
            (
                residual_x, residual_y, residual_z,
                wr_x, wr_y, wr_z,
                wr_prev_x, wr_prev_y, wr_prev_z,
                wv_x, wv_y, wv_z,
                wv_prev_x, wv_prev_y, wv_prev_z,
                wp_prev_x, wp_prev_y, wp_prev_z,
                wv_drift_x, wv_drift_y, wv_drift_z,
                bias_x, bias_y, bias_z,
                dtmotion
            ) = _motion_core.motion_step(
                acc.x, acc.y, acc.z,
                q.w, q.x, q.y, q.z,
                r33[0, 0], r33[0, 1], r33[0, 2],
                r33[1, 0], r33[1, 1], r33[1, 2],
                r33[2, 0], r33[2, 1], r33[2, 2],
                self.localGravity,
                self.residuals_bias.x, self.residuals_bias.y, self.residuals_bias.z,
                self.worldResiduals_previous.x, self.worldResiduals_previous.y, self.worldResiduals_previous.z,
                self.worldVelocity.x, self.worldVelocity.y, self.worldVelocity.z,
                self.worldVelocity_previous.x, self.worldVelocity_previous.y, self.worldVelocity_previous.z,
                self.worldPosition_previous.x, self.worldPosition_previous.y, self.worldPosition_previous.z,
                self.worldVelocity_drift.x, self.worldVelocity_drift.y, self.worldVelocity_drift.z,
                self.driftLearningAlpha,
                dt,
                self.dtmotion,
                moving,
                motion_ended,
                MINMOTIONTIME
            )
        else:
            (
                residual_x, residual_y, residual_z,
                wr_x, wr_y, wr_z,
                wr_prev_x, wr_prev_y, wr_prev_z,
                wv_x, wv_y, wv_z,
                wv_prev_x, wv_prev_y, wv_prev_z,
                wp_prev_x, wp_prev_y, wp_prev_z,
                wv_drift_x, wv_drift_y, wv_drift_z,
                bias_x, bias_y, bias_z,
                dtmotion
            ) = _motion_step_python(
                acc.x, acc.y, acc.z,
                q.w, q.x, q.y, q.z,
                r33,
                self.localGravity,
                self.residuals_bias.x, self.residuals_bias.y, self.residuals_bias.z,
                self.worldResiduals_previous.x, self.worldResiduals_previous.y, self.worldResiduals_previous.z,
                self.worldVelocity.x, self.worldVelocity.y, self.worldVelocity.z,
                self.worldVelocity_previous.x, self.worldVelocity_previous.y, self.worldVelocity_previous.z,
                self.worldPosition_previous.x, self.worldPosition_previous.y, self.worldPosition_previous.z,
                self.worldVelocity_drift.x, self.worldVelocity_drift.y, self.worldVelocity_drift.z,
                self.driftLearningAlpha,
                dt,
                self.dtmotion,
                moving,
                motion_ended,
                MINMOTIONTIME
            )

        self.residuals.x = residual_x
        self.residuals.y = residual_y
        self.residuals.z = residual_z

        self.worldResiduals.x = wr_x
        self.worldResiduals.y = wr_y
        self.worldResiduals.z = wr_z

        self.worldResiduals_previous.x = wr_prev_x
        self.worldResiduals_previous.y = wr_prev_y
        self.worldResiduals_previous.z = wr_prev_z

        self.worldVelocity.x = wv_x
        self.worldVelocity.y = wv_y
        self.worldVelocity.z = wv_z

        self.worldVelocity_previous.x = wv_prev_x
        self.worldVelocity_previous.y = wv_prev_y
        self.worldVelocity_previous.z = wv_prev_z

        self.worldPosition.x = wp_prev_x
        self.worldPosition.y = wp_prev_y
        self.worldPosition.z = wp_prev_z

        self.worldPosition_previous.x = wp_prev_x
        self.worldPosition_previous.y = wp_prev_y
        self.worldPosition_previous.z = wp_prev_z

        self.worldVelocity_drift.x = wv_drift_x
        self.worldVelocity_drift.y = wv_drift_y
        self.worldVelocity_drift.z = wv_drift_z

        self.residuals_bias.x = bias_x
        self.residuals_bias.y = bias_y
        self.residuals_bias.z = bias_z

        self.dtmotion = dtmotion

        self.dt = dt


def _motion_step_python(
    accx: float, accy: float, accz: float,
    qw: float, qx: float, qy: float, qz: float,
    r33,
    local_gravity: float,
    bias_x: float, bias_y: float, bias_z: float,
    wr_prev_x: float, wr_prev_y: float, wr_prev_z: float,
    wv_x: float, wv_y: float, wv_z: float,
    wv_prev_x: float, wv_prev_y: float, wv_prev_z: float,
    wp_prev_x: float, wp_prev_y: float, wp_prev_z: float,
    wv_drift_x: float, wv_drift_y: float, wv_drift_z: float,
    drift_alpha: float,
    dt: float,
    dtmotion: float,
    moving: bool,
    motion_ended: bool,
    min_motion_time: float,
):
    # q2gravity + sensor residual on sensor frame
    gx = 2.0 * (qx * qz - qw * qy)
    gy = 2.0 * (qy * qz + qw * qx)
    gz = 1.0 - 2.0 * (qx * qx + qy * qy)

    residual_x = accx - (local_gravity * gx) - bias_x
    residual_y = accy - (local_gravity * gy) - bias_y
    residual_z = accz - (local_gravity * gz) - bias_z

    # residuals in world frame: r33 * residual
    wr_x = r33[0, 0] * residual_x + r33[0, 1] * residual_y + r33[0, 2] * residual_z
    wr_y = r33[1, 0] * residual_x + r33[1, 1] * residual_y + r33[1, 2] * residual_z
    wr_z = r33[2, 0] * residual_x + r33[2, 1] * residual_y + r33[2, 2] * residual_z

    if moving:
        scale = 0.5 * dt
        # trapezoidal integrate acceleration -> velocity
        wv_x = wv_prev_x + (wr_x + wr_prev_x) * scale
        wv_y = wv_prev_y + (wr_y + wr_prev_y) * scale
        wv_z = wv_prev_z + (wr_z + wr_prev_z) * scale
        dtmotion += dt

        # subtract learned drift
        wv_x -= wv_drift_x * dt
        wv_y -= wv_drift_y * dt
        wv_z -= wv_drift_z * dt

        # trapezoidal integrate velocity -> position
        wp_x = wp_prev_x + (wv_x + wv_prev_x) * scale
        wp_y = wp_prev_y + (wv_y + wv_prev_y) * scale
        wp_z = wp_prev_z + (wv_z + wv_prev_z) * scale

        wr_prev_x = wr_x
        wr_prev_y = wr_y
        wr_prev_z = wr_z

        wv_prev_x = wv_x
        wv_prev_y = wv_y
        wv_prev_z = wv_z

        wp_prev_x = wp_x
        wp_prev_y = wp_y
        wp_prev_z = wp_z

    else: # no Motion
        # Estimate Velocity Bias when not moving
        # When motion ends, velocity should be zero
        if motion_ended and (dtmotion > min_motion_time):
            alpha = drift_alpha
            one_minus_alpha = 1.0 - alpha
            scale = alpha / dtmotion
            wv_drift_x = wv_drift_x * one_minus_alpha + wv_x * scale
            wv_drift_y = wv_drift_y * one_minus_alpha + wv_y * scale
            wv_drift_z = wv_drift_z * one_minus_alpha + wv_z * scale
            dtmotion = 0.0

        # Reset Residuals / Velocity
        wr_prev_x = 0.0
        wr_prev_y = 0.0
        wr_prev_z = 0.0
        wv_x = 0.0
        wv_y = 0.0
        wv_z = 0.0
        wv_prev_x = 0.0
        wv_prev_y = 0.0
        wv_prev_z = 0.0

        # Update acceleration bias, when not moving residuals should be zero
        one_minus_alpha = 1.0 - drift_alpha
        bias_x = (bias_x * one_minus_alpha) + (residual_x * drift_alpha)
        bias_y = (bias_y * one_minus_alpha) + (residual_y * drift_alpha)
        bias_z = (bias_z * one_minus_alpha) + (residual_z * drift_alpha)

    return (
        residual_x, residual_y, residual_z,
        wr_x, wr_y, wr_z,
        wr_prev_x, wr_prev_y, wr_prev_z,
        wv_x, wv_y, wv_z,
        wv_prev_x, wv_prev_y, wv_prev_z,
        wp_prev_x, wp_prev_y, wp_prev_z,
        wv_drift_x, wv_drift_y, wv_drift_z,
        bias_x, bias_y, bias_z,
        dtmotion,
    )
