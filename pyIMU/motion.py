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
from pyIMU.utilities import gravity, RunningAverage, q2gravity
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
            ) = _motion_core.motion_step_quat(
                acc.x, acc.y, acc.z,
                q.w, q.x, q.y, q.z,
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
        else:
            dtmotion = _motion_step_python(
                acc,
                q,
                self.localGravity,
                self.residuals,
                self.residuals_bias,
                self.worldResiduals,
                self.worldResiduals_previous,
                self.worldVelocity,
                self.worldVelocity_previous,
                self.worldPosition_previous,
                self.worldVelocity_drift,
                self.driftLearningAlpha,
                dt,
                self.dtmotion,
                moving,
                motion_ended,
                MINMOTIONTIME
            )
            # keep position mirror field behavior consistent with C-core path
            self.worldPosition = Vector3D(self.worldPosition_previous)

        self.dtmotion = dtmotion

        self.dt = dt


def _motion_step_python(
    acc: Vector3D,         # acceleration in sensor frame, in m/s^2
    q: Quaternion,         # orientation of sensor frame with respect to world frame
    local_gravity: float,  # local gravity magnitude, in m/s^2
    residual: Vector3D,    # residual acceleration in sensor frame, in m/s^2
    bias: Vector3D,        # acceleration bias, in m/s^2
    wr: Vector3D,          # residual acceleration in world frame, in m/s^2
    wr_prev: Vector3D,     # previous residuals in world frame, in m/s^2
    wv: Vector3D,          # current velocity in world frame, in m/s
    wv_prev: Vector3D,     # previous velocity in world frame, in m/s
    wp_prev: Vector3D,     # previous position in world frame, in m
    wv_drift: Vector3D,    # learned drift velocity in world frame, in m/s
    drift_alpha: float,    # learning rate for drift estimation
    dt: float,             # time step, in s
    dtmotion: float,       # accumulated motion time, in s
    moving: bool,          # flag indicating if device is moving
    motion_ended: bool,    # flag indicating if motion has ended
    min_motion_time: float # minimum motion time threshold, in s
):
    # q2gravity + sensor residual on sensor frame
    g = q2gravity(q)
    # residual = acc - local_gravity * g - bias
    residual.x = acc.x - local_gravity * g.x - bias.x
    residual.y = acc.y - local_gravity * g.y - bias.y
    residual.z = acc.z - local_gravity * g.z - bias.z

    # residuals in world frame: r33 * residual (inline, no NumPy matrix path)
    xx = q.x * q.x
    xy = q.x * q.y
    xz = q.x * q.z
    xw = q.x * q.w
    yy = q.y * q.y
    yz = q.y * q.z
    yw = q.y * q.w
    zz = q.z * q.z
    zw = q.z * q.w

    wr.x = (1.0 - 2.0 * (yy + zz)) * residual.x + (2.0 * (xy - zw)) * residual.y + (2.0 * (xz + yw)) * residual.z
    wr.y = (2.0 * (xy + zw)) * residual.x + (1.0 - 2.0 * (xx + zz)) * residual.y + (2.0 * (yz - xw)) * residual.z
    wr.z = (2.0 * (xz - yw)) * residual.x + (2.0 * (yz + xw)) * residual.y + (1.0 - 2.0 * (xx + yy)) * residual.z

    if moving:
        scale = 0.5 * dt
        # trapezoidal integrate acceleration -> velocity
        # wv = wv_prev + (wr + wr_prev) * scale
        wv.x = wv_prev.x + (wr.x + wr_prev.x) * scale
        wv.y = wv_prev.y + (wr.y + wr_prev.y) * scale
        wv.z = wv_prev.z + (wr.z + wr_prev.z) * scale
        dtmotion += dt

        # subtract learned drift
        # wv = wv - wv_drift * dt
        wv.x -= wv_drift.x * dt
        wv.y -= wv_drift.y * dt
        wv.z -= wv_drift.z * dt

        # trapezoidal integrate velocity -> position
        # wp = wp_prev + (wv + wv_prev) * scale
        wp_prev.x += (wv.x + wv_prev.x) * scale
        wp_prev.y += (wv.y + wv_prev.y) * scale
        wp_prev.z += (wv.z + wv_prev.z) * scale

        # wr_prev = wr
        # wv_prev = wv
        wr_prev.x = wr.x
        wr_prev.y = wr.y
        wr_prev.z = wr.z
        wv_prev.x = wv.x
        wv_prev.y = wv.y
        wv_prev.z = wv.z

    else: # no Motion
        # Estimate Velocity Bias when not moving
        # When motion ends, velocity should be zero
        if motion_ended and (dtmotion > min_motion_time):
            one_minus_alpha = 1.0 - drift_alpha
            scale = drift_alpha / dtmotion
            wv_drift.x = wv_drift.x * one_minus_alpha + wv.x * scale
            wv_drift.y = wv_drift.y * one_minus_alpha + wv.y * scale
            wv_drift.z = wv_drift.z * one_minus_alpha + wv.z * scale
            dtmotion = 0.0

        # Reset Residuals / Velocity
        wr_prev.x = 0.0
        wr_prev.y = 0.0
        wr_prev.z = 0.0
        wv.x = 0.0
        wv.y = 0.0
        wv.z = 0.0
        wv_prev.x = 0.0
        wv_prev.y = 0.0
        wv_prev.z = 0.0

        # Update acceleration bias, when not moving residuals should be zero
        one_minus_alpha = 1.0 - drift_alpha
        # bias = bias * (1 - alpha) + residual * alpha
        bias.x = bias.x * one_minus_alpha + residual.x * drift_alpha
        bias.y = bias.y * one_minus_alpha + residual.y * drift_alpha
        bias.z = bias.z * one_minus_alpha + residual.z * drift_alpha

    return dtmotion
