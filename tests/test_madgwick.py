import math

import pytest

from pyIMU.madgwick import Madgwick, updateIMU_inplace
from pyIMU.quaternion import Vector3D, Quaternion
from pyIMU.utilities import rpy2q


def test_madgwick_stationary_stability_marg():
    f = Madgwick(dt=0.01, gain_imu=0.033, gain_marg=0.041)

    q_prev = None
    for _ in range(200):
        q = f.update(
            gyr=Vector3D(0.0, 0.0, 0.0),
            acc=Vector3D(0.0, 0.0, 1.0),
            mag=Vector3D(0.4, 0.1, 0.2),
            dt=0.01,
        )
        q_prev = Quaternion(q)

    q_next = f.update(
        gyr=Vector3D(0.0, 0.0, 0.0),
        acc=Vector3D(0.0, 0.0, 1.0),
        mag=Vector3D(0.4, 0.1, 0.2),
        dt=0.01,
    )

    assert abs(q.norm - 1.0) < 1e-9
    assert abs(q_next.w - q_prev.w) < 1e-4
    assert abs(q_next.x - q_prev.x) < 1e-4
    assert abs(q_next.y - q_prev.y) < 1e-4
    assert abs(q_next.z - q_prev.z) < 1e-4


def test_madgwick_imu_only_path_runs_and_normalizes():
    f = Madgwick(dt=0.01)

    for _ in range(100):
        q = f.update(
            gyr=Vector3D(0.01, -0.02, 0.005),
            acc=Vector3D(0.0, 0.0, 1.0),
            mag=None,
            dt=0.01,
        )

    assert abs(q.norm - 1.0) < 1e-9


def test_madgwick_acc_input_mps2_path():
    f = Madgwick(acc_in_g=False, dt=0.01)

    q = f.update(
        gyr=Vector3D(0.0, 0.0, 0.0),
        acc=Vector3D(0.0, 0.0, 9.80665),
        mag=Vector3D(0.3, 0.0, 0.4),
        dt=0.01,
    )

    assert abs(q.norm - 1.0) < 1e-9


def test_madgwick_update_imu_inplace_mutates_reference():
    q = Quaternion(1.0, 0.0, 0.0, 0.0)
    out = updateIMU_inplace(
        q,
        gyr=Vector3D(0.01, 0.0, 0.0),
        acc=Vector3D(0.0, 0.0, 1.0),
        dt=0.01,
        gain=0.033,
    )

    assert out is q
    assert abs(q.norm - 1.0) < 1e-9


def test_madgwick_acc_outputs_stationary_are_near_zero():
    f = Madgwick(dt=0.01)
    for _ in range(50):
        f.update(
            gyr=Vector3D(0.0, 0.0, 0.0),
            acc=Vector3D(0.0, 0.0, 1.0),
            mag=None,
            dt=0.01,
        )

    assert abs(f.azero.x) < 1e-3
    assert abs(f.azero.y) < 1e-3
    assert abs(f.azero.z) < 1e-3
    assert abs(f.aglobal.x) < 1e-3
    assert abs(f.aglobal.y) < 1e-3
    assert abs(f.aglobal.z) < 1e-3


def test_madgwick_aglobal_rotates_sensor_acceleration_to_earth_frame():
    """
    Uses a pose with 30° roll, 20° pitch, and 10° yaw.
    Constructs sensor measurements from known Earth-frame gravity and linear acceleration.
    Confirms azero remains correct in the sensor frame.
    Confirms aglobal should recover the known Earth-frame acceleration.
    """
    f = Madgwick(dt=0.01)
    f.q = rpy2q(
        math.radians(30.0),
        math.radians(20.0),
        math.radians(10.0),
    )

    earth_linear = Vector3D(0.2, -0.1, 0.05)
    earth_gravity = Vector3D(0.0, 0.0, 1.0)
    sensor_linear = earth_linear.rotate(f.q.r33.T)
    sensor_gravity = earth_gravity.rotate(f.q.r33.T)
    f.acc = sensor_gravity + sensor_linear

    f._update_acc_outputs()

    assert f.azero.x == pytest.approx(sensor_linear.x)
    assert f.azero.y == pytest.approx(sensor_linear.y)
    assert f.azero.z == pytest.approx(sensor_linear.z)
    assert f.aglobal.x == pytest.approx(earth_linear.x)
    assert f.aglobal.y == pytest.approx(earth_linear.y)
    assert f.aglobal.z == pytest.approx(earth_linear.z)


def test_madgwick_exposes_acc_outputs():
    f = Madgwick(dt=0.01)
    f.update(
        gyr=Vector3D(0.01, -0.02, 0.03),
        acc=Vector3D(0.0, 0.0, 1.0),
        mag=None,
        dt=0.01,
    )

    assert isinstance(f.azero, Vector3D)
    assert isinstance(f.aglobal, Vector3D)
