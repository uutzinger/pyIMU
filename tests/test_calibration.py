import numpy as np

from pyIMU.quaternion import Vector3D
from pyIMU.calibration import (
    InertialCalibration,
    MagnetometerCalibration,
    calibrate_inertial,
    calibrate_magnetic,
)


def test_inertial_identity_defaults():
    raw = Vector3D(1.2, -0.3, 0.7)
    out = InertialCalibration().apply(raw)
    assert abs(out.x - raw.x) < 1e-12
    assert abs(out.y - raw.y) < 1e-12
    assert abs(out.z - raw.z) < 1e-12


def test_inertial_offset_scale_misalignment():
    raw = Vector3D(2.0, 3.0, 4.0)
    model = InertialCalibration(
        offset=Vector3D(1.0, 1.0, 1.0),
        sensitivity=Vector3D(2.0, 3.0, 4.0),
        misalignment=np.array([
            [1.0, 0.1, 0.0],
            [0.0, 1.0, 0.2],
            [0.0, 0.0, 1.0],
        ]),
    )
    out = model.apply(raw)

    # centered = [1,2,3], scaled = [2,6,12]
    # M @ scaled = [2.6, 8.4, 12]
    assert abs(out.x - 2.6) < 1e-12
    assert abs(out.y - 8.4) < 1e-12
    assert abs(out.z - 12.0) < 1e-12


def test_magnetometer_identity_defaults():
    raw = Vector3D(10.0, 20.0, 30.0)
    out = MagnetometerCalibration().apply(raw)
    assert abs(out.x - raw.x) < 1e-12
    assert abs(out.y - raw.y) < 1e-12
    assert abs(out.z - raw.z) < 1e-12


def test_magnetometer_hard_soft_iron():
    raw = Vector3D(2.0, 4.0, 8.0)
    out = calibrate_magnetic(
        raw,
        hard_iron=Vector3D(1.0, 2.0, 4.0),
        soft_iron=np.diag([2.0, 3.0, 4.0]),
    )
    # (raw-hard_iron) = [1,2,4] -> diag scale => [2,6,16]
    assert abs(out.x - 2.0) < 1e-12
    assert abs(out.y - 6.0) < 1e-12
    assert abs(out.z - 16.0) < 1e-12


def test_one_shot_inertial_default_identity():
    raw = Vector3D(-0.1, 0.2, 0.3)
    out = calibrate_inertial(raw)
    assert abs(out.x - raw.x) < 1e-12
    assert abs(out.y - raw.y) < 1e-12
    assert abs(out.z - raw.z) < 1e-12
