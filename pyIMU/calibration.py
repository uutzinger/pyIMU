"""
Sensor calibration models compatible with the x-io/imufusion approach.

Inertial calibration (accelerometer/gyroscope):
    calibrated = misalignment @ (diag(sensitivity) @ (raw - offset))

Magnetometer calibration:
    calibrated = soft_iron @ (raw - hard_iron)
"""

from dataclasses import dataclass, field
from copy import copy
import numpy as np

from pyIMU.quaternion import Vector3D


def _to_vector3d(value) -> Vector3D:
    if isinstance(value, Vector3D):
        return Vector3D(value)
    if isinstance(value, (list, tuple, np.ndarray)) and len(value) == 3:
        return Vector3D(value)
    raise TypeError(f"Expected Vector3D or length-3 array-like, got {type(value)}")


def _to_matrix3(value, default_identity: bool = False) -> np.ndarray:
    if value is None and default_identity:
        return np.eye(3, dtype=float)
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3, 3):
        raise ValueError(f"Expected 3x3 matrix, got shape {arr.shape}")
    return arr


@dataclass
class InertialCalibration:
    """
    Calibration model for accelerometer or gyroscope.

    Defaults are identity calibration:
    - offset = (0, 0, 0)
    - sensitivity = (1, 1, 1)
    - misalignment = I
    """

    offset: Vector3D = field(default_factory=lambda: Vector3D(0.0, 0.0, 0.0))
    sensitivity: Vector3D = field(default_factory=lambda: Vector3D(1.0, 1.0, 1.0))
    misalignment: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))

    def __post_init__(self):
        self.offset = _to_vector3d(self.offset)
        self.sensitivity = _to_vector3d(self.sensitivity)
        self.misalignment = _to_matrix3(self.misalignment)

    def apply(self, raw) -> Vector3D:
        raw_v = _to_vector3d(raw)
        centered = raw_v - self.offset
        scaled = Vector3D(
            centered.x * self.sensitivity.x,
            centered.y * self.sensitivity.y,
            centered.z * self.sensitivity.z,
        )
        calibrated = np.dot(self.misalignment, scaled.v)
        return Vector3D(calibrated)


@dataclass
class MagnetometerCalibration:
    """
    Calibration model for magnetometer.

    Defaults are identity calibration:
    - hard_iron = (0, 0, 0)
    - soft_iron = I
    """

    hard_iron: Vector3D = field(default_factory=lambda: Vector3D(0.0, 0.0, 0.0))
    soft_iron: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=float))

    def __post_init__(self):
        self.hard_iron = _to_vector3d(self.hard_iron)
        self.soft_iron = _to_matrix3(self.soft_iron)

    def apply(self, raw) -> Vector3D:
        raw_v = _to_vector3d(raw)
        centered = raw_v - self.hard_iron
        calibrated = np.dot(self.soft_iron, centered.v)
        return Vector3D(calibrated)


def calibrate_inertial(raw, offset=None, sensitivity=None, misalignment=None) -> Vector3D:
    """One-shot inertial calibration helper."""
    model = InertialCalibration(
        offset=Vector3D(0.0, 0.0, 0.0) if offset is None else copy(_to_vector3d(offset)),
        sensitivity=Vector3D(1.0, 1.0, 1.0) if sensitivity is None else copy(_to_vector3d(sensitivity)),
        misalignment=np.eye(3, dtype=float) if misalignment is None else _to_matrix3(misalignment),
    )
    return model.apply(raw)


def calibrate_magnetic(raw, hard_iron=None, soft_iron=None) -> Vector3D:
    """One-shot magnetometer calibration helper."""
    model = MagnetometerCalibration(
        hard_iron=Vector3D(0.0, 0.0, 0.0) if hard_iron is None else copy(_to_vector3d(hard_iron)),
        soft_iron=np.eye(3, dtype=float) if soft_iron is None else _to_matrix3(soft_iron),
    )
    return model.apply(raw)
