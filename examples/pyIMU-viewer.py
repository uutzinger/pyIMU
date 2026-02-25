#!/usr/bin/env python3
"""
pyIMU-viewer

PyQt viewer for MPU6050 (Adafruit CircuitPython) using pyIMU Fusion.

Features:
- Reads accelerometer + gyroscope from MPU6050.
- Applies optional accelerometer/gyroscope calibration from JSON.
- Runs pyIMU Fusion (IMU mode, no magnetometer).
- Renders a marked 3D rectangular body in real time.
- Draws gravity-free acceleration vector from body center (fixed 0.1 g full scale).
- Displays heading (yaw) in top-right corner.
- "Zero yaw" button to define current yaw as heading 0.

Dependencies:
- PyQt5
- numpy
- adafruit-circuitpython-mpu6050 (for real sensor mode)
- adafruit-blinka + board + busio (for real sensor mode)

Calibration file format (JSON, all fields optional):
{
  "accelerometer": {
    "offset": [0.0, 0.0, 0.0],
    "sensitivity": [1.0, 1.0, 1.0],
    "misalignment": [[1,0,0],[0,1,0],[0,0,1]]
  },
  "gyroscope": {
    "offset": [0.0, 0.0, 0.0],
    "sensitivity": [1.0, 1.0, 1.0],
    "misalignment": [[1,0,0],[0,1,0],[0,0,1]]
  }
}
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5 import QtCore, QtGui, QtSvg, QtWidgets

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pyIMU.calibration import InertialCalibration
from pyIMU.fusion import Fusion
from pyIMU.quaternion import Vector3D
from pyIMU.utilities import gravity, q2rpy, rpy2q

# Tucson, Arizona (same coordinates used in README motion example).
TUCSON_LATITUDE_DEG = 32.253460
TUCSON_ALTITUDE_M = 730.0
TUCSON_GRAVITY_MS2 = gravity(latitude=TUCSON_LATITUDE_DEG, altitude=TUCSON_ALTITUDE_M)

def wrap_deg(angle_deg: float) -> float:
    return angle_deg % 360.0


def vec3_to_np(v: Vector3D) -> np.ndarray:
    return np.array([float(v.x), float(v.y), float(v.z)], dtype=float)


@dataclass
class CalibSet:
    acc: InertialCalibration
    gyr: InertialCalibration


def _parse_inertial_node(node: Dict) -> InertialCalibration:
    offset = node.get("offset", [0.0, 0.0, 0.0])
    sensitivity = node.get("sensitivity", node.get("scale", [1.0, 1.0, 1.0]))
    misalignment = node.get("misalignment", np.eye(3, dtype=float))
    return InertialCalibration(
        offset=Vector3D(offset),
        sensitivity=Vector3D(sensitivity),
        misalignment=np.array(misalignment, dtype=float),
    )


def load_calibration(path: Optional[str]) -> CalibSet:
    if not path:
        return CalibSet(acc=InertialCalibration(), gyr=InertialCalibration())

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    acc_node = data.get("accelerometer", data.get("acc", {}))
    gyr_node = data.get("gyroscope", data.get("gyro", data.get("gyr", {})))
    return CalibSet(
        acc=_parse_inertial_node(acc_node),
        gyr=_parse_inertial_node(gyr_node),
    )


class SensorSource:
    def read(self) -> Tuple[Vector3D, Vector3D, float]:
        raise NotImplementedError

class MockSource(SensorSource):
    def __init__(self, gravity_ms2: float, speed: float = 0.35):
        self.t0 = time.perf_counter()
        self.last_ts = self.t0
        self.gravity_ms2 = float(gravity_ms2)
        self.speed = max(0.01, float(speed))
        self.acc_dyn_bias = np.zeros(3, dtype=float)
        self.acc_dyn_tau_s = 8.0

    def read(self) -> Tuple[Vector3D, Vector3D, float]:
        t = time.perf_counter() - self.t0
        ts = time.perf_counter()
        dt = max(1e-4, min(0.2, ts - self.last_ts))
        self.last_ts = ts
        tsim = self.speed * t

        # Continuous 360-degree rotations: roll fastest, then pitch, then yaw.
        roll_period = 4.0
        pitch_period = 10.0
        yaw_period = 18.0

        w_roll = (2.0 * math.pi) / roll_period
        w_pitch = (2.0 * math.pi) / pitch_period
        w_yaw = (2.0 * math.pi) / yaw_period

        roll = w_roll * tsim
        pitch = w_pitch * tsim
        yaw = w_yaw * tsim

        acc_dyn_raw = np.array(
            [
                0.04 * math.sin(2.3 * tsim),
                0.02 * math.cos(1.9 * tsim),
                0.01 * math.sin(1.2 * tsim + 0.3),
            ],
            dtype=float,
        ) * self.gravity_ms2
        alpha = max(0.0, min(1.0, dt / max(self.acc_dyn_tau_s, 1e-3)))
        self.acc_dyn_bias += alpha * (acc_dyn_raw - self.acc_dyn_bias)
        acc_dyn = acc_dyn_raw - self.acc_dyn_bias

        q = rpy2q(roll, pitch, yaw)
        r = q.r33
        g_world_ned = np.array([0.0, 0.0, 1.0], dtype=float) * self.gravity_ms2
        g_sensor = r.T @ g_world_ned
        acc = g_sensor + acc_dyn

        gyr = Vector3D(w_roll * self.speed, w_pitch * self.speed, w_yaw * self.speed)
        return gyr, Vector3D(acc), ts


class MPU6050Source(SensorSource):
    def __init__(self, i2c_bus: int = 1, address: int = 0x68):
        del i2c_bus  # API placeholder for future bus selection on Linux hosts.
        import board  # type: ignore
        import busio  # type: ignore
        import adafruit_mpu6050  # type: ignore

        i2c = busio.I2C(board.SCL, board.SDA)
        self.sensor = adafruit_mpu6050.MPU6050(i2c, address=address)

    def read(self) -> Tuple[Vector3D, Vector3D, float]:
        ax, ay, az = self.sensor.acceleration
        gx, gy, gz = self.sensor.gyro
        tmp = self.sensor.temperature 
        return Vector3D(gx, gy, gz), Vector3D(ax, ay, az), time.perf_counter()

class PoseWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAutoFillBackground(True)

        self.q = rpy2q(0.0, 0.0, 0.0)
        self.azero = Vector3D(0.0, 0.0, 0.0)
        self.heading_deg = 0.0
        self.status = "initializing"
        self.acc_ned_ms2 = np.zeros(3, dtype=float)
        self.vel_ned_mps = np.zeros(3, dtype=float)
        self.pos_ned_m = np.zeros(3, dtype=float)
        self.motion_detected = False

        # Rectangular body dimensions (x=length, y=width, z=thickness)
        self.body_dim = np.array([1.8, 1.0, 0.55], dtype=float)
        self.camera_z = 4.6
        self.focal = 1.5
        self.base_scale = 260.0
        self.zoom = 1.0

        self.col_x = QtGui.QColor(46, 108, 255)   # blue
        self.col_y = QtGui.QColor(214, 66, 66)    # red
        self.col_z = QtGui.QColor(46, 164, 79)    # green
        self.col_acc = QtGui.QColor(245, 225, 90)
        self.face_ref_px = 512.0
        self.texture_flip_y = False
        self.svg_world_size = float(self.body_dim[2])  # fixed SVG size = body thickness (z)
        self.rotate_180_faces = {"px", "ny"}  # front and left use opposite-corner anchoring
        self.light_dir = np.array([1.0, 1.0, 1.0], dtype=float)  # top-right-back in viewer space
        self.light_dir /= np.linalg.norm(self.light_dir)
        self.light_ambient = 0.22
        self.light_diffuse = 0.78

        # Keep display/body axes aligned with sensor/NED convention:
        # x forward (North), y right (East), z down (Down) at q=identity.
        self.display_axis_remap = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )

        self.face_definitions = [
            ("pz", [7, 6, 5, 4], np.array([0.0, 0.0, 1.0], dtype=float)),   # +z (flip local y)
            ("nz", [0, 1, 2, 3], np.array([0.0, 0.0, -1.0], dtype=float)),  # -z
            ("px", [5, 6, 2, 1], np.array([1.0, 0.0, 0.0], dtype=float)),   # +x (flip local z)
            ("nx", [0, 3, 7, 4], np.array([-1.0, 0.0, 0.0], dtype=float)),  # -x
            ("py", [3, 2, 6, 7], np.array([0.0, 1.0, 0.0], dtype=float)),   # +y
            ("ny", [4, 5, 1, 0], np.array([0.0, -1.0, 0.0], dtype=float)),  # -y (flip local z)
        ]

        self.face_renderers: Dict[str, QtSvg.QSvgRenderer] = {}
        assets_dir = pathlib.Path(__file__).resolve().parent / "assets" / "pyIMU-viewer"
        for key, _, _ in self.face_definitions:
            svg_path = assets_dir / f"{key}.svg"
            if svg_path.exists():
                renderer = QtSvg.QSvgRenderer(str(svg_path))
                if renderer.isValid():
                    renderer.setAspectRatioMode(QtCore.Qt.KeepAspectRatio)
                    self.face_renderers[key] = renderer

    def set_pose(
        self,
        q,
        azero: Vector3D,
        heading_deg: float,
        status: str,
        acc_ned_ms2: Optional[np.ndarray] = None,
        vel_ned_mps: Optional[np.ndarray] = None,
        pos_ned_m: Optional[np.ndarray] = None,
        motion_detected: Optional[bool] = None,
    ):
        self.q = q
        self.azero = Vector3D(azero)
        self.heading_deg = heading_deg
        self.status = status
        if acc_ned_ms2 is not None:
            self.acc_ned_ms2 = np.array(acc_ned_ms2, dtype=float)
        if vel_ned_mps is not None:
            self.vel_ned_mps = np.array(vel_ned_mps, dtype=float)
        if pos_ned_m is not None:
            self.pos_ned_m = np.array(pos_ned_m, dtype=float)
        if motion_detected is not None:
            self.motion_detected = bool(motion_detected)
        self.update()

    def _project(self, p: np.ndarray, center: QtCore.QPointF) -> Optional[QtCore.QPointF]:
        zc = self.camera_z - p[2]
        if zc <= 0.05:
            return None
        s = self.focal / zc * self.base_scale * self.zoom
        return QtCore.QPointF(center.x() + p[0] * s, center.y() - p[1] * s)

    def _draw_arrow(self, painter: QtGui.QPainter, p0: QtCore.QPointF, p1: QtCore.QPointF, color: QtGui.QColor):
        pen = QtGui.QPen(color, 2)
        painter.setPen(pen)
        painter.drawLine(p0, p1)
        v = QtCore.QLineF(p0, p1)
        if v.length() < 2.0:
            return
        head_len = min(10.0, 0.25 * v.length())
        angle = math.radians(-v.angle())
        left = QtCore.QPointF(
            p1.x() - head_len * math.cos(angle - 0.45),
            p1.y() + head_len * math.sin(angle - 0.45),
        )
        right = QtCore.QPointF(
            p1.x() - head_len * math.cos(angle + 0.45),
            p1.y() + head_len * math.sin(angle + 0.45),
        )
        painter.drawLine(p1, left)
        painter.drawLine(p1, right)

    def _render_svg_on_quad(
        self,
        painter: QtGui.QPainter,
        quad: QtGui.QPolygonF,
        key: str,
        edge_u_world: float,
        edge_v_world: float,
    ):
        renderer = self.face_renderers.get(key)
        if renderer is None:
            return

        vb = renderer.viewBoxF()
        if vb.width() > 0 and vb.height() > 0:
            w = float(vb.width())
            h = float(vb.height())
        else:
            native_size = renderer.defaultSize()
            w = float(max(1, native_size.width()))
            h = float(max(1, native_size.height()))

        # Anchor SVG top-left to face top-left and preserve aspect ratio.
        # Use one global world size for all faces, so scaling does not depend
        # on side aspect ratio/orientation.
        if key in self.rotate_180_faces:
            p0 = quad[2]
            pu = quad[3] - quad[2]
            pv = quad[1] - quad[2]
        else:
            p0 = quad[0]
            pu = quad[1] - quad[0]
            pv = quad[3] - quad[0]
        len_u = math.hypot(pu.x(), pu.y())
        len_v = math.hypot(pv.x(), pv.y())
        if len_u <= 1e-6 or len_v <= 1e-6 or edge_u_world <= 1e-9 or edge_v_world <= 1e-9:
            return

        s_world = self.svg_world_size / max(w, h)
        draw_u_world = min(edge_u_world, s_world * w)
        draw_v_world = min(edge_v_world, s_world * h)
        fu = draw_u_world / edge_u_world
        fv = draw_v_world / edge_v_world

        dst = QtGui.QPolygonF(
            [
                p0,
                QtCore.QPointF(p0.x() + pu.x() * fu, p0.y() + pu.y() * fu),
                QtCore.QPointF(
                    p0.x() + pu.x() * fu + pv.x() * fv,
                    p0.y() + pu.y() * fu + pv.y() * fv,
                ),
                QtCore.QPointF(p0.x() + pv.x() * fv, p0.y() + pv.y() * fv),
            ]
        )

        if self.texture_flip_y:
            src = QtGui.QPolygonF(
                [
                    QtCore.QPointF(0.0, h),
                    QtCore.QPointF(w, h),
                    QtCore.QPointF(w, 0.0),
                    QtCore.QPointF(0.0, 0.0),
                ]
            )
        else:
            src = QtGui.QPolygonF(
                [
                    QtCore.QPointF(0.0, 0.0),
                    QtCore.QPointF(w, 0.0),
                    QtCore.QPointF(w, h),
                    QtCore.QPointF(0.0, h),
                ]
            )
        transform = QtGui.QTransform()
        ok = QtGui.QTransform.quadToQuad(src, dst, transform)
        if not ok:
            return

        painter.save()
        clip_path = QtGui.QPainterPath()
        clip_path.addPolygon(quad)
        painter.setClipPath(clip_path)
        painter.setWorldTransform(transform, False)
        renderer.render(painter, QtCore.QRectF(0.0, 0.0, w, h))
        painter.restore()

    def _point_in_triangle_3d(self, p: np.ndarray, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
        v0 = c - a
        v1 = b - a
        v2 = p - a
        dot00 = float(np.dot(v0, v0))
        dot01 = float(np.dot(v0, v1))
        dot02 = float(np.dot(v0, v2))
        dot11 = float(np.dot(v1, v1))
        dot12 = float(np.dot(v1, v2))
        denom = dot00 * dot11 - dot01 * dot01
        if abs(denom) < 1e-12:
            return False
        inv = 1.0 / denom
        u = (dot11 * dot02 - dot01 * dot12) * inv
        v = (dot00 * dot12 - dot01 * dot02) * inv
        eps = 1e-6
        return (u >= -eps) and (v >= -eps) and (u + v <= 1.0 + eps)

    def _point_in_quad_3d(self, p: np.ndarray, quad: np.ndarray) -> bool:
        a, b, c, d = quad
        return self._point_in_triangle_3d(p, a, b, c) or self._point_in_triangle_3d(p, a, c, d)

    def _point_occluded_by_faces(self, point_world: np.ndarray, camera_world: np.ndarray, face_quads_world: List[np.ndarray]) -> bool:
        ray = point_world - camera_world
        ray_len = float(np.linalg.norm(ray))
        if ray_len <= 1e-9:
            return False

        for quad in face_quads_world:
            v0, v1, _, v3 = quad
            normal = np.cross(v1 - v0, v3 - v0)
            denom = float(np.dot(normal, ray))
            if abs(denom) < 1e-12:
                continue
            t = float(np.dot(normal, v0 - camera_world) / denom)
            # Ignore intersections at the camera, behind camera, or at/very near the queried point.
            if t <= 1e-6 or t >= 1.0 - 1e-5:
                continue
            hit = camera_world + t * ray
            if self._point_in_quad_3d(hit, quad):
                return True
        return False

    def wheelEvent(self, event: QtGui.QWheelEvent):  # noqa: N802
        delta = event.angleDelta().y()
        if delta == 0:
            return
        step = 1.10 if delta > 0 else 1.0 / 1.10
        self.zoom = max(0.25, min(8.0, self.zoom * step))
        self.update()
        event.accept()

    def paintEvent(self, event):  # noqa: N802
        del event
        painter = QtGui.QPainter(self)
        try:
            painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
            painter.fillRect(self.rect(), QtGui.QColor(20, 22, 24))

            cx = self.width() * 0.5
            cy = self.height() * 0.54
            center_2d = QtCore.QPointF(cx, cy)

            # Body vertices in sensor frame.
            hx, hy, hz = 0.5 * self.body_dim
            verts_body = np.array(
                [
                    [-hx, -hy, -hz],
                    [hx, -hy, -hz],
                    [hx, hy, -hz],
                    [-hx, hy, -hz],
                    [-hx, -hy, hz],
                    [hx, -hy, hz],
                    [hx, hy, hz],
                    [-hx, hy, hz],
                ],
                dtype=float,
            )

            r_sensor = self.q.r33
            r = r_sensor @ self.display_axis_remap
            verts_world = (r @ verts_body.T).T
            verts_screen: List[Optional[QtCore.QPointF]] = [self._project(v, center_2d) for v in verts_world]
            face_quads_world = [np.array([verts_world[i] for i in idxs], dtype=float) for _, idxs, _ in self.face_definitions]
            camera_world = np.array([0.0, 0.0, self.camera_z], dtype=float)

            draw_faces = []
            for key, idxs, n_local in self.face_definitions:
                p3d = [verts_world[i] for i in idxs]
                p2d = [verts_screen[i] for i in idxs]
                if any(p is None for p in p2d):
                    continue

                # Use explicit outward normals to avoid missing faces from winding mismatches.
                normal_world = r @ n_local
                center = np.mean(p3d, axis=0)
                cam_vec = np.array([0.0, 0.0, self.camera_z], dtype=float) - center
                if np.dot(normal_world, cam_vec) <= 0.0:
                    continue
                edge_u_world = float(np.linalg.norm(p3d[1] - p3d[0]))
                edge_v_world = float(np.linalg.norm(p3d[3] - p3d[0]))
                depth = np.mean([self.camera_z - p[2] for p in p3d])
                draw_faces.append((depth, p2d, key, edge_u_world, edge_v_world, normal_world))

            draw_faces.sort(reverse=True, key=lambda x: x[0])  # draw far -> near

            for _, p2d, key, edge_u_world, edge_v_world, normal_world in draw_faces:
                poly = QtGui.QPolygonF(p2d)  # type: ignore[arg-type]
                painter.setPen(QtGui.QPen(QtGui.QColor(235, 235, 235), 1.4))
                nrm = float(np.linalg.norm(normal_world))
                if nrm > 1e-9:
                    n_unit = normal_world / nrm
                else:
                    n_unit = normal_world
                lambert = max(0.0, float(np.dot(n_unit, self.light_dir)))
                intensity = self.light_ambient + self.light_diffuse * lambert
                intensity = max(0.0, min(1.0, intensity))
                gray = int(round(255.0 * intensity))
                painter.setBrush(QtGui.QColor(gray, gray, gray, 255))
                painter.drawPolygon(poly)
                self._render_svg_on_quad(
                    painter=painter,
                    quad=poly,
                    key=key,
                    edge_u_world=edge_u_world,
                    edge_v_world=edge_v_world,
                )

            # Draw acceleration vector in body frame, fixed full-scale at 0.1 g.
            a_body = vec3_to_np(self.azero)
            a_mag = float(np.linalg.norm(a_body))
            a_clip = min(a_mag, 0.1)
            if a_mag > 1e-8:
                a_body_scaled = a_body * (a_clip / a_mag) * (0.9 / 0.1)
            else:
                a_body_scaled = np.zeros(3, dtype=float)
            p0 = self._project(np.zeros(3, dtype=float), center_2d)
            p1 = self._project(r_sensor @ a_body_scaled, center_2d)
            if p0 is not None and p1 is not None:
                painter.setPen(QtGui.QPen(self.col_acc, 2))
                painter.drawLine(p0, p1)
                painter.setBrush(self.col_acc)
                painter.setPen(QtGui.QPen(self.col_acc, 1))
                painter.drawEllipse(p1, 7.0, 7.0)
                painter.setBrush(QtCore.Qt.NoBrush)
                painter.setPen(QtGui.QPen(self.col_acc, 1.5))
                painter.drawText(p1 + QtCore.QPointF(10, -8), f"|a|={a_mag:.3f} g")

            # Color-coded hairline axis triad from front-right-bottom corner.
            hair_len = 0.5 * float(self.body_dim[2])  # half of thickness
            corner_local = np.array([hx, hy, hz], dtype=float)  # front-right-bottom
            corner_world = r @ corner_local
            end_x = corner_world + r @ np.array([hair_len, 0.0, 0.0], dtype=float)
            end_y = corner_world + r @ np.array([0.0, hair_len, 0.0], dtype=float)
            end_z = corner_world + r @ np.array([0.0, 0.0, hair_len], dtype=float)

            p_corner = self._project(corner_world, center_2d)
            p_x = self._project(end_x, center_2d)
            p_y = self._project(end_y, center_2d)
            p_z = self._project(end_z, center_2d)
            corner_visible = not self._point_occluded_by_faces(corner_world, camera_world, face_quads_world)
            if p_corner is not None and corner_visible:
                if p_x is not None:
                    painter.setPen(QtGui.QPen(self.col_x, 1.4))
                    painter.drawLine(p_corner, p_x)
                    painter.drawText(p_x + QtCore.QPointF(4, -4), "x")
                if p_y is not None:
                    painter.setPen(QtGui.QPen(self.col_y, 1.4))
                    painter.drawLine(p_corner, p_y)
                    painter.drawText(p_y + QtCore.QPointF(4, -4), "y")
                if p_z is not None:
                    painter.setPen(QtGui.QPen(self.col_z, 1.4))
                    painter.drawLine(p_corner, p_z)
                    painter.drawText(p_z + QtCore.QPointF(4, -4), "z")

            # Color-coded negative-axis hairline triad from back-left-top corner.
            corner2_local = np.array([-hx, -hy, -hz], dtype=float)  # back-left-top
            corner2_world = r @ corner2_local
            end_nx = corner2_world + r @ np.array([-hair_len, 0.0, 0.0], dtype=float)
            end_ny = corner2_world + r @ np.array([0.0, -hair_len, 0.0], dtype=float)
            end_nz = corner2_world + r @ np.array([0.0, 0.0, -hair_len], dtype=float)

            p_corner2 = self._project(corner2_world, center_2d)
            p_nx = self._project(end_nx, center_2d)
            p_ny = self._project(end_ny, center_2d)
            p_nz = self._project(end_nz, center_2d)
            corner2_visible = not self._point_occluded_by_faces(corner2_world, camera_world, face_quads_world)
            if p_corner2 is not None and corner2_visible:
                if p_nx is not None:
                    painter.setPen(QtGui.QPen(self.col_x, 1.4))
                    painter.drawLine(p_corner2, p_nx)
                    painter.drawText(p_nx + QtCore.QPointF(4, -4), "-x")
                if p_ny is not None:
                    painter.setPen(QtGui.QPen(self.col_y, 1.4))
                    painter.drawLine(p_corner2, p_ny)
                    painter.drawText(p_ny + QtCore.QPointF(4, -4), "-y")
                if p_nz is not None:
                    painter.setPen(QtGui.QPen(self.col_z, 1.4))
                    painter.drawLine(p_corner2, p_nz)
                    painter.drawText(p_nz + QtCore.QPointF(4, -4), "-z")

            painter.setPen(QtGui.QPen(QtGui.QColor(235, 235, 235), 1))
            painter.setFont(QtGui.QFont("DejaVu Sans Mono", 11))
            heading_text = f"Heading: {self.heading_deg:6.1f} deg"
            stat_text = f"Source: {self.status}"
            a_mag = float(np.linalg.norm(self.acc_ned_ms2))
            v_mag = float(np.linalg.norm(self.vel_ned_mps))
            p_mag = float(np.linalg.norm(self.pos_ned_m))
            row_a = f"a[NED] {self.acc_ned_ms2[0]:+8.3f} {self.acc_ned_ms2[1]:+8.3f} {self.acc_ned_ms2[2]:+8.3f} |{a_mag:7.3f}| m/s^2"
            row_v = f"v[NED] {self.vel_ned_mps[0]:+8.3f} {self.vel_ned_mps[1]:+8.3f} {self.vel_ned_mps[2]:+8.3f} |{v_mag:7.3f}| m/s"
            row_p = f"p[NED] {self.pos_ned_m[0]:+8.3f} {self.pos_ned_m[1]:+8.3f} {self.pos_ned_m[2]:+8.3f} |{p_mag:7.3f}| m"
            fm = painter.fontMetrics()
            text_w = max(
                fm.horizontalAdvance(heading_text),
                fm.horizontalAdvance(row_a),
                fm.horizontalAdvance(row_v),
                fm.horizontalAdvance(row_p),
            )
            x_right = max(10, self.width() - text_w - 14)
            painter.drawText(x_right, 26, heading_text)
            painter.drawText(x_right, 46, row_a)
            painter.drawText(x_right, 66, row_v)
            painter.drawText(x_right, 86, row_p)

            led_color = QtGui.QColor(52, 220, 92) if self.motion_detected else QtGui.QColor(75, 75, 75)
            mot_x = x_right + fm.horizontalAdvance(heading_text) + 12
            painter.setBrush(led_color)
            painter.setPen(QtGui.QPen(QtGui.QColor(230, 230, 230), 1))
            painter.drawEllipse(QtCore.QPointF(mot_x, 20), 6.0, 6.0)
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.setPen(QtGui.QPen(QtGui.QColor(235, 235, 235), 1))
            painter.drawText(mot_x + 12, 26, "MOT")
            painter.drawText(14, 26, stat_text)
        finally:
            painter.end()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        source: SensorSource,
        calib: CalibSet,
        update_hz: float = 60.0,
    ):
        super().__init__()
        self.setWindowTitle("pyIMU-viewer")
        self.resize(920, 720)

        self.source = source
        self.calib = calib
        self.local_gravity_ms2 = TUCSON_GRAVITY_MS2
        self.fusion = Fusion(
            dt=1.0 / max(1.0, update_hz),
            convention="NED",
            acc_in_g=True,
            gyr_in_dps=False,
            k_init=10.0,
            k_normal=0.5,
        )

        self.yaw_zero = 0.0
        self.last_timestamp: Optional[float] = None
        self.source_name = type(source).__name__
        self.acc_ned_ms2 = np.zeros(3, dtype=float)
        self.vel_ned_mps = np.zeros(3, dtype=float)
        self.pos_ned_m = np.zeros(3, dtype=float)
        self.prev_acc_ned_ms2: Optional[np.ndarray] = None
        self.prev_vel_ned_mps = np.zeros(3, dtype=float)
        self.motion_detected = False

        root = QtWidgets.QWidget()
        self.setCentralWidget(root)
        layout = QtWidgets.QVBoxLayout(root)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        self.pose_widget = PoseWidget()
        layout.addWidget(self.pose_widget, stretch=1)

        controls = QtWidgets.QHBoxLayout()
        self.btn_zero_yaw = QtWidgets.QPushButton("Zero yaw")
        self.btn_zero_yaw.clicked.connect(self._on_zero_yaw)
        controls.addWidget(self.btn_zero_yaw)
        self.btn_reset_vel = QtWidgets.QPushButton("Reset Velocity")
        self.btn_reset_vel.clicked.connect(self._on_reset_velocity)
        controls.addWidget(self.btn_reset_vel)
        self.btn_reset_pos = QtWidgets.QPushButton("Reset Position")
        self.btn_reset_pos.clicked.connect(self._on_reset_position)
        controls.addWidget(self.btn_reset_pos)
        self.btn_reset_both = QtWidgets.QPushButton("Reset V+P")
        self.btn_reset_both.clicked.connect(self._on_reset_vp)
        controls.addWidget(self.btn_reset_both)
        controls.addStretch(1)
        layout.addLayout(controls)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._tick)
        self.timer.start(max(5, int(1000.0 / max(update_hz, 1.0))))

    def _on_zero_yaw(self):
        yaw = q2rpy(self.fusion.q).z
        self.yaw_zero = yaw

    def _on_reset_velocity(self):
        self.vel_ned_mps[:] = 0.0
        self.prev_vel_ned_mps[:] = 0.0
        self.prev_acc_ned_ms2 = None

    def _on_reset_position(self):
        self.pos_ned_m[:] = 0.0

    def _on_reset_vp(self):
        self._on_reset_velocity()
        self._on_reset_position()

    def _tick(self):
        try:
            gyr_raw, acc_raw, ts = self.source.read()
        except Exception as exc:
            self.pose_widget.set_pose(
                q=self.fusion.q,
                azero=Vector3D(0.0, 0.0, 0.0),
                heading_deg=0.0,
                status=f"sensor error: {exc}",
                acc_ned_ms2=self.acc_ned_ms2,
                vel_ned_mps=self.vel_ned_mps,
                pos_ned_m=self.pos_ned_m,
                motion_detected=self.motion_detected,
            )
            return

        gyr = self.calib.gyr.apply(gyr_raw)
        acc_ms2 = self.calib.acc.apply(acc_raw)
        acc = acc_ms2 / self.local_gravity_ms2

        if self.last_timestamp is None:
            dt = self.fusion.dt
        else:
            dt = float(max(1e-4, min(0.2, ts - self.last_timestamp)))
        self.last_timestamp = ts

        q = self.fusion.update_inplace(gyr=gyr, acc=acc, mag=None, dt=dt)

        rpy = q2rpy(q)
        heading_deg = wrap_deg(math.degrees(rpy.z - self.yaw_zero))
        q_display = rpy2q(0.0, 0.0, -self.yaw_zero) * q

        # Gravity-removed acceleration from Fusion in earth frame, in g -> m/s^2.
        self.acc_ned_ms2 = np.array(
            [self.fusion.aglobal.x, self.fusion.aglobal.y, self.fusion.aglobal.z],
            dtype=float,
        ) * self.local_gravity_ms2

        old_vel = self.vel_ned_mps.copy()
        if self.prev_acc_ned_ms2 is None:
            self.vel_ned_mps += self.acc_ned_ms2 * dt
        else:
            self.vel_ned_mps += 0.5 * (self.prev_acc_ned_ms2 + self.acc_ned_ms2) * dt
        self.pos_ned_m += 0.5 * (old_vel + self.vel_ned_mps) * dt
        self.prev_vel_ned_mps = self.vel_ned_mps.copy()
        self.prev_acc_ned_ms2 = self.acc_ned_ms2.copy()
        self.motion_detected = (float(np.linalg.norm(self.acc_ned_ms2)) > 0.05) or (float(np.linalg.norm(self.vel_ned_mps)) > 0.02)

        self.pose_widget.set_pose(
            q=q_display,
            azero=self.fusion.azero,
            heading_deg=heading_deg,
            status=self.source_name,
            acc_ned_ms2=self.acc_ned_ms2,
            vel_ned_mps=self.vel_ned_mps,
            pos_ned_m=self.pos_ned_m,
            motion_detected=self.motion_detected,
        )


def build_source(use_mock: bool, i2c_bus: int, address: int, gravity_ms2: float, mock_speed: float) -> SensorSource:
    if use_mock:
        return MockSource(gravity_ms2=gravity_ms2, speed=mock_speed)
    try:
        return MPU6050Source(i2c_bus=i2c_bus, address=address)
    except Exception as exc:
        print(f"[pyIMU-viewer] Falling back to mock source: {exc}", file=sys.stderr)
        return MockSource(gravity_ms2=gravity_ms2, speed=mock_speed)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MPU6050 + Fusion PyQt pose viewer")
    p.add_argument("--calibration", type=str, default=None, help="Calibration JSON file path.")
    p.add_argument("--hz", type=float, default=60.0, help="UI/filter update frequency in Hz.")
    p.add_argument("--i2c-bus", type=int, default=1, help="I2C bus index (placeholder on Blinka hosts).")
    p.add_argument("--address", type=lambda x: int(x, 0), default=0x68, help="MPU6050 I2C address (e.g. 0x68).")
    p.add_argument("--mock", action="store_true", help="Force mock data mode.")
    p.add_argument("--mock-speed", type=float, default=0.35, help="Mock motion speed multiplier (1.0 = original).")
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)
    calib = load_calibration(args.calibration)
    source = build_source(args.mock, args.i2c_bus, args.address, TUCSON_GRAVITY_MS2, args.mock_speed)

    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow(source=source, calib=calib, update_hz=max(5.0, args.hz))
    win.show()
    return app.exec_()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
