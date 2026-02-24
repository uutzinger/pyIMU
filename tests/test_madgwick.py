from pyIMU.madgwick import Madgwick, updateIMU_inplace
from pyIMU.quaternion import Vector3D, Quaternion


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
