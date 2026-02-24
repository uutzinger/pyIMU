from pyIMU.fusion import Fusion
from pyIMU.quaternion import Vector3D, Quaternion


def test_fusion_stationary_identity():
    f = Fusion(k_init=1.0, k_normal=0.1, t_init=0.5)
    q_prev = Quaternion(f.q)

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


def test_fusion_bias_stationary_gating():
    f = Fusion(fc_bias=0.05, omega_min=0.2, t_bias=0.1)
    gyr = Vector3D(0.05, -0.04, 0.03)
    acc = Vector3D(0.0, 0.0, 1.0)

    # keep below omega_min long enough for bias estimator to enable
    for _ in range(100):
        f.update(gyr=gyr, acc=acc, mag=None, dt=0.01)

    assert abs(f.gyro_bias.x) > 0.0
    assert abs(f.gyro_bias.y) > 0.0
    assert abs(f.gyro_bias.z) > 0.0


def test_fusion_rejection_paths_do_not_break_update():
    f = Fusion(
        mag_min=0.8,      # force rejection for provided mag norm
        mag_max=1.2,
        g_deviation=0.02, # reject non-1g accel after short interval
        t_acc_reject=0.02,
    )

    for _ in range(10):
        q = f.update(
            gyr=Vector3D(0.01, 0.01, 0.01),
            acc=Vector3D(0.5, 0.5, 2.0),  # off-nominal acceleration
            mag=Vector3D(0.1, 0.1, 0.1),  # outside mag range
            dt=0.01,
        )

    assert abs(q.norm - 1.0) < 1e-9


def test_fusion_acc_input_mps2_path():
    f = Fusion(acc_in_g=False)

    q = f.update(
        gyr=Vector3D(0.0, 0.0, 0.0),
        acc=Vector3D(0.0, 0.0, 9.80665),
        mag=Vector3D(0.3, 0.0, 0.4),
        dt=0.01,
    )
    assert abs(q.norm - 1.0) < 1e-9


def test_fusion_update_inplace_returns_internal_quaternion():
    f = Fusion()
    q_ref = f.q

    q_out = f.update_inplace(
        gyr=Vector3D(0.0, 0.0, 0.0),
        acc=Vector3D(0.0, 0.0, 1.0),
        mag=None,
        dt=0.01,
    )

    assert q_out is q_ref
    assert f.q is q_ref
    assert abs(q_out.norm - 1.0) < 1e-9
