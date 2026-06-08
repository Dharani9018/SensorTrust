import numpy as np


def compute_f1(
        z,
        gps_speed,
        heading_gate=2.0
):
    """
    Feature 1 — GPS–IMU Kinematic Consistency

    F1 =
        |z_gps_delta_v - z_imu_delta_v|
      + |z_gps_heading_rate - z_imu_yaw_rate|

    Heading component is enabled only when
    GPS speed exceeds heading_gate.
    """

    speed_consistency = np.abs(
        z["gps_delta_v"] -
        z["imu_delta_v"]
    )

    heading_consistency = np.where(
        gps_speed > heading_gate,
        np.abs(
            z["gps_heading_rate"] -
            z["imu_yaw_rate"]
        ),
        0.0
    )

    f1 = (
        speed_consistency +
        heading_consistency
    )

    return {
        "speed_consistency": speed_consistency,
        "heading_consistency": heading_consistency,
        "f1": f1
    }


def extract_all_f1_features(
        z,
        gps_speed,
        heading_gate=2.0
):
    """
    Wrapper for project pipeline.
    """

    min_len = min(
        len(z["gps_delta_v"]),
        len(z["imu_delta_v"]),
        len(z["gps_heading_rate"]),
        len(z["imu_yaw_rate"]),
        len(gps_speed)
    )

    z_trim = {
        "gps_delta_v": z["gps_delta_v"][:min_len],
        "imu_delta_v": z["imu_delta_v"][:min_len],
        "gps_heading_rate": z["gps_heading_rate"][:min_len],
        "imu_yaw_rate": z["imu_yaw_rate"][:min_len]
    }

    gps_speed_trim = gps_speed[:min_len]

    return compute_f1(
        z=z_trim,
        gps_speed=gps_speed_trim,
        heading_gate=heading_gate
    )