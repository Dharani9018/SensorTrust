"""IMU Motion Proxy Extraction.

Extracts motion signals from OXTS IMU data:
    - imu_delta_v:  integrated acceleration over a sliding window (m/s)
    - imu_yaw_rate: direct gyro z-axis reading (rad/s)
"""
import numpy as np


def extract_imu_delta_v(oxts_data, dt=0.1035, window=5):
    """Compute IMU-measured speed change over a sliding window.
    
    Integrates acceleration over the last `window` frames.
    This measures the CHANGE in speed, not absolute speed.
    No drift accumulation because the window is short (~0.5 seconds).
    
    Args:
        oxts_data: pykitti oxts list
        dt: time between frames
        window: number of frames to integrate over
    
    Returns:
        np.array of shape (N,) — speed change over window (m/s)
        First `window` elements are NaN.
    """
    ax = np.array([frame.packet.ax for frame in oxts_data])
    
    delta_v = np.full(len(ax), np.nan)
    
    for i in range(window, len(ax)):
        # Integrate acceleration over the window
        dv = 0.0
        for j in range(i - window + 1, i + 1):
            dv += 0.5 * (ax[j] + ax[j-1]) * dt
        delta_v[i] = dv
    
    return delta_v


def extract_imu_yaw_rate(oxts_data):
    """Extract direct yaw rate from gyro z-axis (packet.wz)."""
    return np.array([frame.packet.wz for frame in oxts_data])


def extract_all_imu_proxies(oxts_data, dt=0.1035, window=5):
    """Extract all IMU motion proxies.
    
    Returns dict with keys:
        delta_v:  speed change over sliding window (m/s)
        yaw_rate: direct gyro reading (rad/s)
    """
    return {
        'delta_v': extract_imu_delta_v(oxts_data, dt, window),
        'yaw_rate': extract_imu_yaw_rate(oxts_data)
    }
