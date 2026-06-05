"""IMU Motion Proxy Extraction.

Extracts motion signals from OXTS IMU data:
    - imu_speed:    integrated forward acceleration (m/s)
    - imu_yaw_rate: direct gyro z-axis reading (rad/s)
"""
import numpy as np


def extract_imu_speed(oxts_data, dt=0.1035):
    """Integrate forward acceleration to get IMU speed.
    
    Simple trapezoidal integration of packet.ax.
    Assumes initial velocity = 0.
    """
    ax = np.array([frame.packet.ax for frame in oxts_data])
    
    # Cumulative trapezoidal integration
    imu_speed = np.zeros(len(ax))
    for i in range(1, len(ax)):
        imu_speed[i] = imu_speed[i-1] + 0.5 * (ax[i] + ax[i-1]) * dt
    
    # Remove drift bias by detrending on stationary segment if available
    # For now, return raw integration
    return imu_speed


def extract_imu_yaw_rate(oxts_data):
    """Extract direct yaw rate from gyro z-axis (packet.wz)."""
    return np.array([frame.packet.wz for frame in oxts_data])


def extract_all_imu_proxies(oxts_data, dt=0.1035):
    """Extract all IMU motion proxies.
    
    Returns dict with keys: speed, yaw_rate
    """
    return {
        'speed': extract_imu_speed(oxts_data, dt),
        'yaw_rate': extract_imu_yaw_rate(oxts_data)
    }
