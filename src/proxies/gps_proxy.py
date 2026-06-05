"""GPS Motion Proxy Extraction.

Extracts motion signals from OXTS GPS/IMU data:
    - gps_speed:        forward velocity magnitude (m/s)
    - gps_heading:      bearing from consecutive positions (radians)
    - gps_heading_rate: rate of heading change (rad/s)
"""
import numpy as np


def extract_gps_speed(oxts_data):
    """Extract forward speed from GPS (packet.vf)."""
    return np.array([frame.packet.vf for frame in oxts_data])


def extract_gps_heading(oxts_data):
    """Compute heading (bearing) from consecutive GPS positions.
    
    Returns array of shape (N,) in radians [0, 2π).
    First element is NaN.
    """
    lats = np.array([frame.packet.lat for frame in oxts_data])
    lons = np.array([frame.packet.lon for frame in oxts_data])
    
    dlat = np.diff(lats)
    dlon = np.diff(lons)
    
    headings = np.arctan2(dlon, dlat)
    headings = np.mod(headings, 2 * np.pi)
    headings = np.insert(headings, 0, np.nan)
    
    return headings


def extract_gps_heading_rate(oxts_data, dt=0.1035):
    """Compute heading change rate from consecutive headings.
    
    Uses dt = 0.1035s based on your measured ~9.7 Hz GPS frequency.
    First two elements are NaN.
    """
    headings = extract_gps_heading(oxts_data)
    
    heading_diff = np.diff(headings)
    heading_diff = np.arctan2(np.sin(heading_diff), np.cos(heading_diff))
    
    heading_rate = heading_diff / dt
    heading_rate = np.insert(heading_rate, 0, [np.nan, np.nan])
    
    return heading_rate


def extract_all_gps_proxies(oxts_data, dt=0.1035):
    """Extract all GPS motion proxies.
    
    Returns dict with keys: speed, heading, heading_rate
    """
    return {
        'speed': extract_gps_speed(oxts_data),
        'heading': extract_gps_heading(oxts_data),
        'heading_rate': extract_gps_heading_rate(oxts_data, dt)
    }
