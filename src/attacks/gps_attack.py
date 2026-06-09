"""GPS Spoofing Attack Injection.

Implements:
    - step_offset:  Sudden position jump
    - slow_drift:   Gradual drift below naive detection thresholds
"""
import numpy as np
import copy


def gps_step_offset(oxts_data, start_frame, lat_offset=0.001, lon_offset=0.002, duration=None):
    """Sudden GPS position jump (step attack).
    
    Args:
        oxts_data: pykitti oxts list
        start_frame: frame index where attack begins
        lat_offset: latitude offset to add
        lon_offset: longitude offset to add
        duration: number of frames to sustain attack (None = until end)
    
    Returns:
        attacked: list of dicts with spoofed GPS values
        labels: binary array (0=clean, 1=attacked)
    """
    total = len(oxts_data)
    if duration is None:
        duration = total - start_frame
    
    attacked = []
    labels = np.zeros(total, dtype=int)
    
    for i, frame in enumerate(oxts_data):
        pkt = frame.packet
        entry = {
            'lat': pkt.lat, 'lon': pkt.lon, 'alt': pkt.alt,
            'vf': pkt.vf, 'vl': pkt.vl, 'vu': pkt.vu,
            'ax': pkt.ax, 'ay': pkt.ay, 'az': pkt.az,
            'wx': pkt.wx, 'wy': pkt.wy, 'wz': pkt.wz
        }
        
        if start_frame <= i < start_frame + duration:
            entry['lat'] += lat_offset
            entry['lon'] += lon_offset
            labels[i] = 1
        
        attacked.append(entry)
    
    return attacked, labels


def gps_slow_drift(oxts_data, start_frame, drift_per_frame=0.00001, duration=None):
    """Gradual GPS drift attack (evasive).
    
    Accumulates small position offset each frame.
    Below typical single-sensor detection thresholds.
    
    Args:
        oxts_data: pykitti oxts list
        start_frame: frame index where attack begins
        drift_per_frame: offset added per frame
        duration: number of frames (None = until end)
    
    Returns:
        attacked: list of dicts with spoofed GPS values
        labels: binary array
    """
    total = len(oxts_data)
    if duration is None:
        duration = total - start_frame
    
    attacked = []
    labels = np.zeros(total, dtype=int)
    drift = 0.0
    
    for i, frame in enumerate(oxts_data):
        pkt = frame.packet
        entry = {
            'lat': pkt.lat, 'lon': pkt.lon, 'alt': pkt.alt,
            'vf': pkt.vf, 'vl': pkt.vl, 'vu': pkt.vu,
            'ax': pkt.ax, 'ay': pkt.ay, 'az': pkt.az,
            'wx': pkt.wx, 'wy': pkt.wy, 'wz': pkt.wz
        }
        
        if start_frame <= i < start_frame + duration:
            drift += drift_per_frame
            entry['lat'] += drift
            entry['lon'] += drift
            labels[i] = 1
        
        attacked.append(entry)
    
    return attacked, labels


def gps_speed_spoof(oxts_data, start_frame, speed_offset=5.0, duration=None):
    """Direct forward speed spoofing.
    
    Adds a fixed offset to GPS forward velocity.
    This directly triggers F1 (speed mismatch with IMU).
    
    Args:
        oxts_data: pykitti oxts list
        start_frame: frame index
        speed_offset: m/s to add to vf
        duration: number of frames
    
    Returns:
        attacked: list of dicts
        labels: binary array
    """
    total = len(oxts_data)
    if duration is None:
        duration = total - start_frame
    
    attacked = []
    labels = np.zeros(total, dtype=int)
    
    for i, frame in enumerate(oxts_data):
        pkt = frame.packet
        entry = {
            'lat': pkt.lat, 'lon': pkt.lon, 'alt': pkt.alt,
            'vf': pkt.vf, 'vl': pkt.vl, 'vu': pkt.vu,
            'ax': pkt.ax, 'ay': pkt.ay, 'az': pkt.az,
            'wx': pkt.wx, 'wy': pkt.wy, 'wz': pkt.wz
        }
        
        if start_frame <= i < start_frame + duration:
            entry['vf'] += speed_offset
            labels[i] = 1
        
        attacked.append(entry)
    
    return attacked, labels
