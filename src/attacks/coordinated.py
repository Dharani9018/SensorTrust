"""Coordinated Multi-Sensor Spoofing Attacks.

Implements:
    - gps_imu:       GPS + IMU spoofed simultaneously (Jung & Yoon bypass pattern)
    - gps_lidar:     GPS + LiDAR spoofed together
    - gps_camera:    GPS + Camera spoofed together
    - imu_lidar:     IMU + LiDAR spoofed together
    - all_four:      All four sensors spoofed consistently
"""
import numpy as np
from .gps_attack import gps_speed_spoof, gps_slow_drift
from .lidar_attack import phantom_inject


def coordinated_gps_imu(oxts_data, velo_scans, start_frame, 
                        gps_speed_offset=5.0, imu_bias=0.5, duration=50):
    """Coordinated GPS + IMU attack.
    
    GPS speed is spoofed. IMU acceleration has a correlated bias added.
    Each sensor looks individually plausible to single-sensor detectors.
    F1 (speed) may not spike if bias is well-chosen.
    F2 and GMIS catch the residual inconsistency with LiDAR/Camera.
    
    This directly implements the Jung & Yoon bypass scenario.
    
    Args:
        oxts_data: pykitti oxts list
        velo_scans: list of LiDAR scans (for F2 validation)
        start_frame: attack start
        gps_speed_offset: m/s to add to GPS vf
        imu_bias: m/s² to add to IMU ax
        duration: number of frames
    
    Returns:
        gps_attacked: list of dicts
        imu_attacked: list of dicts
        lidar_attacked: list (unchanged LiDAR)
        labels: binary array
    """
    total = len(oxts_data)
    labels = np.zeros(total, dtype=int)
    
    gps_attacked = []
    imu_attacked = []
    lidar_attacked = []
    
    for i, frame in enumerate(oxts_data):
        pkt = frame.packet
        gps_entry = {
            'lat': pkt.lat, 'lon': pkt.lon, 'alt': pkt.alt,
            'vf': pkt.vf, 'vl': pkt.vl, 'vu': pkt.vu,
            'ax': pkt.ax, 'ay': pkt.ay, 'az': pkt.az,
            'wx': pkt.wx, 'wy': pkt.wy, 'wz': pkt.wz
        }
        imu_entry = {
            'lat': pkt.lat, 'lon': pkt.lon, 'alt': pkt.alt,
            'vf': pkt.vf, 'vl': pkt.vl, 'vu': pkt.vu,
            'ax': pkt.ax, 'ay': pkt.ay, 'az': pkt.az,
            'wx': pkt.wx, 'wy': pkt.wy, 'wz': pkt.wz
        }
        
        if start_frame <= i < start_frame + duration:
            gps_entry['vf'] += gps_speed_offset
            imu_entry['ax'] += imu_bias
            labels[i] = 1
        
        gps_attacked.append(gps_entry)
        imu_attacked.append(imu_entry)
        
        if i < len(velo_scans):
            lidar_attacked.append(velo_scans[i].copy())
    
    return gps_attacked, imu_attacked, lidar_attacked, labels


def coordinated_gps_lidar(oxts_data, velo_scans, start_frame,
                          gps_speed_offset=5.0, phantom_points=100, duration=50):
    """Coordinated GPS + LiDAR attack.
    
    GPS speed and LiDAR scene are both spoofed.
    F2 may not spike if both are manipulated consistently.
    F1 and GMIS catch the inconsistency with IMU/Camera.
    
    Args:
        oxts_data: pykitti oxts list
        velo_scans: list of LiDAR scans
        start_frame: attack start
        gps_speed_offset: m/s
        phantom_points: number of phantom points per scan
        duration: number of frames
    
    Returns:
        gps_attacked, lidar_attacked, labels
    """
    total = len(oxts_data)
    labels = np.zeros(total, dtype=int)
    
    gps_attacked = []
    lidar_attacked = []
    
    for i, frame in enumerate(oxts_data):
        pkt = frame.packet
        gps_entry = {
            'lat': pkt.lat, 'lon': pkt.lon, 'alt': pkt.alt,
            'vf': pkt.vf, 'vl': pkt.vl, 'vu': pkt.vu,
            'ax': pkt.ax, 'ay': pkt.ay, 'az': pkt.az,
            'wx': pkt.wx, 'wy': pkt.wy, 'wz': pkt.wz
        }
        
        if start_frame <= i < start_frame + duration:
            gps_entry['vf'] += gps_speed_offset
            labels[i] = 1
        
        gps_attacked.append(gps_entry)
        
        if i < len(velo_scans):
            if start_frame <= i < start_frame + duration:
                lidar_attacked.append(phantom_inject(velo_scans[i], phantom_points))
            else:
                lidar_attacked.append(velo_scans[i].copy())
    
    return gps_attacked, lidar_attacked, labels


def coordinated_gps_camera(oxts_data, camera_frames, start_frame,
                           gps_speed_offset=5.0, cam_noise_std=25.0, duration=50):
    """Coordinated GPS + Camera attack.
    
    GPS speed spoofed. Camera has Gaussian noise added to degrade optical flow.
    Tests symmetric design — no single sensor is trusted.
    
    Args:
        oxts_data: pykitti oxts list
        camera_frames: list of numpy image arrays
        start_frame: attack start
        gps_speed_offset: m/s
        cam_noise_std: standard deviation of Gaussian noise
        duration: number of frames
    
    Returns:
        gps_attacked, camera_attacked, labels
    """
    total = len(oxts_data)
    labels = np.zeros(total, dtype=int)
    
    gps_attacked = []
    camera_attacked = []
    
    for i, frame in enumerate(oxts_data):
        pkt = frame.packet
        gps_entry = {
            'lat': pkt.lat, 'lon': pkt.lon, 'alt': pkt.alt,
            'vf': pkt.vf, 'vl': pkt.vl, 'vu': pkt.vu,
            'ax': pkt.ax, 'ay': pkt.ay, 'az': pkt.az,
            'wx': pkt.wx, 'wy': pkt.wy, 'wz': pkt.wz
        }
        
        if start_frame <= i < start_frame + duration:
            gps_entry['vf'] += gps_speed_offset
            labels[i] = 1
        
        gps_attacked.append(gps_entry)
        
        if i < len(camera_frames):
            if start_frame <= i < start_frame + duration:
                noisy = camera_frames[i].astype(np.float32)
                noisy += np.random.normal(0, cam_noise_std, noisy.shape)
                noisy = np.clip(noisy, 0, 255).astype(np.uint8)
                camera_attacked.append(noisy)
            else:
                camera_attacked.append(camera_frames[i].copy())
    
    return gps_attacked, camera_attacked, labels


def coordinated_all_four(oxts_data, velo_scans, camera_frames, start_frame,
                         gps_speed_offset=5.0, imu_bias=0.3,
                         phantom_points=50, cam_noise_std=15.0, duration=50):
    """All four sensors coordinated attack.
    
    Ultimate test case. All sensors spoofed simultaneously but consistently.
    Even this should produce detectable GMIS if the spoofing is not perfectly
    physically consistent across all modalities.
    
    Returns:
        gps_attacked, imu_attacked, lidar_attacked, camera_attacked, labels
    """
    total = len(oxts_data)
    labels = np.zeros(total, dtype=int)
    
    gps_attacked = []
    imu_attacked = []
    lidar_attacked = []
    camera_attacked = []
    
    for i, frame in enumerate(oxts_data):
        pkt = frame.packet
        gps_entry = {
            'lat': pkt.lat, 'lon': pkt.lon, 'alt': pkt.alt,
            'vf': pkt.vf, 'vl': pkt.vl, 'vu': pkt.vu,
            'ax': pkt.ax, 'ay': pkt.ay, 'az': pkt.az,
            'wx': pkt.wx, 'wy': pkt.wy, 'wz': pkt.wz
        }
        imu_entry = {
            'lat': pkt.lat, 'lon': pkt.lon, 'alt': pkt.alt,
            'vf': pkt.vf, 'vl': pkt.vl, 'vu': pkt.vu,
            'ax': pkt.ax, 'ay': pkt.ay, 'az': pkt.az,
            'wx': pkt.wx, 'wy': pkt.wy, 'wz': pkt.wz
        }
        
        if start_frame <= i < start_frame + duration:
            gps_entry['vf'] += gps_speed_offset
            imu_entry['ax'] += imu_bias
            labels[i] = 1
        
        gps_attacked.append(gps_entry)
        imu_attacked.append(imu_entry)
        
        if i < len(velo_scans):
            if start_frame <= i < start_frame + duration:
                lidar_attacked.append(phantom_inject(velo_scans[i], phantom_points))
            else:
                lidar_attacked.append(velo_scans[i].copy())
        
        if i < len(camera_frames):
            if start_frame <= i < start_frame + duration:
                noisy = camera_frames[i].astype(np.float32)
                noisy += np.random.normal(0, cam_noise_std, noisy.shape)
                noisy = np.clip(noisy, 0, 255).astype(np.uint8)
                camera_attacked.append(noisy)
            else:
                camera_attacked.append(camera_frames[i].copy())
    
    return gps_attacked, imu_attacked, lidar_attacked, camera_attacked, labels
