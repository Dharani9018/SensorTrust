"""LiDAR Spoofing Attack Injection.

Implements:
    - phantom_inject:  Ghost obstacle points in front of vehicle
    - point_removal:   Erase real points (obstacle hiding)
"""
import numpy as np


def phantom_inject(scan, n_points=100, distance=5.0, width=2.0, height=1.5):
    """Inject phantom obstacle points into a LiDAR scan.
    
    Creates a cluster of fake points directly ahead of the vehicle.
    These have no matching neighbors in ego-motion compensated scans,
    causing ICP residual to spike → F2 detects the anomaly.
    
    KITTI coordinate system: x=forward, y=left, z=up
    
    Args:
        scan: (N, 4) array — x, y, z, intensity
        n_points: number of phantom points
        distance: forward distance in meters
        width: lateral spread in meters
        height: vertical spread in meters
    
    Returns:
        (N + n_points, 4) array with injected points
    """
    phantom = np.zeros((n_points, 4))
    phantom[:, 0] = distance + np.random.uniform(-0.5, 0.5, n_points)
    phantom[:, 1] = np.random.uniform(-width/2, width/2, n_points)
    phantom[:, 2] = np.random.uniform(-0.5, height, n_points)
    phantom[:, 3] = np.random.uniform(0.3, 0.7, n_points)
    
    return np.vstack([scan, phantom])


def inject_phantom_sequence(velo_scans, start_frame, n_points=100, 
                            distance=5.0, duration=None):
    """Inject phantom obstacles into a sequence of LiDAR scans.
    
    Args:
        velo_scans: list of (N, 4) numpy arrays
        start_frame: frame index to start attack
        n_points: phantom points per scan
        distance: forward distance
        duration: number of frames (None = until end)
    
    Returns:
        attacked: list of modified scans
        labels: binary array
    """
    total = len(velo_scans)
    if duration is None:
        duration = total - start_frame
    
    attacked = []
    labels = np.zeros(total, dtype=int)
    
    for i, scan in enumerate(velo_scans):
        if start_frame <= i < start_frame + duration:
            attacked.append(phantom_inject(scan, n_points, distance))
            labels[i] = 1
        else:
            attacked.append(scan.copy())
    
    return attacked, labels


def point_removal(scan, fraction=0.3):
    """Remove a fraction of real points (obstacle hiding attack).
    
    Args:
        scan: (N, 4) array
        fraction: fraction of points to remove
    
    Returns:
        (M, 4) array with fewer points
    """
    n_keep = int(scan.shape[0] * (1 - fraction))
    indices = np.random.choice(scan.shape[0], n_keep, replace=False)
    return scan[indices]
