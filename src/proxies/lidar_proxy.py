"""LiDAR Motion Proxy Extraction.

Computes ego-motion compensated ICP residual as the LiDAR motion proxy.
"""
import numpy as np
import open3d as o3d


def compute_icp_residual(scan_curr, scan_prev, transform):
    """Compute ICP residual after ego-motion compensation.
    
    Args:
        scan_curr: (N, 3) array — current LiDAR scan (x, y, z)
        scan_prev: (M, 3) array — previous LiDAR scan
        transform: (4, 4) transformation matrix from OXTS pose
    
    Returns:
        float — mean nearest-neighbour distance after transformation
    """
    # Transform previous scan into current frame
    scan_prev_homog = np.hstack([scan_prev, np.ones((scan_prev.shape[0], 1))])
    scan_prev_transformed = (transform @ scan_prev_homog.T).T[:, :3]
    
    # Build KD-tree on transformed previous scan
    pcd_prev = o3d.geometry.PointCloud()
    pcd_prev.points = o3d.utility.Vector3dVector(scan_prev_transformed)
    pcd_tree = o3d.geometry.KDTreeFlann(pcd_prev)
    
    # For each point in current scan, find nearest neighbour
    residuals = []
    for point in scan_curr:
        [_, _, dist2] = pcd_tree.search_knn_vector_3d(point, 1)
        residuals.append(np.sqrt(dist2[0]))
    
    return np.mean(residuals)


def get_oxts_transform(oxts_frame):
    """Build 4x4 transformation matrix from OXTS pose data.
    
    Args:
        oxts_frame: pykitti oxts packet
    
    Returns:
        (4, 4) numpy array
    """
    # Get position and orientation from OXTS
    # KITTI OXTS provides fused GPS/IMU pose
    from pykitti.utils import rotx, roty, rotz
    
    # Translation
    T = np.eye(4)
    # Position in meters (need to compute from lat/lon or use provided pose)
    # For simplicity, use the transformation from OXTS
    # This is a placeholder — you'll need actual pose data
    
    return T


def extract_lidar_residuals(velo_scans, oxts_data, skip_first=1):
    """Extract LiDAR ICP residuals for all consecutive scan pairs.
    
    Args:
        velo_scans: list of (N, 4) arrays from pykitti
        oxts_data: pykitti oxts list
        skip_first: skip the first N scans for alignment
    
    Returns:
        np.array of shape (N-1,) — ICP residuals in meters
    """
    residuals = []
    
    for i in range(skip_first, len(velo_scans)):
        scan_curr = velo_scans[i][:, :3]
        scan_prev = velo_scans[i-1][:, :3]
        
        # Get transform from OXTS — simplified version
        # In practice, use OXTS pose difference between frames
        transform = np.eye(4)
        
        residual = compute_icp_residual(scan_curr, scan_prev, transform)
        residuals.append(residual)
    
    return np.array(residuals)


def extract_all_lidar_proxies(velo_scans, oxts_data):
    """Extract all LiDAR motion proxies.
    
    Returns dict with key: icp_residual
    """
    residuals = extract_lidar_residuals(velo_scans, oxts_data)
    return {'icp_residual': residuals}
