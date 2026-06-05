"""Camera Motion Proxy Extraction.

Extracts mean optical flow magnitude per frame using Farneback dense optical flow.
"""
import numpy as np
import cv2


def extract_optical_flow_magnitude(camera_frames, every_n=1):
    """Compute mean optical flow magnitude between consecutive frames.
    
    Args:
        camera_frames: list of PIL Images or numpy arrays
        every_n: process every Nth frame (1 = all frames)
    
    Returns:
        np.array of shape (N-1,) — mean flow magnitude in px/frame
    """
    flows = []
    
    for i in range(0, len(camera_frames) - 1, every_n):
        # Convert to grayscale numpy arrays
        prev = np.array(camera_frames[i])
        curr = np.array(camera_frames[i + 1])
        
        if len(prev.shape) == 3:
            prev = cv2.cvtColor(prev, cv2.COLOR_RGB2GRAY)
            curr = cv2.cvtColor(curr, cv2.COLOR_RGB2GRAY)
        
        # Farneback dense optical flow
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None,
            pyr_scale=0.5,   # Pyramid scale
            levels=3,         # Pyramid levels
            winsize=15,       # Window size
            iterations=3,     # Iterations per level
            poly_n=5,         # Polynomial neighbourhood
            poly_sigma=1.2,   # Gaussian sigma
            flags=0
        )
        
        # Mean magnitude
        mag = np.mean(np.linalg.norm(flow, axis=2))
        flows.append(mag)
    
    return np.array(flows)


def extract_all_camera_proxies(camera_frames):
    """Extract all camera motion proxies.
    
    Returns dict with key: flow_magnitude
    Note: flow has length N-1 (one less than frames).
    """
    flow = extract_optical_flow_magnitude(camera_frames)
    return {'flow_magnitude': flow}
