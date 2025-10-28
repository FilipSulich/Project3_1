"""
Extract 3D model points from a Blender frame.

This takes one frame's depth map and mask, converts it to a point cloud,
and transforms it to the object's local coordinate frame (removes the pose).
The result is the canonical 3D model that DenseFusion needs. It is used as a default pose model while training (the predicted rotation and translation is applied to this pose to get the observed one).
"""

import numpy as np
import json
from PIL import Image
import OpenEXR
import Imath
import os


def load_depth_exr(path):
    """Load depth from EXR file"""
    exr_file = OpenEXR.InputFile(path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1

    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    channel_name = 'R' if 'R' in header['channels'] else 'Z'
    depth_str = exr_file.channel(channel_name, FLOAT)
    depth = np.frombuffer(depth_str, dtype=np.float32).reshape((height, width))
    return depth


def depth_to_point_cloud(depth, mask, cam_fx, cam_fy, cam_cx, cam_cy):
    """Convert depth map to 3D point cloud"""
    height, width = depth.shape

    # Create coordinate grids
    ymap = np.array([[j for i in range(width)] for j in range(height)])
    xmap = np.array([[i for i in range(width)] for j in range(height)])

    # Get masked pixels
    mask_bool = mask > 0

    # Get depth values
    z = depth[mask_bool]
    y_coords = ymap[mask_bool]
    x_coords = xmap[mask_bool]

    # Back-project to 3D
    x = (x_coords - cam_cx) * z / cam_fx
    y = (y_coords - cam_cy) * z / cam_fy

    points = np.stack([x, y, z], axis=1)
    return points


def extract_model_points(root_dir, frame_idx=0, num_points=1000):
    """
    Extract canonical 3D model points from a specific frame.

    Args:
        root_dir: Directory with Blender data
        frame_idx: Which frame to use (use one where object is well-visible)
        num_points: Target number of points to keep
    """

    # Load camera intrinsics
    camera_json = os.path.join(root_dir, "/Users/filipsulich/Developer/uni/Project3_1/DenseFusionModel/Project31/DenseFusion/datasets/Blender/camera_settings.json")
    with open(camera_json, 'r') as f:
        cam_data = json.load(f)
        K = np.array(cam_data['K'], dtype=np.float32)
        cam_fx, cam_fy = K[0, 0], K[1, 1]
        cam_cx, cam_cy = K[0, 2], K[1, 2]

    # Get frame number (assumes format frame_XXXX_rgb.png)
    import glob
    rgb_files = sorted(glob.glob(os.path.join(root_dir, "*_rgb.png")))
    if frame_idx >= len(rgb_files):
        frame_idx = 0

    rgb_path = rgb_files[frame_idx]
    base_name = os.path.splitext(rgb_path)[0].replace("_rgb", "")
    frame_num = base_name.split('_')[-1]

    depth_path = os.path.join(root_dir, f"frame_{frame_num}_depth.exr")
    mask_path = os.path.join(root_dir, f"frame_{frame_num}_mask.png")
    meta_path = os.path.join(root_dir, f"frame_{frame_num}_meta.json")

    print(f"Using frame: {frame_num}")

    # Load data
    depth = load_depth_exr(depth_path)
    mask = np.array(Image.open(mask_path).convert('L'))

    with open(meta_path, 'r') as f:
        meta = json.load(f)

    # Get pose matrix (world to camera transform)
    pose_matrix = np.array(meta['pose_matrix'], dtype=np.float32)
    R = pose_matrix[:3, :3]
    t = pose_matrix[:3, 3]

    # Convert depth to point cloud in camera frame
    points_cam = depth_to_point_cloud(depth, mask, cam_fx, cam_fy, cam_cx, cam_cy)

    print(f"Extracted {len(points_cam)} points from depth map")

    # Transform points to object's local coordinate frame
    # points_cam = R @ points_obj + t
    # => points_obj = R^T @ (points_cam - t)
    R_inv = R.T
    points_obj = (points_cam - t) @ R_inv.T

    # Subsample to target number of points
    if len(points_obj) > num_points:
        # Random sampling (you could use farthest point sampling for better coverage)
        indices = np.random.choice(len(points_obj), num_points, replace=False)
        points_obj = points_obj[indices]

    print(f"Final model has {len(points_obj)} points")

    # Center the model at origin (optional, but recommended)
    centroid = np.mean(points_obj, axis=0)
    points_obj = points_obj - centroid

    print(f"Model centroid: {centroid}")
    print(f"Model bounds: min={points_obj.min(axis=0)}, max={points_obj.max(axis=0)}")

    return points_obj


def save_model_points(points, output_path):
    """Save model points to .xyz file"""
    with open(output_path, 'w') as f:
        for point in points:
            f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
    print(f"Saved model to {output_path}")


if __name__ == '__main__':
    root_dir = "/Users/filipsulich/Library/CloudStorage/GoogleDrive-sulich.f@gmail.com/My Drive/Project3-1/DenseFusion_Blender_data"
    output_path = os.path.join(root_dir, "model_points.xyz")

    model_points = extract_model_points(root_dir, frame_idx=0, num_points=1000)
    save_model_points(model_points, output_path)
    
    print("done - model points saved")

