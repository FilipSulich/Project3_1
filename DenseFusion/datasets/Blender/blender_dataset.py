import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np
import os
import glob
from PIL import Image
import json
import imageio 

def depth_to_point_cloud(depth_map, mask, K_matrix):
    """
    Converts a depth map and mask into a 3D point cloud.

    Args:
        depth_map (np.array): HxW depth map (float, in meters).
        mask (np.array): HxW boolean mask (True where object is present).
        intrinsics_matrix (np.array): 3x3 camera intrinsics K.

    Returns:
        np.array: Nx3 point cloud (float, in meters), N = number of valid points.
    """
    fx, fy = K_matrix[0, 0], K_matrix[1, 1]
    cx, cy = K_matrix[0, 2], K_matrix[1, 2] 

    # Get pixel coordinates of masked points
    row, col = np.where(mask) # v = row (y), u = column (x)

    # Get depth values for these pixels
    z = depth_map[row, col]

    # Filter out invalid depth values (optional, depends on your EXR)
    valid_indices = z > 0
    row, col, z = row[valid_indices], col[valid_indices], z[valid_indices]

    # Project pixels to 3D space (pinhole camera model)
    x = (col - cx) * z / fx
    y = (row - cy) * z / fy

    # Stack coordinates into Nx3 point cloud
    points = np.vstack((x, y, z)).T
    print(f"Generated point cloud with {points.shape[0]} points.")
    return points.astype(np.float32)


class BlenderDataset(Dataset):
    def __init__(self, root_dir, img_size=224, num_points=1000, mode='train'):
        """
        Args:
            root_dir (string): Directory with all the Blender output files.
            img_size (int): Size to resize RGB images to (must match DenseFusion input).
            num_points (int): Number of points to sample for the point cloud.
            mode (string): 'train' or 'test' (or 'val') - could be used later for splits.
        """
        self.root_dir = root_dir
        self.img_size = img_size
        self.num_points = num_points
        self.mode = mode

        # Find all RGB images, assume other files have corresponding names
        self.rgb_files = sorted(glob.glob(os.path.join(root_dir, "*_rgb.png")))
        
        if not self.rgb_files:
             raise FileNotFoundError(f"No '*_rgb.png' files found in {root_dir}")
        print(f"Found {len(self.rgb_files)} samples in {root_dir}")

        
        try:
            with open(os.path.join(root_dir, "camera_settings.json"), 'r') as f:
                self.intrinsics_dict = json.load(f)
                self.intrinsics = np.array(self.intrinsics_dict['K']).astype(np.float32)
        except FileNotFoundError:
            raise FileNotFoundError("no camera settings json file")

        if mode == 'train':
            self.rgb_transform = T.Compose([
                T.ColorJitter(0.2, 0.2, 0.2, 0.05), 
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])
        else:
            self.rgb_transform = T.Compose([
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
            ])

        self.mask_transform = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.rgb_files)

    def __getitem__(self, idx):
        rgb_path = self.rgb_files[idx]
        base_name = rgb_path.replace("_rgb.png", "")
        frame_num_str = base_name.split('_')[-1] # frame_####_png -> we get the #### part

        depth_path = os.path.join(self.root_dir, f"frame_{frame_num_str}_depth.exr")
        mask_path = os.path.join(self.root_dir, f"frame_{frame_num_str}_mask.png")
        meta_path = os.path.join(self.root_dir, f"frame_{frame_num_str}_meta.json")

        try:
            rgb_image = Image.open(rgb_path).convert("RGB").resize((self.img_size, self.img_size))
            # Load depth using imageio for EXR format
            depth_map = imageio.v2.imread(depth_path)
            # Ensure depth is single channel if it has multiple identical ones
            if depth_map.ndim == 3:
                depth_map = depth_map[:, :, 0]

            mask_image = Image.open(mask_path).convert("L").resize((self.img_size, self.img_size), Image.NEAREST)

            with open(meta_path, 'r') as f:
                meta = json.load(f)
            pose = np.array(meta['pose_matrix']).astype(np.float32)

        except FileNotFoundError as e:    
            raise FileNotFoundError(f"Missing file for sample {idx}: {e}")
        except Exception as e:
             raise RuntimeError(f"Error processing sample {idx}")


        # --- Process Data ---

        # 1. Apply transforms to RGB
        rgb_tensor = self.rgb_transform(rgb_image)

        # 2. Convert Mask to tensor and boolean numpy array
        mask_tensor = self.mask_transform(mask_image) # [1, H, W], values 0.0 to 1.0
        mask_np_bool = mask_tensor.squeeze(0).numpy() > 0.5 # [H, W], boolean

        # 3. Convert Depth to Point Cloud using correct size mask
        # We need the depth map at its original resolution for accurate point cloud
        original_mask_pil = Image.open(mask_path).convert("L")
        original_mask_np_bool = np.array(original_mask_pil) > 0 # Original size, boolean
        point_cloud = depth_to_point_cloud(depth_map, original_mask_np_bool, self.intrinsics)

        # 4. Sample or Pad Point Cloud to fixed number (self.num_points)
        if len(point_cloud) == 0:
             #print(f"Warning: No valid points found for index {idx}, returning zero cloud.")
             point_cloud_sampled = np.zeros((self.num_points, 3), dtype=np.float32)
        elif len(point_cloud) >= self.num_points:
            # Sample points if too many (use farthest point sampling or random choice)
            choice = np.random.choice(len(point_cloud), self.num_points, replace=False)
            point_cloud_sampled = point_cloud[choice, :]
        else:
            # Pad by repeating existing points if too few (better than zeros)
            #print(f"Warning: Only {len(point_cloud)} points found for index {idx}, padding.")
            indices = np.random.choice(len(point_cloud), self.num_points - len(point_cloud))
            padding = point_cloud[indices, :]
            point_cloud_sampled = np.concatenate([point_cloud, padding], axis=0)

        point_cloud_tensor = torch.from_numpy(point_cloud_sampled)

        # 5. Convert Pose and Intrinsics to Tensors
        pose_tensor = torch.from_numpy(pose)
        intrinsics_tensor = torch.from_numpy(self.intrinsics)

        # 6. Get mask indices (where object pixels are in the *resized* image)
        mask_indices = torch.argwhere(mask_tensor.squeeze(0) > 0.5).long() # Shape [N_pixels, 2] (row, col or y, x)

        # Ensure we have at least one index if mask is not empty
        if len(mask_indices) == 0 and mask_np_bool.any():
             print(f"Warning: Mask indices empty despite non-empty mask for index {idx}. Check mask processing.")
             # Add a dummy index? Or handle differently? For now, let's keep it empty.

        # DenseFusion might expect a specific object index (usually 0 or 1 if only one object type)
        obj_idx = torch.tensor([1], dtype=torch.int32) # Assuming object index 1

        # Return dictionary (keys might need to match train.py exactly)
        return {
            'rgb': rgb_tensor,                # [3, H, W]
            'cloud': point_cloud_tensor,      # [num_points, 3] - Often called 'cloud' or 'pts'
            'choose': mask_indices,           # [N_pixels, 2] - Often called 'choose' or 'idx'
            'cam_ K': intrinsics_tensor,      # [3, 3] - Check key name in train.py
            'gt_RTs': pose_tensor.unsqueeze(0),# [1, 4, 4] - Often expects batch dim
            'cls_indexes': obj_idx,           # [1] - Often called 'cls_indexes' or 'cat_id'
            # 'meta': {'item_path': rgb_path} # Optional metadata for debugging
        }

# --- Example Usage (Place in a separate test script or under if __name__ == '__main__':) ---
# import sys
# sys.path.append('.') # If BlenderDataset is in the main directory
# from blender_dataset import BlenderDataset
#
# if __name__ == '__main__':
#     DATASET_ROOT = "/Users/filipsulich/Library/CloudStorage/GoogleDrive-sulich.f@gmail.com/My Drive/Project3-1/DenseFusion_Blender_data" # Your path
#     try:
#         dataset = BlenderDataset(root_dir=DATASET_ROOT, img_size=128, num_points=500) # Use smaller size for quick test
#         print(f"Dataset size: {len(dataset)}")
#
#         if len(dataset) > 0:
#             sample = dataset[0] # Try loading the first sample
#             if sample:
#                 print("\nSample loaded successfully:")
#                 for key, value in sample.items():
#                     if isinstance(value, torch.Tensor):
#                         print(f"  {key}: Tensor shape {value.shape}, dtype {value.dtype}")
#                     else:
#                         print(f"  {key}: Type {type(value)}")
#             else:
#                 print("Failed to load sample 0.")
#         else:
#             print("Dataset is empty or path is incorrect.")
#
#     except Exception as e:
#          print(f"\nAn error occurred during dataset testing: {e}")
#          import traceback
#          traceback.print_exc()