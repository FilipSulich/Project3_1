import torch
import torch.utils.data as data
from PIL import Image
import numpy as np
import os
import glob
import json
import torchvision.transforms as transforms
import numpy.ma as ma
import random
import OpenEXR
import Imath

CAMERA_JSON_PATH = "/Users/filipsulich/Developer/uni/Project3_1/DenseFusionModel/Project31/DenseFusion/datasets/Blender/camera_settings.json"


class PoseDataset(data.Dataset):
    """
    DenseFusion-compatible dataset for Blender-rendered synthetic data.

    Expected file structure in root directory:
        frame_0001_rgb.png      - RGB image
        frame_0001_depth.exr    - Depth map (meters)
        frame_0001_mask.png     - Object segmentation mask
        frame_0001_meta.json    - Contains pose_matrix and camera intrinsics K
        model_points.xyz        - 3D model (canonical shape)
    """

    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine):
        """
        Args:
            mode: 'train' or 'test'
            num_pt: Number of points to sample from observed cloud
            add_noise: Add noise during training for robustness
            root: Directory containing Blender data
            noise_trans: Translation noise magnitude (e.g., 0.03 = 3cm)
            refine: Use refinement network (affects num_pt_mesh)
        """
        self.root = root
        self.mode = mode
        self.num_pt = num_pt
        self.add_noise = add_noise
        self.noise_trans = noise_trans
        self.refine = refine
        self.cam_cx = 0
        self.cam_cy = 0
        self.cam_fx = 0
        self.cam_fy = 0

        
        self.rgb_files = sorted(glob.glob(os.path.join(root, "*_rgb.png")))
        if not self.rgb_files:
            raise FileNotFoundError(f"No rgb files found in {root}")

        self.length = len(self.rgb_files)
        print(f"Loaded {self.length} samples from {root}")

        self.cam_fx, self.cam_fy, self.cam_cx, self.cam_cy = self._load_camera_intrinsics()

        model_path = os.path.join(root, "model_points.xyz")
        if os.path.exists(model_path):
            self.model_points = self._load_model_points(model_path) 
        else:
            print(f"WARNING: No model_points.xyz found, using random points")
            self.model_points = np.random.randn(500, 3).astype(np.float32) * 0.1 # in case the model is not found, we use some random points

        test_img = Image.open(self.rgb_files[0])
        self.img_width, self.img_height = test_img.size
        print(f"Image dimensions: {self.img_width}x{self.img_height}")

        # Create coordinate maps for depth unprojection
        self.xmap = np.array([[j for i in range(self.img_width)] for j in range(self.img_height)])
        self.ymap = np.array([[i for i in range(self.img_width)] for j in range(self.img_height)])

        # Image transforms
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)  # Data augmentation
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet normalization
                                        std=[0.229, 0.224, 0.225])

        # Model point sampling
        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.minimum_num_pt = 50

    def _load_model_points(self, path):
        """Load canonical 3D model from .xyz file - this is the same for all frames"""
        points = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    x, y, z = map(float, line.strip().split())
                    points.append([x, y, z])
        return np.array(points, dtype=np.float32)

    def _load_camera_intrinsics(self):
        """Load camera intrinsics (fx, fy, cx, cy) from JSON"""
        with open(CAMERA_JSON_PATH, 'r') as f:
            cam_data = json.load(f)
            K = np.array(cam_data['K'], dtype=np.float32)
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
        return fx, fy, cx, cy

    def _load_depth_exr(self, path):
        """Load depth map from OpenEXR file (Blender format)"""
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

    def _get_bbox_from_mask(self, mask):
        """
        Compute bounding box from binary mask with expansion.

        Uses border_list to quantize bbox sizes (from original DenseFusion).
        This ensures consistent crop sizes for batching.
        """
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not rows.any() or not cols.any():
            return 0, mask.shape[0], 0, mask.shape[1]

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmax += 1
        cmax += 1

        # Expand bbox to quantized sizes
        r_b = rmax - rmin
        c_b = cmax - cmin
        border_list = [-1, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520, 560, 600, 640, 680]

        for tt in range(len(border_list) - 1):
            if r_b > border_list[tt] and r_b < border_list[tt + 1]:
                r_b = border_list[tt + 1]
                break
        for tt in range(len(border_list) - 1):
            if c_b > border_list[tt] and c_b < border_list[tt + 1]:
                c_b = border_list[tt + 1]
                break

        # Center the bbox
        center = [(rmin + rmax) / 2, (cmin + cmax) / 2]
        rmin = int(center[0] - r_b / 2)
        rmax = int(center[0] + r_b / 2)
        cmin = int(center[1] - c_b / 2)
        cmax = int(center[1] + c_b / 2)

        # Clamp to image bounds
        if rmin < 0:
            rmax -= rmin
            rmin = 0
        if cmin < 0:
            cmax -= cmin
            cmin = 0
        if rmax > self.img_height:
            rmin -= (rmax - self.img_height)
            rmax = self.img_height
        if cmax > self.img_width:
            cmin -= (cmax - self.img_width)
            cmax = self.img_width

        rmin = max(0, rmin)
        cmin = max(0, cmin)
        rmax = min(self.img_height, rmax)
        cmax = min(self.img_width, cmax)

        return rmin, rmax, cmin, cmax

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        """
        Returns:
            cloud: [num_pt, 3] - Observed point cloud from depth (INPUT)
            choose: [1, num_pt] - Indices into cropped image
            img: [3, H, W] - Cropped RGB around object (INPUT) - bounding box from mask
            target: [num_pt_mesh, 3] - Ground truth positions (LABEL) - default position transformed by (R,t)
            model_points: [num_pt_mesh, 3] - Canonical model shape
            idx: [1] - Object class index (0 for single object)
        """
        rgb_path = self.rgb_files[idx]
        base_name = os.path.splitext(rgb_path)[0].replace("_rgb", "")
        frame_num = base_name.split('_')[-1]

        #paths to appriopiate files
        depth_path = os.path.join(self.root, f"frame_{frame_num}_depth.exr") 
        mask_path = os.path.join(self.root, f"frame_{frame_num}_mask.png")
        meta_path = os.path.join(self.root, f"frame_{frame_num}_meta.json")

        img = Image.open(rgb_path) # load the RGB image
        depth = self._load_depth_exr(depth_path) # we load the depth map in meters (how far each pixel is from the camera)
        mask = np.array(Image.open(mask_path).convert('L')) # Convert to grayscale (the object is white on black bg)

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        pose_matrix = np.array(meta['pose_matrix'], dtype=np.float32) # Get pose matrix (world to camera transform)
        target_r = pose_matrix[:3, :3]  # Rotation matrix (from the pose matrix)
        target_t = pose_matrix[:3, 3]   # Translation vector (from the pose matrix)

        # === Create masks ===
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))  # Valid depth (non-zero)
        mask_label = ma.getmaskarray(ma.masked_equal(mask, 0))       # Background pixels from mask
        mask = ~mask_label * mask_depth  # Object pixels with valid depth

        # Check minimum points
        if len(mask.nonzero()[0]) < self.minimum_num_pt:
            cc = torch.LongTensor([0])
            return cc, cc, cc, cc, cc, cc

        # === Apply color jitter (data augmentation) ===
        if self.add_noise:
            img = self.trancolor(img)

        # === Crop to object bounding box ===
        rmin, rmax, cmin, cmax = self._get_bbox_from_mask(~mask_label)
        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))  # HWC -> CHW
        img_masked = img[:, rmin:rmax, cmin:cmax]

        # === Sample points from cropped region ===
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        if len(choose) == 0:
            cc = torch.LongTensor([0])
            return cc, cc, cc, cc, cc, cc

        # Subsample or pad to num_pt
        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        # Create observed cloud (from depth) (identical to LinemodDataset)
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

        # Back-project to 3D (pinhole camera model)
        pt2 = depth_masked  # Z
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx  # X
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy  # Y
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        # Add noise for robustness
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])
        if self.add_noise:
            cloud = np.add(cloud, add_t)

        # === Sample model points ===
        dellist = [j for j in range(0, len(self.model_points))]
        if self.refine:
            num_points_mesh = min(self.num_pt_mesh_large, len(self.model_points))
        else:
            num_points_mesh = min(self.num_pt_mesh_small, len(self.model_points))

        if len(self.model_points) > num_points_mesh:
            dellist = random.sample(dellist, len(self.model_points) - num_points_mesh)
        model_points = np.delete(self.model_points, dellist, axis=0)

        # === Create target (ground truth) ===
        # Transform model points to camera space: target = R @ model + t
        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t + add_t)
        else:
            target = np.add(target, target_t)

        # === Format outputs ===
        choose = np.array([choose])  # [1, num_pt]

        # Normalize image: uint8 [0,255] -> float [0,1] -> ImageNet normalized
        img_masked_normalized = img_masked.astype(np.float32) / 255.0

        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked_normalized)), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([0])  # Object class (0 for single-object dataset)

    def get_sym_list(self):
        """Return list of symmetric object indices (empty if no symmetry)"""
        return []

    def get_num_points_mesh(self):
        """Return number of model points"""
        if self.refine:
            return self.num_pt_mesh_large
        else:
            return self.num_pt_mesh_small
