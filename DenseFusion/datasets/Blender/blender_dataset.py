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


class PoseDataset(data.Dataset):
    def __init__(self, mode, num_pt, add_noise, root, noise_trans, refine):
        """
        Blender dataset compatible with DenseFusion training.

        Args:
            mode: 'train' or 'test'
            num_pt: Number of points to sample from point cloud
            add_noise: Whether to add noise during training
            root: Root directory containing Blender output
            noise_trans: Amount of translation noise to add
            refine: Whether to use refinement (affects model_points count)
        """
        self.root = root
        self.mode = mode
        self.num_pt = num_pt
        self.add_noise = add_noise
        self.noise_trans = noise_trans
        self.refine = refine

        # Find all RGB images
        self.rgb_files = sorted(glob.glob(os.path.join(root, "*_rgb.png")))
        if not self.rgb_files:
            raise FileNotFoundError(f"No '*_rgb.png' files found in {root}")

        self.length = len(self.rgb_files)
        print(f"Loaded {self.length} samples from {root}")

        # Load camera intrinsics
        camera_json = os.path.join(root, "/Users/filipsulich/Developer/uni/Project3_1/DenseFusionModel/Project31/DenseFusion/datasets/Blender/camera_settings.json")
        with open(camera_json, 'r') as f:
            cam_data = json.load(f)
            K = np.array(cam_data['K'], dtype=np.float32)
            self.cam_fx, self.cam_fy = K[0, 0], K[1, 1]
            self.cam_cx, self.cam_cy = K[0, 2], K[1, 2]

        # Load 3D model points (you need to provide this file!)
        # For now, create dummy model points - REPLACE THIS with actual .ply or .xyz file
        model_path = os.path.join(root, "model_points.xyz")
        if os.path.exists(model_path):
            self.model_points = self.load_model_points(model_path)
        else:
            print(f"Warning: No model_points.xyz found, using dummy points")
            self.model_points = np.random.randn(500, 3).astype(np.float32) * 0.1

        # Coordinate maps for depth->pointcloud conversion
        # Get image dimensions from first file
        test_img = Image.open(self.rgb_files[0])
        self.img_width, self.img_height = test_img.size
        print(f"Image dimensions: {self.img_width}x{self.img_height}")

        self.xmap = np.array([[j for i in range(self.img_width)] for j in range(self.img_height)])
        self.ymap = np.array([[i for i in range(self.img_width)] for j in range(self.img_height)])

        # Transforms
        self.trancolor = transforms.ColorJitter(0.2, 0.2, 0.2, 0.05)
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])

        self.num_pt_mesh_small = 500
        self.num_pt_mesh_large = 2600
        self.minimum_num_pt = 50

    def load_model_points(self, path):
        """Load 3D model points from .xyz file"""
        points = []
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    x, y, z = map(float, line.strip().split())
                    points.append([x, y, z])
        return np.array(points, dtype=np.float32)

    def load_depth_exr(self, path):
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

    def get_bbox(self, mask):
        """Get bounding box from mask"""
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not rows.any() or not cols.any():
            return 0, mask.shape[0], 0, mask.shape[1]

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        rmax += 1
        cmax += 1

        # Expand bbox slightly
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

        center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
        rmin = center[0] - int(r_b / 2)
        rmax = center[0] + int(r_b / 2)
        cmin = center[1] - int(c_b / 2)
        cmax = center[1] + int(c_b / 2)

        # Clamp to image bounds (use actual image dimensions)
        img_height, img_width = mask.shape
        if rmin < 0:
            rmax -= rmin
            rmin = 0
        if cmin < 0:
            cmax -= cmin
            cmin = 0
        if rmax > img_height:
            rmin -= (rmax - img_height)
            rmax = img_height
        if cmax > img_width:
            cmin -= (cmax - img_width)
            cmax = img_width

        # Final safety clamp
        rmin = max(0, rmin)
        cmin = max(0, cmin)
        rmax = min(img_height, rmax)
        cmax = min(img_width, cmax)

        return rmin, rmax, cmin, cmax

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Get file paths
        rgb_path = self.rgb_files[idx]
        base_name = os.path.splitext(rgb_path)[0].replace("_rgb", "")
        frame_num = base_name.split('_')[-1]

        depth_path = os.path.join(self.root, f"frame_{frame_num}_depth.exr")
        mask_path = os.path.join(self.root, f"frame_{frame_num}_mask.png")
        meta_path = os.path.join(self.root, f"frame_{frame_num}_meta.json")

        # Load data
        img = Image.open(rgb_path)
        depth = self.load_depth_exr(depth_path)
        mask = np.array(Image.open(mask_path).convert('L'))

        with open(meta_path, 'r') as f:
            meta = json.load(f)

        # Get pose (4x4 matrix -> extract R and t)
        pose_matrix = np.array(meta['pose_matrix'], dtype=np.float32)
        target_r = pose_matrix[:3, :3]  # Rotation
        target_t = pose_matrix[:3, 3]   # Translation

        # Create masks
        mask_depth = ma.getmaskarray(ma.masked_not_equal(depth, 0))
        mask_label = ma.getmaskarray(ma.masked_equal(mask, 0))  # True where mask=0 (background)
        mask = ~mask_label * mask_depth  # Invert to get True where object is

        # Check minimum points
        if len(mask.nonzero()[0]) < self.minimum_num_pt:
            # Return dummy data if mask too small
            cc = torch.LongTensor([0])
            return cc, cc, cc, cc, cc, cc

        # Apply color jitter if training with noise
        if self.add_noise:
            img = self.trancolor(img)

        # Get bounding box and crop (use ~mask_label since mask_label is inverted)
        rmin, rmax, cmin, cmax = self.get_bbox(~mask_label)
        img = np.transpose(np.array(img)[:, :, :3], (2, 0, 1))
        img_masked = img[:, rmin:rmax, cmin:cmax]

        # Get choose indices (flat indices into cropped region)
        choose = mask[rmin:rmax, cmin:cmax].flatten().nonzero()[0]

        if len(choose) == 0:
            # Return dummy data if no valid pixels after cropping
            cc = torch.LongTensor([0])
            return cc, cc, cc, cc, cc, cc

        if len(choose) > self.num_pt:
            c_mask = np.zeros(len(choose), dtype=int)
            c_mask[:self.num_pt] = 1
            np.random.shuffle(c_mask)
            choose = choose[c_mask.nonzero()]
        else:
            choose = np.pad(choose, (0, self.num_pt - len(choose)), 'wrap')

        # Convert depth to point cloud using chosen pixels
        depth_masked = depth[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        xmap_masked = self.xmap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)
        ymap_masked = self.ymap[rmin:rmax, cmin:cmax].flatten()[choose][:, np.newaxis].astype(np.float32)

        # Depth is in meters from Blender, convert to point cloud
        pt2 = depth_masked
        pt0 = (ymap_masked - self.cam_cx) * pt2 / self.cam_fx
        pt1 = (xmap_masked - self.cam_cy) * pt2 / self.cam_fy
        cloud = np.concatenate((pt0, pt1, pt2), axis=1)

        # Add noise if training
        add_t = np.array([random.uniform(-self.noise_trans, self.noise_trans) for i in range(3)])
        if self.add_noise:
            cloud = np.add(cloud, add_t)

        # Sample model points
        dellist = [j for j in range(0, len(self.model_points))]
        if self.refine:
            num_points_mesh = min(self.num_pt_mesh_large, len(self.model_points))
        else:
            num_points_mesh = min(self.num_pt_mesh_small, len(self.model_points))

        if len(self.model_points) > num_points_mesh:
            dellist = random.sample(dellist, len(self.model_points) - num_points_mesh)
        model_points = np.delete(self.model_points, dellist, axis=0)

        # Transform model points to get target
        target = np.dot(model_points, target_r.T)
        if self.add_noise:
            target = np.add(target, target_t + add_t)
        else:
            target = np.add(target, target_t)

        # Format choose as [1, num_pt]
        choose = np.array([choose])

        # Return as tuple matching YCB/LINEMOD format
        return torch.from_numpy(cloud.astype(np.float32)), \
               torch.LongTensor(choose.astype(np.int32)), \
               self.norm(torch.from_numpy(img_masked.astype(np.float32))), \
               torch.from_numpy(target.astype(np.float32)), \
               torch.from_numpy(model_points.astype(np.float32)), \
               torch.LongTensor([0])  # Object class index (0 since single object)

    def get_sym_list(self):
        """Return list of symmetric object indices"""
        return []  # Add indices if your object has symmetry

    def get_num_points_mesh(self):
        """Return number of model points used"""
        return self.num_pt_mesh_large if self.refine else self.num_pt_mesh_small


if __name__ == '__main__':
    # Test the dataset
    root = "/Users/filipsulich/Library/CloudStorage/GoogleDrive-sulich.f@gmail.com/My Drive/Project3-1/DenseFusion_Blender_data"

    dataset = PoseDataset(mode='train', num_pt=1000, add_noise=False,
                         root=root, noise_trans=0.03, refine=False)

    print(f"Dataset length: {len(dataset)}")

    if len(dataset) > 0:
        sample = dataset[0]
        print("\nSample 0 output:")
        print("  cloud:", sample[0].shape, sample[0].dtype)
        print("  choose:", sample[1].shape, sample[1].dtype)
        print("  img:", sample[2].shape, sample[2].dtype)
        print("  target:", sample[3].shape, sample[3].dtype)
        print("  model_points:", sample[4].shape, sample[4].dtype)
        print("  idx:", sample[5].shape, sample[5].dtype)
