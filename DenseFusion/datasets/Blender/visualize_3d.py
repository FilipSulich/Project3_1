import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from blender_dataset import PoseDataset

root = "/Users/filipsulich/Library/CloudStorage/GoogleDrive-sulich.f@gmail.com/My Drive/Project3-1/DenseFusion_Blender_data"

dataset = PoseDataset(mode='train', num_pt=1000, add_noise=False, root=root, noise_trans=0.03, refine=False)

# Get sample
cloud, choose, img, target, model_points, obj_idx = dataset[1]

# Convert to numpy
cloud_np = cloud.numpy()
target_np = target.numpy()
model_points_np = model_points.numpy()

# Denormalize image for visualization
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
img_denorm = img * std + mean
img_denorm = torch.clamp(img_denorm, 0, 1)
img_np = img_denorm.permute(1, 2, 0).numpy()

print(f"Cloud shape: {cloud_np.shape}")
print(f"Cloud range: x=[{cloud_np[:, 0].min():.3f}, {cloud_np[:, 0].max():.3f}], "
      f"y=[{cloud_np[:, 1].min():.3f}, {cloud_np[:, 1].max():.3f}], "
      f"z=[{cloud_np[:, 2].min():.3f}, {cloud_np[:, 2].max():.3f}]")
print(f"Target shape: {target_np.shape}")
print(f"Target range: x=[{target_np[:, 0].min():.3f}, {target_np[:, 0].max():.3f}], "
      f"y=[{target_np[:, 1].min():.3f}, {target_np[:, 1].max():.3f}], "
      f"z=[{target_np[:, 2].min():.3f}, {target_np[:, 2].max():.3f}]")
print(f"Model points shape: {model_points_np.shape}")

# Create figure with subplots
fig = plt.figure(figsize=(20, 10))

# 1. RGB image
ax1 = fig.add_subplot(2, 4, 1)
ax1.imshow(img_np)
ax1.set_title('Cropped RGB')
ax1.axis('off')

# 2. Observed cloud (camera frame) - XYZ
ax2 = fig.add_subplot(2, 4, 2, projection='3d')
ax2.scatter(cloud_np[:, 0], cloud_np[:, 1], cloud_np[:, 2], c='blue', s=1, alpha=0.5)
ax2.set_xlabel('X')
ax2.set_ylabel('Y')
ax2.set_zlabel('Z')
ax2.set_title(f'Observed Cloud\n(camera frame, n={len(cloud_np)})')
ax2.view_init(elev=20, azim=45)

# 3. Observed cloud - XY view (top-down)
ax3 = fig.add_subplot(2, 4, 3, projection='3d')
ax3.scatter(cloud_np[:, 0], cloud_np[:, 1], cloud_np[:, 2], c='blue', s=1, alpha=0.5)
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Z')
ax3.set_title('Observed Cloud\n(top view)')
ax3.view_init(elev=90, azim=-90)

# 4. Observed cloud - XZ view (side)
ax4 = fig.add_subplot(2, 4, 4, projection='3d')
ax4.scatter(cloud_np[:, 0], cloud_np[:, 1], cloud_np[:, 2], c='blue', s=1, alpha=0.5)
ax4.set_xlabel('X')
ax4.set_ylabel('Y')
ax4.set_zlabel('Z')
ax4.set_title('Observed Cloud\n(side view)')
ax4.view_init(elev=0, azim=-90)

# 5. Target (ground truth transformed model)
ax5 = fig.add_subplot(2, 4, 5, projection='3d')
ax5.scatter(target_np[:, 0], target_np[:, 1], target_np[:, 2], c='red', s=1, alpha=0.5)
ax5.set_xlabel('X')
ax5.set_ylabel('Y')
ax5.set_zlabel('Z')
ax5.set_title(f'Target (GT transformed)\n(camera frame, n={len(target_np)})')
ax5.view_init(elev=20, azim=45)

# 6. Target - top view
ax6 = fig.add_subplot(2, 4, 6, projection='3d')
ax6.scatter(target_np[:, 0], target_np[:, 1], target_np[:, 2], c='red', s=1, alpha=0.5)
ax6.set_xlabel('X')
ax6.set_ylabel('Y')
ax6.set_zlabel('Z')
ax6.set_title('Target\n(top view)')
ax6.view_init(elev=90, azim=-90)

# 7. Original model points (local frame)
ax7 = fig.add_subplot(2, 4, 7, projection='3d')
ax7.scatter(model_points_np[:, 0], model_points_np[:, 1], model_points_np[:, 2], c='green', s=1, alpha=0.5)
ax7.set_xlabel('X')
ax7.set_ylabel('Y')
ax7.set_zlabel('Z')
ax7.set_title(f'Model Points (local frame)\n(centered, n={len(model_points_np)})')
ax7.view_init(elev=20, azim=45)

# 8. Overlay: Observed vs Target
ax8 = fig.add_subplot(2, 4, 8, projection='3d')
ax8.scatter(cloud_np[:, 0], cloud_np[:, 1], cloud_np[:, 2], c='blue', s=1, alpha=0.3, label='Observed')
ax8.scatter(target_np[:, 0], target_np[:, 1], target_np[:, 2], c='red', s=1, alpha=0.3, label='Target')
ax8.set_xlabel('X')
ax8.set_ylabel('Y')
ax8.set_zlabel('Z')
ax8.set_title('Overlay: Observed vs Target\n(should align)')
ax8.legend()
ax8.view_init(elev=20, azim=45)

plt.tight_layout()
plt.savefig('visualize_3d.png', dpi=150, bbox_inches='tight')
print("\nSaved to visualize_3d.png")
