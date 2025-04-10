# modules/post_process.py

import os
import sys
import torch
import torch.nn.functional as F
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from unet3d import UNet3D  

def load_brain_shift_model(model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
    model = UNet3D()  
    model.load_state_dict(torch.load(model_path, map_location=device))  
    model.to(device)
    model.eval() 
    return model

def warp_image(image, flow):
    d, h, w = image.shape[-3:]
    grid = torch.meshgrid(
        torch.linspace(-1, 1, d),
        torch.linspace(-1, 1, h),
        torch.linspace(-1, 1, w),
        indexing="ij"
    )
    grid = torch.stack(grid, dim=-1).to(image.device)  # Shape: (D, H, W, 3)

    # **Scale down the flow field even more**
    warped_grid = grid + flow.squeeze(0).permute(1, 2, 3, 0) * 0.0001  

    # **Ensure warped grid is normalized correctly**
    warped_grid = torch.clamp(warped_grid, -1, 1)  # Clip values within [-1,1] to avoid out-of-bounds warping

    # Warp the image
    warped_image = F.grid_sample(image, warped_grid.unsqueeze(0), align_corners=True, mode="bilinear")
    
    return warped_image.squeeze().cpu().numpy()
