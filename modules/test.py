# modules/test.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from unet3d_new import UNet3D
from dataset import UltrasoundDataset
from load_model import load_brain_shift_model

def compute_brain_shift(flow):
    """
    Computes the brain shift **without collapsing the 3D displacement vectors**.
    Args:
        flow (torch.Tensor): Shape (1, 3, 128, 128, 128), predicted shift vectors.
    Returns:
        mean_shift (float), max_shift (float), full_shift_map (numpy array).
    """
    # Compute magnitude correctly but keep all three components
    shift_magnitude = torch.norm(flow, dim=1, keepdim=True)  # (1, 1, 128, 128, 128) instead of (1,128,128,128)
    
    return shift_magnitude.mean().item(), shift_magnitude.max().item(), flow.cpu().detach().numpy()  



def world_to_voxel(landmarks, origin, spacing):
    """
    Convert world coordinates (mm) to voxel coordinates (indices).
    Args:
        landmarks (numpy array): (N, 3) array of world coordinates.
        origin (numpy array): (3,) array of the image's origin in world coordinates.
        spacing (tuple): Voxel spacing in each dimension (mm per voxel).
    Returns:
        numpy array of voxel indices.
    """
    return np.round((landmarks - origin) / spacing).astype(int)
