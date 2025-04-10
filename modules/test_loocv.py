import torch
import os
import numpy as np
from scipy.stats import pearsonr
from models.load_model import load_brain_shift_model
from models.dataset_loocv import UltrasoundDataset

def evaluate_loocv(model, test_loader, device):
    """Evaluate brain shift prediction on LOOCV test patient."""
    model.eval()
    predicted_landmark_shifts = []
    ground_truth_vectors = []

    for pre, post, pre_landmarks, post_landmarks in test_loader:
        pre, post = pre.to(device), post.to(device)
        with torch.no_grad():
            flow = model(pre, post)
            mean_shift, max_shift, shift_map = compute_brain_shift(flow)

        # **Extract shifts at valid landmark positions**
        pre_landmarks = pre_landmarks.squeeze(0).cpu().numpy()
        post_landmarks = post_landmarks.squeeze(0).cpu().numpy()

        for i in range(pre_landmarks.shape[0]):  
            x, y, z = pre_landmarks[i]
            predicted_landmark_shifts.append(shift_map[0, :, int(x), int(y), int(z)])  # Extract shift vector

        # Store ground truth vectors
        ground_truth_vectors.append(post_landmarks - pre_landmarks)

    # Convert to NumPy arrays
    predicted_landmark_shifts = np.array(predicted_landmark_shifts)
    ground_truth_vectors = np.vstack(ground_truth_vectors)

    # **Compute Metrics**
    predicted_magnitudes = np.linalg.norm(predicted_landmark_shifts, axis=1)
    ground_truth_magnitudes = np.linalg.norm(ground_truth_vectors, axis=1)

    mae = np.mean(np.abs(predicted_magnitudes - ground_truth_magnitudes))
    rmse = np.sqrt(np.mean((predicted_magnitudes - ground_truth_magnitudes) ** 2))
    r_value, _ = pearsonr(predicted_magnitudes, ground_truth_magnitudes)
    mpe = np.mean((predicted_magnitudes - ground_truth_magnitudes) / ground_truth_magnitudes) * 100

    # **Compute Directional Errors**
    gt_norm = ground_truth_vectors / (np.linalg.norm(ground_truth_vectors, axis=1, keepdims=True) + 1e-6)
    pred_norm = predicted_landmark_shifts / (np.linalg.norm(predicted_landmark_shifts, axis=1, keepdims=True) + 1e-6)
    cosine_sim = np.sum(gt_norm * pred_norm, axis=1)
    mean_cosine_sim = np.mean(cosine_sim)
    angular_error = np.arccos(np.clip(cosine_sim, -1.0, 1.0)) * (180 / np.pi)
    mean_angular_error = np.mean(angular_error)

    # **Return Results for LOOCV**
    return {
        "MAE": mae,
        "RMSE": rmse,
        "R": r_value,
        "MPE": mpe,
        "Cosine": mean_cosine_sim,
        "Angular Error": mean_angular_error
    }


def compute_brain_shift(flow):
    """
    Computes the brain shift **without collapsing the 3D displacement vectors**.
    Args:
        flow (torch.Tensor): Shape (1, 3, 128, 128, 128), predicted shift vectors.
    Returns:
        mean_shift (float), max_shift (float), full_shift_map (numpy array).
    """
    # **Compute magnitude correctly but keep all three components**
    shift_magnitude = torch.norm(flow, dim=1, keepdim=True)  # (1, 1, 128, 128, 128) instead of (1,128,128,128)
    
    return shift_magnitude.mean().item(), shift_magnitude.max().item(), flow.cpu().detach().numpy()  # ⬅️ Keep full shape!



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
