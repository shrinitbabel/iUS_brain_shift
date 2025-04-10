import torch
import os
import numpy as np
from torch.utils.data import DataLoader
from unet3d_new import UNet3D
from dataset_loocv import UltrasoundDataset

# Path to dataset
DATASET_DIR = "Dataset"  # Now merged dataset containing all patients 02-14

# Get list of all patient folders (02 to 14)
all_patients = sorted([p for p in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, p))])

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# LOOCV Results Storage
loocv_results = []
import torch.nn.functional as F

def sample_flow_at_landmarks(flow, pre_landmarks):
    """Interpolates flow field at given landmark positions"""
    batch_size, _, d, h, w = flow.shape  # Flow field shape
    num_points = pre_landmarks.shape[1]

    # Normalize landmarks to [-1, 1] range for grid_sample
    norm_landmarks = pre_landmarks.clone()
    norm_landmarks[..., 0] = (norm_landmarks[..., 0] / (w - 1)) * 2 - 1
    norm_landmarks[..., 1] = (norm_landmarks[..., 1] / (h - 1)) * 2 - 1
    norm_landmarks[..., 2] = (norm_landmarks[..., 2] / (d - 1)) * 2 - 1

    # Reshape to match grid_sample format
    norm_landmarks = norm_landmarks.unsqueeze(1).unsqueeze(1)  # Add two singleton dimensions

    # Sample flow at landmark locations
    sampled_flow = F.grid_sample(flow, norm_landmarks, align_corners=True, mode="bilinear")

    # Remove extra dimensions
    return sampled_flow.view(batch_size, num_points, 3)

# 🔹 **Loss Function (Includes Landmark Supervision)**
def loss_function(flow, pre, post, pre_landmarks, post_landmarks):
    """Computes image loss, smoothness loss, landmark loss, and angular loss"""
    warped_pre = F.grid_sample(pre, flow.permute(0, 2, 3, 4, 1), align_corners=True)
    l2_loss = F.mse_loss(warped_pre, post)  

    # 🔹 Smoothness Regularization
    smoothness_loss = torch.mean(torch.abs(flow[:, :, :, :-1, :-1] - flow[:, :, :, 1:, 1:])) 

    # 🔹 Landmark Loss
    pre_landmarks, post_landmarks = pre_landmarks.to(flow.device), post_landmarks.to(flow.device)
    sampled_flow = sample_flow_at_landmarks(flow, pre_landmarks)
    landmark_loss = F.mse_loss(pre_landmarks + sampled_flow, post_landmarks)

    # 🔹 Angular Loss
    predicted_vectors = sampled_flow
    gt_vectors = post_landmarks - pre_landmarks

    pred_norm = torch.norm(predicted_vectors, dim=-1, keepdim=True) + 1e-6
    gt_norm = torch.norm(gt_vectors, dim=-1, keepdim=True) + 1e-6
    predicted_vectors = predicted_vectors / pred_norm
    gt_vectors = gt_vectors / gt_norm

    cosine_sim = torch.sum(predicted_vectors * gt_vectors, dim=-1)  
    cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)  
    angular_loss = torch.mean(torch.acos(cosine_sim))  

    # 🔹 Final Loss (Weighted)
    total_loss = (
        l2_loss 
        + 0.01 * smoothness_loss  
        + 0.05 * landmark_loss  
        + 0.1 * angular_loss    
    )

    return total_loss


for test_patient in all_patients:
    print(f"\n🔄 **LOOCV - Leaving Out Patient {test_patient} for Testing**")

    # **Split dataset:** Train on all except the test patient
    train_patients = [p for p in all_patients if p != test_patient]

    # **Load Training Data**
    train_dataset = UltrasoundDataset(dataset_dir=DATASET_DIR, patient_list=train_patients)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

    # **Load Model**
    model = UNet3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    # **Train Model**
    num_epochs = 10
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for pre, post, pre_landmarks, post_landmarks in train_loader:
            pre, post = pre.to(device), post.to(device)
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            flow = model(pre, post)
            loss = loss_function(flow, pre, post, pre_landmarks, post_landmarks)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step(epoch_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

    # **Save LOOCV Model for this Fold**
    model_path = f"models/brain_shift_model_LOOCV_{test_patient}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✅ Model for LOOCV Fold (Test: {test_patient}) saved at {model_path}")


# **Compute and Print Final LOOCV Metrics**
mean_mae = np.mean([res['MAE'] for res in loocv_results])
mean_rmse = np.mean([res['RMSE'] for res in loocv_results])
mean_r = np.mean([res['R'] for res in loocv_results])
mean_mpe = np.mean([res['MPE'] for res in loocv_results])
mean_cosine = np.mean([res['Cosine'] for res in loocv_results])
mean_angle = np.mean([res['Angular Error'] for res in loocv_results])

print("\n📊 **Final LOOCV Evaluation Metrics (Averaged Over All Patients):**")
print(f"🔹 MAE: {mean_mae:.3f} mm")
print(f"🔹 RMSE: {mean_rmse:.3f} mm")
print(f"🔹 Pearson R: {mean_r:.3f}")
print(f"🔹 MPE: {mean_mpe:.2f}%")
print(f"🔹 Cosine Similarity: {mean_cosine:.3f}")
print(f"🔹 Angular Error: {mean_angle:.2f}°")
