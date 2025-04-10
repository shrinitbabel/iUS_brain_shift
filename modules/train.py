# modules/train.py

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os

from unet3d import UNet3D
from load_model import load_brain_shift_model
from dataset import UltrasoundDataset

DATASET_DIR = "Training Dataset"
# load dataset
dataset = UltrasoundDataset(DATASET_DIR)
train_loader = DataLoader(dataset, batch_size=1, shuffle=True)

# training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = UNet3D().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
ALL_PATIENTS = sorted([f"{int(d):02}" for d in os.listdir("Dataset") if os.path.isdir(os.path.join("Dataset", d))])
TRAINED_PATIENTS = sorted([f"{int(os.path.basename(p)):02}" for p in dataset.patient_dirs])
TEST_PATIENTS = [p for p in ALL_PATIENTS if p not in TRAINED_PATIENTS]

# fallback case
test_patient_id = TEST_PATIENTS[0] if TEST_PATIENTS else "unknown"
print(f"In this set, the test patient is: {test_patient_id}")


def sample_flow_at_landmarks(flow, pre_landmarks):
    """Interpolates flow field at given landmark positions"""
    batch_size, _, d, h, w = flow.shape  
    num_points = pre_landmarks.shape[1]

    # Normalize landmarks to [-1, 1] range for grid_sample
    norm_landmarks = pre_landmarks.clone()
    norm_landmarks[..., 0] = (norm_landmarks[..., 0] / (w - 1)) * 2 - 1
    norm_landmarks[..., 1] = (norm_landmarks[..., 1] / (h - 1)) * 2 - 1
    norm_landmarks[..., 2] = (norm_landmarks[..., 2] / (d - 1)) * 2 - 1

    # Reshape to match grid_sample format
    norm_landmarks = norm_landmarks.unsqueeze(1).unsqueeze(1)  

    # Sample flow at landmark locations
    sampled_flow = F.grid_sample(flow, norm_landmarks, align_corners=True, mode="bilinear")

    # Remove extra dimensions
    return sampled_flow.view(batch_size, num_points, 3)


# Loss Function 
def loss_function(flow, pre, post, pre_landmarks, post_landmarks):
    # Warp pre-op image using predicted flow field
    warped_pre = F.grid_sample(pre, flow.permute(0, 2, 3, 4, 1), align_corners=True)
    l2_loss = F.mse_loss(warped_pre, post)  # L2 loss on image alignment

    # 🔹 Smoothness Regularization (TV Loss)
    smoothness_loss = torch.mean(torch.abs(flow[:, :, :, :-1, :-1] - flow[:, :, :, 1:, 1:])) 

    # 🔹 Landmark Loss (sample flow at landmark locations)
    pre_landmarks = pre_landmarks.to(device)
    post_landmarks = post_landmarks.to(device)
    sampled_flow = sample_flow_at_landmarks(flow, pre_landmarks)
    landmark_loss = F.mse_loss(pre_landmarks + sampled_flow, post_landmarks)

    # 🔹 Angular Loss (Replaces Cosine Similarity Loss)
    predicted_vectors = sampled_flow
    gt_vectors = post_landmarks - pre_landmarks

    # Normalize vectors (avoid NaNs)
    pred_norm = torch.norm(predicted_vectors, dim=-1, keepdim=True) + 1e-6
    gt_norm = torch.norm(gt_vectors, dim=-1, keepdim=True) + 1e-6

    predicted_vectors = predicted_vectors / pred_norm
    gt_vectors = gt_vectors / gt_norm

    # Compute angular difference (cosine similarity based)
    cosine_sim = torch.sum(predicted_vectors * gt_vectors, dim=-1)  
    cosine_sim = torch.clamp(cosine_sim, -1.0, 1.0)  
    angular_loss = torch.mean(torch.acos(cosine_sim))  

    # Final Loss (adjust weights for stability)
    total_loss = (
        l2_loss 
        + 0.01 * smoothness_loss 
        + 0.05 * landmark_loss  # Reduce landmark loss weight
        + 0.1 * angular_loss    # Increase directional alignment loss
    )

    return total_loss



num_epochs = 10  # change epochs as needed
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

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
    
    scheduler.step(epoch_loss)  # change learning rate as needed
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}")

# Save Model
model_save_path = f"models/brain_shift_model_{test_patient_id}.pt"
torch.save(model.state_dict(), model_save_path)
print(f"Model training complete! ✅ Saved to {model_save_path}")
print("Training set patients:", dataset.patient_dirs)