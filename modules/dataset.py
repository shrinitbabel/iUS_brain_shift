# modules/dataset.py

import numpy as np
import os
import nibabel as nib
import torch
from torch.utils.data import Dataset, DataLoader
import scipy.ndimage

# Define dataset path
TARGET_SHAPE = (128, 128, 128)

def load_mnc(file_path):
    """Load MINC .mnc file and return only the numpy array."""
    img = nib.load(file_path)
    data = img.get_fdata()
    return data 

def normalize(volume):
    """Normalize the volume between 0 and 1."""
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    return volume.astype(np.float32)

def resize_volume(volume, target_shape=TARGET_SHAPE):
    """Resize 3D volume to a fixed shape."""
    factors = [target_shape[i] / volume.shape[i] for i in range(3)]
    return scipy.ndimage.zoom(volume, factors, order=1)

# Parse Tag File (Landmarks)
def parse_tag_file(tag_file_path):
    pre_landmarks, post_landmarks = [], []
    with open(tag_file_path, 'r') as file:
        lines = file.readlines()
        points_started = False
        for line in lines:
            line = line.strip()
            if line.startswith("Points"):
                points_started = True
                continue
            if points_started and ";" not in line and line:
                try:
                    coords = [float(x) for x in line.split() if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
                    if len(coords) == 6:
                        pre_landmarks.append(coords[:3])
                        post_landmarks.append(coords[3:])
                except ValueError:
                    continue
    return np.array(pre_landmarks), np.array(post_landmarks)

# Dataset Class
class UltrasoundDataset(Dataset):
    def __init__(self, dataset_dir):
        self.patient_dirs = sorted([os.path.join(dataset_dir, d) for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))])

    def __len__(self):
        return len(self.patient_dirs)

    def __getitem__(self, idx):
        patient_dir = self.patient_dirs[idx]
        pre_path = os.path.join(patient_dir, "pre.mnc")
        post_path = os.path.join(patient_dir, "post.mnc")
        tag_path = os.path.join(patient_dir, next(f for f in os.listdir(patient_dir) if f.endswith(".tag")))

        pre_volume = resize_volume(normalize(load_mnc(pre_path)))
        post_volume = resize_volume(normalize(load_mnc(post_path)))

        pre_tensor = torch.tensor(pre_volume).unsqueeze(0).float()  
        post_tensor = torch.tensor(post_volume).unsqueeze(0).float()

        pre_landmarks, post_landmarks = parse_tag_file(tag_path)

        return pre_tensor, post_tensor, torch.tensor(pre_landmarks).float(), torch.tensor(post_landmarks).float()