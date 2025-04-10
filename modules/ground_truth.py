# modules/ground_truth.py

import numpy as np

def parse_tag_file(tag_file_path):
    """
    Parse a .tag file and extract pre-resection and post-resection landmarks.
    
    Args:
        tag_file_path (str): Path to the .tag file.
    
    Returns:
        pre_landmarks (numpy.ndarray): Pre-resection landmarks (N x 3 array).
        post_landmarks (numpy.ndarray): Post-resection landmarks (N x 3 array).
    """
    pre_landmarks = []
    post_landmarks = []

    with open(tag_file_path, 'r') as file:
        lines = file.readlines()
        points_started = False

        for line in lines:
            line = line.strip()

            # Check if we reached the landmark points section
            if line.startswith("Points"):
                points_started = True
                continue

            # If we are in the points section, process the coordinates
            if points_started:
                if line == "" or ";" in line:
                    continue  

                # Try parsing numerical values, handle errors
                try:
                    coordinates = [float(x) for x in line.split() if x.replace('.', '', 1).replace('-', '', 1).isdigit()]
                    if len(coordinates) == 6:  
                        pre_landmarks.append(coordinates[:3])  # First 3 are pre-op
                        post_landmarks.append(coordinates[3:])  # Last 3 are post-op
                except ValueError:
                    print(f"Skipping invalid line in {tag_file_path}: {line}")
                    continue  

    return np.array(pre_landmarks), np.array(post_landmarks)

