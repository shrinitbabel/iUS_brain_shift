# iUS_brain_shift

This project presents a deep learning framework to estimate **intraoperative brain shift** using paired pre- and post-resection intraoperative ultrasound (iUS) volumes from the **BITE dataset**. The system predicts a dense 3D deformation field and is supervised with sparse anatomical landmarks for robust, interpretable shift estimation.

We evaluate two 3D U-Net architectures (baseline and enhanced) as well as their weighted ensemble, apply Grad-CAM explainability, and report extensive error metrics (MAE, angular deviation) using leave-one-out cross-validation (LOOCV) on 13 patients.

The contents of this repository have been submitted for consideration to a scholarly journal and are subject to copyright protection. Unauthorized duplication or reproduction is prohibited.

---

## Repository Structure

```plaintext
iUS_brain_shift/
├── brainshiftvectors/         # CSVs with predicted vs. ground-truth shift vectors (per test patient)
│   ├── brain_shift_vectors_test02.csv
│   ├── ...
│   └── brain_shift_vectors_test14.csv
├── model_explainability/      # Grad-CAM volumes and slice visualizations
│   ├── gradcam_volume_02.npy
│   ├── ...
│   ├── gradcam_volume_14.npy
│   └── gradcam_slices/        # 2D overlay heatmaps for patient 14 (expandable)
├── models/                    # Saved trained models
│   ├── brain_shift_model_02_baseline.pt
│   ├── ...
│   ├── brain_shift_model_14_baseline.pt
│   ├── brain_shift_model_02_enhanced.pt
│   ├── ...
│   └── brain_shift_model_14_enhanced.pt
├── modules/                   # Core codebase
│   ├── dataset.py             # Ultrasound dataset class
│   ├── explainability.py      # 3D Grad-CAM utilities
│   ├── ground_truth.py        # Tag file parser for landmarks
│   ├── load_model.py          # Model loading helpers
│   ├── post_process.py        # Warping, vector analysis
│   ├── test.py                # Core inference + evaluation logic
│   ├── train.py               # Model training loop
│   ├── unet3d.py              # Baseline 3D U-Net architecture
│   ├── unet3d_new.py          # Enhanced 3D U-Net architecture
│   └── utils.py               # Miscellaneous tools
├── results/                   # Raw result files per model
│   ├── unet3d_ensemble.xlsx   # results for ensemble
│   ├── unet3d_new.xlsx        # results for model 2
│   ├── unet3d_original.xlsx   # results for model 1/original
├── ensemble.ipynb             # Combines outputs from both models
├── model_testing.ipynb        # Evaluates a model on a single test patient
├── statistics.ipynb           # Computes MAE, angular error, SD, IQR per patient based on results/~.xlsx
├── train_and_generate_model.ipynb # Full LOOCV training pipeline
├── unet_explainability.ipynb  # Generates and saves 3D Grad-CAM volumes
├── requirements.txt           # python package requirements
├── Dockerfile                 # For reproducibility (optional)
├── LICENSE
└── README.md                  # You are here
```

---

## How to Run

> **Recommended Environment**: Python 3.9+ with PyTorch

### 1. Setup

```bash
git clone https://github.com/yourusername/iUS_brain_shift.git
cd iUS_brain_shift
pip install -r requirements.txt
```

### 2. Train the Model (LOOCV)

- Open `train_and_generate_model.ipynb` in Jupyter Notebook.
- This notebook trains both Baseline and Enhanced 3D U-Nets on 13 patients and saves the model for the held-out patient.

### 3. Test a Model

- Use `model_testing.ipynb` to:
    - Load a trained model.
    - Compute error metrics.
    - Visualize predicted shifts.
    - Compare with tag file landmarks.

### 4. Evaluate Metrics

- Use `statistics.ipynb` to compute:
    - Mean & Median Absolute Error (MAE).
    - Angular Error (°).
    - Cosine Similarity.
    - Standard Deviation (SD) & Interquartile Range (IQR).
    - Per-patient performance.

### 5. Interpretability

- Use `unet_explainability.ipynb` and `visualization.ipynb` to generate:
    - 3D Grad-CAM volumes.
    - Overlays with ultrasound for neurosurgical interpretability.
- View results slice-by-slice or in 3D using `vedo`.

### 6. Ensemble Model (Optional)

- Use `ensemble.ipynb` to average flow predictions from baseline and enhanced models to test ensemble performance. Uses 7:3 weighting. 

---

## Architectures

- **Baseline U-Net3D**: Simple, compact, ReLU-based.
- **Enhanced U-Net3D**: Uses BatchNorm, LeakyReLU, Dropout.

Both trained using L2 + landmark + smoothness + angular loss.

---

## Results Summary

- Evaluated with LOOCV across 13 BITE patients.
- Metrics include MAE, RMSE, angular deviation, and Grad-CAM heatmaps.
- Ensemble improves angular alignment while individual models trade off magnitude vs direction.

---

## License

This project is licensed under the terms of the MIT License.

---

## Acknowledgments

- **Dataset**: BITE (Brain Images of Tumors for Evaluation).
- **Frameworks**: PyTorch, Vedo, Nibabel.
