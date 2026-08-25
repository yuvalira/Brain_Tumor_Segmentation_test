# Central-Slice Brain Tumor Segmentation with Log-Space GMMs

Classical binary whole-tumor segmentation on BraTS 2020 using only axial slice 80 from T1, T1ce, T2 and FLAIR.

## Models

- **Raw (4D):** log-space multiclass GMM baseline.
- **Boundary distance (5D):** raw modalities plus relative 2D brain-boundary distance.
- **Symmetry (8D):** raw modalities plus bilateral NDI along the dataset's verified symmetry axis.
- **Boundary + Symmetry:** weighted probability fusion of the separately trained 5D and 8D GMMs.
- **Combined:** boundary-symmetry fusion with protected FLAIR/T2 whole-tumor and T1/T1ce core guidance.

Within each outer fold, the baseline selects image-processing and probability parameters on the inner validation set. Its image-processing and segmentation thresholds are frozen for every advanced model in that fold. Advanced selection is deliberately low-dimensional: calibration offset, one fusion weight, and hierarchy-specific thresholds where applicable. Validation mean Dice remains the primary objective.

The Combined hierarchy is retained only when it improves validation Dice over Boundary + Symmetry by at least 0.005. Otherwise, Combined uses the exact no-hierarchy fusion fallback for that fold.

## Evaluation

Evaluation uses five patient-level outer folds. Inside each outer fold, the remaining patients are split into GMM-training and parameter-validation groups. GMMs never see the validation or outer-test patients, and outer-test results never affect parameter or hierarchy selection. Every patient receives one out-of-fold test prediction. The report includes mean and standard deviation across outer folds, validation-to-test gaps, failure counts, paired scatterplots, qualitative examples and pipeline diagnostics. Empty ground truth with empty prediction is scored as Dice = IoU = 1.

## Run

1. Place the BraTS HDF5 central-slice files in the default dataset directory or set `BRATS_DATA_DIR`.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Open `main.ipynb` and run it from top to bottom.

The long-running cross-validation cell saves each fold independently. To resume an interrupted run, keep:

```python
FORCE_NEW_FOLDS = False
FORCE_RETRAIN_FOLD_MODELS = False
REUSE_COMPLETED_FOLDS = True
```

Do not regenerate the folds after inspecting outer-test results. Set `FORCE_RETRAIN_FOLD_MODELS=True` only after changing GMM fitting or features; post-processing changes do not require it.

The dataset and medical image files must not be committed to GitHub.
