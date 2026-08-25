# Central-Slice Brain Tumor Segmentation with Log-Space GMMs

Classical binary whole-tumor segmentation on BraTS 2020 using only axial slice 80 from T1, T1ce, T2 and FLAIR.

## Models

- **Raw (4D):** log-space multiclass GMM baseline.
- **Boundary distance (5D):** raw modalities plus relative 2D brain-boundary distance.
- **Symmetry (8D):** raw modalities plus bilateral NDI along the dataset's verified symmetry axis.
- **Hierarchical modalities:** raw GMM with FLAIR/T2 whole-tumor and T1/T1ce core branches.
- **Combined:** intensity, boundary distance, symmetry and bounded hierarchical evidence.

The baseline selects image-processing and probability parameters on the validation set. Image-processing parameters are then frozen, and only probability parameters are selected for the advanced models. Validation mean Dice remains the primary objective; nearly equal trials prefer fewer missed tumors, fewer empty-slice false positives, less-extreme parameters and lower Dice variability.

## Evaluation

The final locked test evaluation reports mean and standard deviation of Dice and IoU, required boxplots, paired scatterplots with Pearson correlation, qualitative examples and additional failure diagnostics. Empty ground truth with empty prediction is scored as Dice = IoU = 1.

## Run

1. Place the BraTS HDF5 central-slice files in the default dataset directory or set `BRATS_DATA_DIR`.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Open `main.ipynb` and run it from top to bottom.

After preprocessing or model changes, run once with:

```python
FORCE_RETRAIN_MODELS = True
REUSE_SELECTED_PARAMETERS = False
```

After a successful complete run, return these switches to `False` and `True`, respectively, to reuse the corrected artifacts.

The dataset and medical image files must not be committed to GitHub.
