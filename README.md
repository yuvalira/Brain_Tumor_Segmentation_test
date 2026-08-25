# Central-Slice Brain Tumor Segmentation with Log-Space GMMs

Classical binary whole-tumor segmentation on BraTS 2020 using only axial slice 80 from T1, T1ce, T2 and FLAIR.

## Models

- **Raw (4D):** log-space multiclass GMM baseline.
- **Boundary distance (5D):** raw modalities plus relative 2D brain-boundary distance.
- **Symmetry (8D):** raw modalities plus bilateral NDI along the dataset's verified symmetry axis.
- **Boundary + Symmetry:** weighted probability fusion of the separately trained 5D and 8D GMMs.
- **Combined:** boundary-symmetry fusion with protected FLAIR/T2 whole-tumor and T1/T1ce core guidance.

The baseline selects image-processing and probability parameters on the validation set. Its image-processing and segmentation thresholds are frozen for every advanced model. Advanced selection is deliberately low-dimensional: calibration offset, one fusion weight, and hierarchy-specific weights where applicable. Validation mean Dice remains the primary objective.

After parameter selection, separate final GMMs are fitted on the combined train + validation development set and evaluated on the locked test set.

## Evaluation

The final test evaluation reports mean and standard deviation of Dice and IoU, required boxplots, paired scatterplots with Pearson correlation, qualitative examples and additional failure diagnostics. Empty ground truth with empty prediction is scored as Dice = IoU = 1.

## Run

1. Place the BraTS HDF5 central-slice files in the default dataset directory or set `BRATS_DATA_DIR`.
2. Install dependencies:

   ```bash
   python -m pip install -r requirements.txt
   ```

3. Open `main.ipynb` and run it from top to bottom.

`main.ipynb` retains the complete model-by-model report and diagnostics. The
five-fold leakage-safe evaluation is available separately in
`nested_cross_validation.ipynb` so it does not replace or hide any part of the
main report.

After preprocessing or model changes, run once with:

```python
FORCE_RETRAIN_MODELS = True
REUSE_SELECTED_PARAMETERS = False
```

After a successful complete run, return these switches to `False` and `True`, respectively, to reuse the corrected selection and final-fit artifacts.

The dataset and medical image files must not be committed to GitHub.
