# Central-Slice Brain Tumor Segmentation with Log-Space GMMs

Classical binary whole-tumor segmentation on axial slice 80 of BraTS 2020. The project compares a raw 4D log-space GMM baseline with relative-distance, bilateral-symmetry, hierarchical-modality, and fully combined advanced models.

The baseline jointly selects image-processing and probabilistic parameters on validation data. Image-processing parameters are then frozen, and every advanced model selects only probabilistic parameters. The test set is evaluated once after all parameters are frozen.

## Pipeline

All training and inference use only slice 80:

```
central-slice MRI -> Z-score normalization -> log-space GMM posterior
-> posterior support -> connected components
-> entropy-weighted component classification
-> ambiguous-space seed expansion -> whole-tumor mask
```

## Data

Do not upload BraTS images to GitHub. The default relative dataset location is:

```
../MRI_2026_datasets/Brats/BraTS2020_training_data/content/data
```

Files follow `volume_<patient>_slice_<slice>.h5` and contain `image` and `mask`. Set `BRATS_DATA_DIR` to override the default.

## Run

1. `python -m pip install -r requirements.txt`
2. Open `main.ipynb`.
3. Fill in both students' names and IDs.
4. Run from top to bottom.
5. Save the executed notebook with all outputs.

Generated splits, model parameters, selected parameters, metrics, figures, and diagnostics are saved in their corresponding project folders.

## Reproducibility

- Random seed: 42
- Central slice: 80
- Binary whole-tumor target
- Patient-level train/validation/test: 250/50/69
- Test evaluation occurs only after model selection

## Authors

- Yuval Ratzabi - student ID: **TODO**
- Second student - name and student ID: **TODO**
