# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter

from config import (
    DATASET_DIR, EPSILON, MODALITY_NAMES, SLICE_NUM,
    SYMMETRY_AXIS, SYMMETRY_BLUR_SIGMA,
)


@dataclass
class SliceData:
    volume_id: int
    image: np.ndarray
    brain_mask: np.ndarray
    tissue_masks: np.ndarray
    whole_tumor: np.ndarray
    distance: np.ndarray
    symmetry: np.ndarray
    symmetric_brain_mask: np.ndarray


def slice_path(volume_id, slice_num=SLICE_NUM, dataset_dir=DATASET_DIR):
    return Path(dataset_dir) / f"volume_{int(volume_id)}_slice_{int(slice_num)}.h5"


def _tissue_masks(mask):
    if mask.ndim == 3:
        if mask.shape[-1] != 3:
            raise ValueError(f"Expected three tumor channels, received {mask.shape}.")
        return mask > 0
    return np.stack([mask == 1, mask == 2, mask == 4], axis=-1)


def _zscore_channels(image, mask):
    result = np.zeros_like(image, dtype=np.float64)
    if not np.any(mask):
        return result
    values = image[mask]
    mean = values.mean(axis=0)
    std = values.std(axis=0)
    std = np.where(std < EPSILON, 1.0, std)
    result[mask] = (values - mean) / std
    return result


def load_preprocessed_slice(volume_id, slice_num=SLICE_NUM, dataset_dir=DATASET_DIR):
    path = slice_path(volume_id, slice_num, dataset_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing central-slice file: {path}")
    with h5py.File(path, "r") as handle:
        raw_image = handle["image"][:].astype(np.float64)
        raw_mask = handle["mask"][:]

    if raw_image.ndim != 3 or raw_image.shape[-1] != len(MODALITY_NAMES):
        raise ValueError(f"Expected image shape (H, W, 4), received {raw_image.shape}.")

    brain_mask = np.any(raw_image > EPSILON, axis=-1)
    image = _zscore_channels(raw_image, brain_mask)
    tissue_masks = _tissue_masks(raw_mask) & brain_mask[..., None]
    whole_tumor = np.any(tissue_masks, axis=-1)

    distance_raw = distance_transform_edt(brain_mask)
    if distance_raw.max() > 0:
        distance_raw = distance_raw / distance_raw.max()
    distance = _zscore_channels(distance_raw[..., None], brain_mask)[..., 0]

    blurred = gaussian_filter(
        raw_image, sigma=(SYMMETRY_BLUR_SIGMA, SYMMETRY_BLUR_SIGMA, 0.0)
    )
    mirrored = np.flip(blurred, axis=SYMMETRY_AXIS)
    mirrored_brain = np.flip(brain_mask, axis=SYMMETRY_AXIS)
    symmetric_brain = brain_mask & mirrored_brain
    denominator = np.abs(blurred) + np.abs(mirrored) + EPSILON
    symmetry_raw = np.clip((blurred - mirrored) / denominator, -1.0, 1.0)
    symmetry = _zscore_channels(symmetry_raw, symmetric_brain)
    symmetry[~symmetric_brain] = 0.0

    return SliceData(
        volume_id=int(volume_id),
        image=image,
        brain_mask=brain_mask,
        tissue_masks=tissue_masks,
        whole_tumor=whole_tumor,
        distance=distance,
        symmetry=symmetry,
        symmetric_brain_mask=symmetric_brain,
    )


def build_features(data, feature_kind):
    if feature_kind == "raw":
        return data.image
    if feature_kind == "distance":
        return np.dstack([data.image, data.distance])
    if feature_kind == "symmetry":
        return np.dstack([data.image, data.symmetry])
    if feature_kind == "combined":
        return np.dstack([data.image, data.symmetry, data.distance])
    raise ValueError(f"Unknown feature kind: {feature_kind}")


def available_volume_ids(dataset_dir=DATASET_DIR, slice_num=SLICE_NUM):
    dataset_dir = Path(dataset_dir)
    ids = []
    for path in dataset_dir.glob(f"volume_*_slice_{slice_num}.h5"):
        try:
            ids.append(int(path.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return sorted(set(ids))
