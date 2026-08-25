# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
import numpy as np

from config import (
    MAX_HEALTHY_SAMPLES_PER_PATIENT,
    MAX_TUMOR_SAMPLES_PER_CLASS_PER_PATIENT,
    RANDOM_SEED,
)
from data.preprocessing import build_features, load_preprocessed_slice


def _random_rows(values, maximum, rng):
    if len(values) <= maximum:
        return np.asarray(values, dtype=np.float64)
    indices = rng.choice(len(values), size=maximum, replace=False)
    return np.asarray(values[indices], dtype=np.float64)


def collect_multiclass_samples(
    volume_ids,
    feature_kind,
    max_healthy=MAX_HEALTHY_SAMPLES_PER_PATIENT,
    max_tumor=MAX_TUMOR_SAMPLES_PER_CLASS_PER_PATIENT,
    seed=RANDOM_SEED,
):
    rng = np.random.default_rng(seed)
    healthy_samples = []
    tumor_samples = [[], [], []]
    class_counts = np.zeros(4, dtype=np.int64)

    for volume_id in volume_ids:
        data = load_preprocessed_slice(int(volume_id))
        features = build_features(data, feature_kind)
        healthy_mask = data.brain_mask & ~data.whole_tumor
        class_counts[0] += int(healthy_mask.sum())
        healthy_samples.append(_random_rows(features[healthy_mask], max_healthy, rng))
        for class_index in range(3):
            class_mask = data.tissue_masks[..., class_index]
            class_counts[class_index + 1] += int(class_mask.sum())
            if np.any(class_mask):
                tumor_samples[class_index].append(
                    _random_rows(features[class_mask], max_tumor, rng)
                )

    samples = [np.concatenate(healthy_samples, axis=0)]
    for class_index, parts in enumerate(tumor_samples):
        if not parts:
            raise ValueError(f"No samples found for tumor class {class_index}.")
        samples.append(np.concatenate(parts, axis=0))
    priors = class_counts / class_counts.sum()
    return samples, priors, class_counts


def collect_binary_branch_samples(
    volume_ids,
    channel_indices,
    positive_tissue_indices,
    max_healthy=MAX_HEALTHY_SAMPLES_PER_PATIENT,
    max_positive=MAX_TUMOR_SAMPLES_PER_CLASS_PER_PATIENT,
    seed=RANDOM_SEED,
):
    rng = np.random.default_rng(seed)
    healthy_samples, positive_samples = [], []
    counts = np.zeros(2, dtype=np.int64)

    for volume_id in volume_ids:
        data = load_preprocessed_slice(int(volume_id))
        features = data.image[..., channel_indices]
        healthy_mask = data.brain_mask & ~data.whole_tumor
        if positive_tissue_indices is None:
            positive_mask = data.whole_tumor
        else:
            positive_mask = np.any(
                data.tissue_masks[..., list(positive_tissue_indices)], axis=-1
            )
        counts += [int(healthy_mask.sum()), int(positive_mask.sum())]
        healthy_samples.append(_random_rows(features[healthy_mask], max_healthy, rng))
        if np.any(positive_mask):
            positive_samples.append(
                _random_rows(features[positive_mask], max_positive, rng)
            )

    if not positive_samples:
        raise ValueError("No positive samples found for hierarchical branch.")
    samples = [
        np.concatenate(healthy_samples, axis=0),
        np.concatenate(positive_samples, axis=0),
    ]
    priors = counts / counts.sum()
    return samples, priors, counts
