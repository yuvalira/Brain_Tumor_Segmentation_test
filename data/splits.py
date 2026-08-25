# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import (
    RANDOM_SEED, SPLITS_DIR, TEST_SIZE, TOTAL_VOLUMES,
    TRAIN_SIZE, VALIDATION_SIZE,
)
from data.preprocessing import load_preprocessed_slice


def _build_strata(volume_ids):
    sizes = np.asarray([
        int(load_preprocessed_slice(volume_id).whole_tumor.sum())
        for volume_id in volume_ids
    ])
    strata = np.full(len(volume_ids), "empty", dtype=object)
    positive = sizes > 0
    if positive.sum() >= 6:
        bins = pd.qcut(sizes[positive], q=3, labels=False, duplicates="drop")
        strata[positive] = np.asarray([f"tumor_q{int(value)}" for value in bins])
    else:
        strata[positive] = "tumor"
    return sizes, strata


def _safe_stratify(labels):
    counts = pd.Series(labels).value_counts()
    return labels if len(counts) > 1 and counts.min() >= 2 else None


def create_splits(force=False, output_dir=SPLITS_DIR):
    output_dir = Path(output_dir)
    paths = {
        "train": output_dir / "train_ids.csv",
        "validation": output_dir / "validation_ids.csv",
        "test": output_dir / "test_ids.csv",
    }
    if all(path.exists() for path in paths.values()) and not force:
        return load_splits(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)
    volume_ids = np.arange(1, TOTAL_VOLUMES + 1)
    tumor_sizes, strata = _build_strata(volume_ids)

    development_ids, test_ids, development_strata, test_strata = train_test_split(
        volume_ids,
        strata,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=_safe_stratify(strata),
    )
    train_ids, validation_ids, train_strata, validation_strata = train_test_split(
        development_ids,
        development_strata,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED + 1,
        shuffle=True,
        stratify=_safe_stratify(development_strata),
    )
    if (len(train_ids), len(validation_ids), len(test_ids)) != (
        TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE
    ):
        raise RuntimeError("Unexpected split sizes.")

    size_lookup = dict(zip(volume_ids, tumor_sizes))
    for name, ids, labels in (
        ("train", train_ids, train_strata),
        ("validation", validation_ids, validation_strata),
        ("test", test_ids, test_strata),
    ):
        frame = pd.DataFrame({
            "volume_id": np.sort(ids),
            "tumor_pixels": [size_lookup[int(value)] for value in np.sort(ids)],
        })
        label_lookup = dict(zip(ids, labels))
        frame["stratum"] = [label_lookup[int(value)] for value in frame["volume_id"]]
        frame.to_csv(paths[name], index=False)

    return load_splits(output_dir)


def load_splits(output_dir=SPLITS_DIR):
    output_dir = Path(output_dir)
    splits = {
        name: pd.read_csv(output_dir / f"{name}_ids.csv")["volume_id"].astype(int).to_numpy()
        for name in ("train", "validation", "test")
    }
    all_ids = np.concatenate(list(splits.values()))
    if len(np.unique(all_ids)) != len(all_ids):
        raise ValueError("A patient occurs in more than one split.")
    return splits


def legacy_splits():
    return {
        "train": np.arange(1, 251),
        "validation": np.arange(251, 301),
        "test": np.arange(301, 370),
    }
