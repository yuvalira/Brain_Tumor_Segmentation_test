# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from config import (
    CV_SPLITS_DIR, INNER_VALIDATION_FRACTION, N_OUTER_FOLDS, RANDOM_SEED,
    SPLITS_DIR, TEST_SIZE, TOTAL_VOLUMES, TRAIN_SIZE, VALIDATION_SIZE,
)


def _build_strata(volume_ids):
    from data.preprocessing import load_preprocessed_slice

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


def _nested_fold_indices(volume_ids, strata):
    splitter = StratifiedKFold(
        n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED
    )
    folds = []
    for fold_index, (development_indices, test_indices) in enumerate(
        splitter.split(volume_ids, strata), start=1
    ):
        development_ids = volume_ids[development_indices]
        development_strata = strata[development_indices]
        validation_size = int(round(
            len(development_ids) * INNER_VALIDATION_FRACTION
        ))
        train_ids, validation_ids = train_test_split(
            development_ids,
            test_size=validation_size,
            random_state=RANDOM_SEED + fold_index,
            shuffle=True,
            stratify=_safe_stratify(development_strata),
        )
        folds.append({
            "fold": fold_index,
            "train": np.sort(train_ids),
            "validation": np.sort(validation_ids),
            "test": np.sort(volume_ids[test_indices]),
        })
    return folds


def _validate_nested_folds(folds, volume_ids):
    outer_test_ids = []
    expected = set(map(int, volume_ids))
    for fold in folds:
        sets = {
            role: set(map(int, fold[role]))
            for role in ("train", "validation", "test")
        }
        if sets["train"] & sets["validation"] or sets["train"] & sets["test"] \
                or sets["validation"] & sets["test"]:
            raise ValueError(f"Patient overlap in outer fold {fold['fold']}.")
        if set.union(*sets.values()) != expected:
            raise ValueError(f"Incomplete patient coverage in fold {fold['fold']}.")
        outer_test_ids.extend(sets["test"])
    if len(outer_test_ids) != len(expected) or set(outer_test_ids) != expected:
        raise ValueError("Each patient must occur in exactly one outer test fold.")


def _nested_paths(output_dir=CV_SPLITS_DIR):
    output_dir = Path(output_dir)
    return {
        fold: {
            role: output_dir / f"fold_{fold}" / f"{role}_ids.csv"
            for role in ("train", "validation", "test")
        }
        for fold in range(1, N_OUTER_FOLDS + 1)
    }


def create_nested_cv_folds(force=False, output_dir=CV_SPLITS_DIR):
    paths = _nested_paths(output_dir)
    if not force and all(
        path.exists() for fold_paths in paths.values()
        for path in fold_paths.values()
    ):
        return load_nested_cv_folds(output_dir)

    volume_ids = np.arange(1, TOTAL_VOLUMES + 1)
    tumor_sizes, strata = _build_strata(volume_ids)
    folds = _nested_fold_indices(volume_ids, strata)
    _validate_nested_folds(folds, volume_ids)
    size_lookup = dict(zip(volume_ids, tumor_sizes))
    stratum_lookup = dict(zip(volume_ids, strata))
    for fold in folds:
        for role in ("train", "validation", "test"):
            ids = fold[role]
            path = paths[fold["fold"]][role]
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({
                "volume_id": ids,
                "tumor_pixels": [size_lookup[int(value)] for value in ids],
                "stratum": [stratum_lookup[int(value)] for value in ids],
            }).to_csv(path, index=False)
    return load_nested_cv_folds(output_dir)


def load_nested_cv_folds(output_dir=CV_SPLITS_DIR):
    paths = _nested_paths(output_dir)
    folds = []
    for fold_index, fold_paths in paths.items():
        fold = {"fold": fold_index}
        for role, path in fold_paths.items():
            fold[role] = pd.read_csv(path)["volume_id"].astype(int).to_numpy()
        folds.append(fold)
    _validate_nested_folds(folds, np.arange(1, TOTAL_VOLUMES + 1))
    return folds


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
