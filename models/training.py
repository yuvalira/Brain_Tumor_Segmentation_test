# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
from pathlib import Path

from config import (
    HEALTHY_COMPONENTS, MODEL_DIR, RANDOM_SEED, TUMOR_COMPONENTS,
)
from data.sampling import (
    collect_binary_branch_samples, collect_multiclass_samples,
)
from models.hierarchical_model import BinaryLogGMMBranch, HierarchicalGMMModel
from models.log_gmm import LogSpaceGMMClassifier


FEATURE_MODELS = {
    "Raw (4D)": "raw",
    "Boundary distance (5D)": "distance",
    "Symmetry (8D)": "symmetry",
    "Combined features (9D)": "combined",
}


def _slug(name):
    return (
        name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        .replace("+", "plus")
    )


def train_feature_model(name, feature_kind, train_ids, force=False):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{_slug(name)}.joblib"
    if path.exists() and not force:
        return LogSpaceGMMClassifier.load(path)

    samples, priors, _ = collect_multiclass_samples(
        train_ids,
        feature_kind=feature_kind,
        seed=RANDOM_SEED + len(feature_kind),
    )
    model = LogSpaceGMMClassifier.fit(
        samples=samples,
        priors=priors,
        component_counts=[HEALTHY_COMPONENTS] + [TUMOR_COMPONENTS] * 3,
        feature_kind=feature_kind,
        name=name,
    )
    model.save(path)
    return model


def train_hierarchical_branches(train_ids, force=False):
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    outer_path = MODEL_DIR / "hierarchical_outer_flair_t2.joblib"
    core_path = MODEL_DIR / "hierarchical_core_t1_t1ce.joblib"

    if outer_path.exists() and core_path.exists() and not force:
        return (
            BinaryLogGMMBranch.load(outer_path),
            BinaryLogGMMBranch.load(core_path),
        )

    outer_samples, _, _ = collect_binary_branch_samples(
        train_ids,
        channel_indices=[2, 3],
        positive_tissue_indices=None,
        seed=RANDOM_SEED + 20,
    )
    core_samples, _, _ = collect_binary_branch_samples(
        train_ids,
        channel_indices=[0, 1],
        positive_tissue_indices=[0, 2],
        seed=RANDOM_SEED + 21,
    )
    outer = BinaryLogGMMBranch.fit(
        outer_samples,
        component_counts=[HEALTHY_COMPONENTS, TUMOR_COMPONENTS],
        channel_indices=[2, 3],
        name="FLAIR-T2 whole-tumor branch",
        seed=RANDOM_SEED + 30,
    )
    core = BinaryLogGMMBranch.fit(
        core_samples,
        component_counts=[HEALTHY_COMPONENTS, TUMOR_COMPONENTS],
        channel_indices=[0, 1],
        name="T1-T1ce core branch",
        seed=RANDOM_SEED + 40,
    )
    outer.save(outer_path)
    core.save(core_path)
    return outer, core


def train_or_load_models(train_ids, force=False):
    raw = train_feature_model("Raw (4D)", "raw", train_ids, force)
    distance = train_feature_model(
        "Boundary distance (5D)", "distance", train_ids, force
    )
    symmetry = train_feature_model("Symmetry (8D)", "symmetry", train_ids, force)
    combined_features = train_feature_model(
        "Combined features (9D)", "combined", train_ids, force
    )
    outer, core = train_hierarchical_branches(train_ids, force)
    return {
        "Raw (4D)": raw,
        "Boundary distance (5D)": distance,
        "Symmetry (8D)": symmetry,
        "Hierarchical modalities": HierarchicalGMMModel(
            raw, outer, core, "Hierarchical modalities"
        ),
        "Combined": HierarchicalGMMModel(
            combined_features, outer, core, "Combined"
        ),
    }
