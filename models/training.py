# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
from pathlib import Path

from config import (
    HEALTHY_COMPONENTS, MODEL_DIR, RANDOM_SEED, TUMOR_COMPONENTS,
)
from data.sampling import (
    collect_binary_branch_samples, collect_multiclass_samples,
)
from models.fusion_model import (
    BoundarySymmetryFusionModel, ProtectedHierarchicalFusionModel,
)
from models.hierarchical_model import BinaryLogGMMBranch
from models.log_gmm import LogSpaceGMMClassifier


def _slug(name):
    return (
        name.lower().replace(" ", "_").replace("(", "").replace(")", "")
        .replace("+", "plus")
    )


def train_feature_model(
    name,
    feature_kind,
    train_ids,
    force=False,
    artifact_dir=MODEL_DIR,
):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    artifact_name = _slug(name)
    if feature_kind == "symmetry":
        artifact_name += "_dataset_axis0_v3"
    path = artifact_dir / f"{artifact_name}.joblib"
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


def train_hierarchical_branches(
    train_ids,
    force=False,
    artifact_dir=MODEL_DIR,
):
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    outer_path = artifact_dir / "hierarchical_outer_flair_t2.joblib"
    core_path = artifact_dir / "hierarchical_core_t1_t1ce.joblib"

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


def train_or_load_models(
    train_ids,
    force=False,
    artifact_scope="selection",
):
    artifact_dir = MODEL_DIR / artifact_scope
    raw = train_feature_model(
        "Raw (4D)", "raw", train_ids, force, artifact_dir
    )
    boundary = train_feature_model(
        "Boundary distance (5D)", "distance", train_ids, force, artifact_dir
    )
    symmetry = train_feature_model(
        "Symmetry (8D)", "symmetry", train_ids, force, artifact_dir
    )
    outer, core = train_hierarchical_branches(
        train_ids, force, artifact_dir
    )
    fusion = BoundarySymmetryFusionModel(
        boundary, symmetry, "Boundary + Symmetry"
    )
    combined = ProtectedHierarchicalFusionModel(
        boundary_model=boundary,
        symmetry_model=symmetry,
        name="Combined",
        outer_branch=outer,
        core_branch=core,
    )
    return {
        "Raw (4D)": raw,
        "Boundary distance (5D)": boundary,
        "Symmetry (8D)": symmetry,
        "Boundary + Symmetry": fusion,
        "Combined": combined,
    }
