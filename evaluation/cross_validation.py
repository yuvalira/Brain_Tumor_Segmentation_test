# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
import json
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CV_OUTPUT_DIR, MIN_HIERARCHY_VALIDATION_GAIN, MODEL_NAMES, RANDOM_SEED,
)
from data.splits import load_nested_cv_folds
from evaluation.evaluate import evaluate_model, save_evaluation
from evaluation.optimization import (
    load_selected_parameters, optimize_advanced_model, optimize_baseline,
    save_selected_parameters,
)
from models.training import train_or_load_models


ADVANCED_MODEL_NAMES = MODEL_NAMES[1:]


def _slug(name):
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "") \
        .replace("+", "plus")


def _with_fold(frame, fold_index):
    result = frame.copy()
    result.insert(0, "fold", int(fold_index))
    return result


def _summary_with_fold(summary, fold_index, hierarchy_selected=None):
    row = {"fold": int(fold_index), **summary}
    if hierarchy_selected is not None:
        row["hierarchy_selected"] = bool(hierarchy_selected)
    return row


def _load_completed_fold(fold_index, fold_dir):
    metadata_path = fold_dir / "fold_metadata.json"
    selected_path = fold_dir / "selected_parameters.json"
    required = [metadata_path, selected_path]
    for model_name in MODEL_NAMES:
        slug = _slug(model_name)
        required.extend([
            fold_dir / "validation" / f"{slug}_summary.csv",
            fold_dir / "test" / f"{slug}_summary.csv",
            fold_dir / "test" / f"{slug}_per_volume.csv",
        ])
    if not all(path.exists() for path in required):
        return None
    load_selected_parameters(selected_path)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hierarchy_selected = bool(metadata["hierarchy_selected"])
    validation_summaries, test_summaries, test_per_volume = [], [], []
    for model_name in MODEL_NAMES:
        slug = _slug(model_name)
        validation_summary = pd.read_csv(
            fold_dir / "validation" / f"{slug}_summary.csv"
        ).iloc[0].to_dict()
        test_summary = pd.read_csv(
            fold_dir / "test" / f"{slug}_summary.csv"
        ).iloc[0].to_dict()
        validation_summaries.append(_summary_with_fold(
            validation_summary, fold_index,
            hierarchy_selected if model_name == "Combined" else None,
        ))
        test_summaries.append(_summary_with_fold(
            test_summary, fold_index,
            hierarchy_selected if model_name == "Combined" else None,
        ))
        test_per_volume.append(_with_fold(pd.read_csv(
            fold_dir / "test" / f"{slug}_per_volume.csv"
        ), fold_index))
    return {
        "validation_summaries": validation_summaries,
        "test_summaries": test_summaries,
        "test_per_volume": test_per_volume,
    }


def _select_combined_or_fallback(
    models, validation_ids, image_params, probability_params, validation_results,
):
    fusion_result = validation_results["Boundary + Symmetry"]
    combined_result = validation_results["Combined"]
    gain = (
        combined_result["summary"]["dice_mean"]
        - fusion_result["summary"]["dice_mean"]
    )
    if gain >= MIN_HIERARCHY_VALIDATION_GAIN:
        probability_params["Combined"]["use_hierarchy"] = True
        return combined_result, True, gain

    probability_params["Combined"] = {
        **probability_params["Boundary + Symmetry"],
        "use_hierarchy": False,
    }
    fallback_result = evaluate_model(
        models["Combined"], validation_ids, image_params,
        probability_params["Combined"],
    )
    return fallback_result, False, gain


def run_outer_fold(
    fold,
    n_baseline_trials=30,
    n_advanced_trials=20,
    force_retrain_models=False,
    reuse_completed=True,
    output_dir=CV_OUTPUT_DIR,
):
    fold_index = int(fold["fold"])
    fold_dir = Path(output_dir) / f"fold_{fold_index}"
    if reuse_completed:
        completed = _load_completed_fold(fold_index, fold_dir)
        if completed is not None:
            print(f"Reusing completed outer fold {fold_index}.")
            return completed
    model_scope = f"nested_cv_v1/fold_{fold_index}"
    models = train_or_load_models(
        fold["train"], force=force_retrain_models, artifact_scope=model_scope
    )

    image_params, raw_params, _, _ = optimize_baseline(
        models["Raw (4D)"], fold["validation"], n_trials=n_baseline_trials
    )
    probability_params = {"Raw (4D)": raw_params}
    validation_results = {
        "Raw (4D)": evaluate_model(
            models["Raw (4D)"], fold["validation"], image_params, raw_params
        )
    }

    for model_offset, model_name in enumerate(ADVANCED_MODEL_NAMES, start=1):
        params, _, _ = optimize_advanced_model(
            models[model_name],
            fold["validation"],
            image_params,
            raw_params,
            n_trials=n_advanced_trials,
            seed=RANDOM_SEED + 100 * fold_index + model_offset,
        )
        probability_params[model_name] = params
        validation_results[model_name] = evaluate_model(
            models[model_name], fold["validation"], image_params, params
        )

    combined_result, hierarchy_selected, hierarchy_gain = \
        _select_combined_or_fallback(
            models, fold["validation"], image_params, probability_params,
            validation_results,
        )
    validation_results["Combined"] = combined_result

    test_results = {}
    for model_name in MODEL_NAMES:
        test_results[model_name] = evaluate_model(
            models[model_name], fold["test"], image_params,
            probability_params[model_name],
        )
        save_evaluation(
            validation_results[model_name], fold_dir / "validation",
            _slug(model_name),
        )
        save_evaluation(
            test_results[model_name], fold_dir / "test", _slug(model_name)
        )

    validation_summaries = {
        name: result["summary"] for name, result in validation_results.items()
    }
    save_selected_parameters(
        image_params,
        probability_params,
        validation_summaries,
        path=fold_dir / "selected_parameters.json",
    )
    metadata = {
        "fold": fold_index,
        "train_patients": len(fold["train"]),
        "validation_patients": len(fold["validation"]),
        "outer_test_patients": len(fold["test"]),
        "hierarchy_selected": hierarchy_selected,
        "hierarchy_candidate_validation_gain": hierarchy_gain,
        "minimum_required_hierarchy_gain": MIN_HIERARCHY_VALIDATION_GAIN,
    }
    fold_dir.mkdir(parents=True, exist_ok=True)
    (fold_dir / "fold_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return {
        "validation_summaries": [
            _summary_with_fold(
                validation_results[name]["summary"], fold_index,
                hierarchy_selected if name == "Combined" else None,
            ) for name in MODEL_NAMES
        ],
        "test_summaries": [
            _summary_with_fold(
                test_results[name]["summary"], fold_index,
                hierarchy_selected if name == "Combined" else None,
            ) for name in MODEL_NAMES
        ],
        "test_per_volume": [
            _with_fold(test_results[name]["per_volume"], fold_index)
            for name in MODEL_NAMES
        ],
    }


def _aggregate_outer_folds(frame, prefix):
    rows = []
    for model_name in MODEL_NAMES:
        group = frame[frame["model"] == model_name]
        rows.append({
            "model": model_name,
            f"{prefix}_dice_mean": group["dice_mean"].mean(),
            f"{prefix}_dice_std_across_folds": group["dice_mean"].std(ddof=0),
            f"{prefix}_iou_mean": group["iou_mean"].mean(),
            f"{prefix}_iou_std_across_folds": group["iou_mean"].std(ddof=0),
            f"{prefix}_tumor_present_dice": group["tumor_present_dice"].mean(),
            f"{prefix}_precision": group["precision"].mean(),
            f"{prefix}_recall": group["recall"].mean(),
            f"{prefix}_missed_tumors": int(group["missed_tumors"].sum()),
            f"{prefix}_tumor_slices": int(group["tumor_slices"].sum()),
            f"{prefix}_empty_slice_fp": int(
                group["empty_slice_false_positives"].sum()
            ),
            f"{prefix}_empty_slices": int(group["empty_slices"].sum()),
        })
    return pd.DataFrame(rows)


def run_nested_cross_validation(
    folds,
    n_baseline_trials=30,
    n_advanced_trials=20,
    force_retrain_models=False,
    reuse_completed_folds=True,
    output_dir=CV_OUTPUT_DIR,
):
    validation_rows, test_rows, per_volume_frames = [], [], []
    for fold in folds:
        print(f"\n{'=' * 72}\nOUTER FOLD {fold['fold']}/{len(folds)}\n{'=' * 72}")
        result = run_outer_fold(
            fold,
            n_baseline_trials=n_baseline_trials,
            n_advanced_trials=n_advanced_trials,
            force_retrain_models=force_retrain_models,
            reuse_completed=reuse_completed_folds,
            output_dir=output_dir,
        )
        validation_rows.extend(result["validation_summaries"])
        test_rows.extend(result["test_summaries"])
        per_volume_frames.extend(result["test_per_volume"])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_frame = pd.DataFrame(validation_rows)
    test_frame = pd.DataFrame(test_rows)
    per_volume = pd.concat(per_volume_frames, ignore_index=True)
    validation_summary = _aggregate_outer_folds(validation_frame, "validation")
    test_summary = _aggregate_outer_folds(test_frame, "outer_test")
    generalization = validation_summary.merge(test_summary, on="model")
    generalization["dice_gap"] = (
        generalization["validation_dice_mean"]
        - generalization["outer_test_dice_mean"]
    )
    hierarchy_selected_folds = int(
        test_frame.loc[
            test_frame["model"] == "Combined", "hierarchy_selected"
        ].sum()
    )
    generalization["hierarchy_selected_folds"] = np.where(
        generalization["model"] == "Combined", hierarchy_selected_folds, np.nan
    )

    validation_frame.to_csv(output_dir / "validation_fold_summaries.csv", index=False)
    test_frame.to_csv(output_dir / "outer_test_fold_summaries.csv", index=False)
    per_volume.to_csv(output_dir / "outer_test_per_volume.csv", index=False)
    generalization.to_csv(output_dir / "validation_vs_outer_test.csv", index=False)
    return {
        "validation_fold_summaries": validation_frame,
        "outer_test_fold_summaries": test_frame,
        "outer_test_per_volume": per_volume,
        "generalization_summary": generalization,
    }


def evaluate_saved_fold_models(fold, volume_ids, model_names=MODEL_NAMES):
    models = train_or_load_models(
        fold["train"], force=False,
        artifact_scope=f"nested_cv_v1/fold_{int(fold['fold'])}",
    )
    fold_dir = Path(CV_OUTPUT_DIR) / f"fold_{int(fold['fold'])}"
    selected = load_selected_parameters(fold_dir / "selected_parameters.json")
    image_params = selected["frozen_image_processing_params"]
    probability_params = selected["model_probability_params"]
    return {
        name: evaluate_model(
            models[name], volume_ids, image_params, probability_params[name],
            return_details=True,
        ) for name in model_names
    }
