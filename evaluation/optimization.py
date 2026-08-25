# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
import json
from pathlib import Path

import optuna

from config import (
    DEFAULT_IMAGE_PROCESSING_PARAMS,
    DEFAULT_PROBABILITY_PARAMS,
    RANDOM_SEED,
    SELECTED_PARAMETERS_PATH,
)
from evaluation.evaluate import evaluate_model, prepare_evidence_cache
from models.hierarchical_model import HierarchicalGMMModel


_PARAMETER_REFERENCE = {
    **DEFAULT_IMAGE_PROCESSING_PARAMS,
    **DEFAULT_PROBABILITY_PARAMS,
    "hierarchy_weight": 1.0,
    "core_weight": 1.0,
}
_PARAMETER_SCALE = {
    "min_component_size": 55.0,
    "closing_size": 6.0,
    "max_expansion_distance": 75.0,
    "log_odds_offset": 16.0,
    "temperature": 1.5,
    "candidate_threshold": 0.50,
    "component_threshold": 0.60,
    "entropy_expansion_threshold": 0.48,
    "posterior_expansion_threshold": 0.43,
    "hierarchy_weight": 2.0,
    "core_weight": 3.75,
}


def _parameter_distance(params):
    differences = [
        ((float(value) - _PARAMETER_REFERENCE[name]) / _PARAMETER_SCALE[name]) ** 2
        for name, value in params.items()
        if name in _PARAMETER_REFERENCE
    ]
    return sum(differences) / max(len(differences), 1)


def _record_selection_diagnostics(trial, summary):
    trial.set_user_attr("missed_tumors", summary["missed_tumors"])
    trial.set_user_attr(
        "empty_slice_false_positives",
        summary["empty_slice_false_positives"],
    )
    trial.set_user_attr("dice_std", summary["dice_std"])
    trial.set_user_attr(
        "parameter_distance", _parameter_distance(trial.params)
    )


def _suggest_probability_params(trial, hierarchical=False):
    params = {
        "log_odds_offset": trial.suggest_float("log_odds_offset", -8.0, 8.0),
        "temperature": trial.suggest_float("temperature", 0.5, 2.0),
        "candidate_threshold": trial.suggest_float(
            "candidate_threshold", 0.05, 0.55
        ),
        "component_threshold": trial.suggest_float(
            "component_threshold", 0.30, 0.90
        ),
        "entropy_expansion_threshold": trial.suggest_float(
            "entropy_expansion_threshold", 0.02, 0.50
        ),
        "posterior_expansion_threshold": trial.suggest_float(
            "posterior_expansion_threshold", 0.02, 0.45
        ),
    }
    if hierarchical:
        params["hierarchy_weight"] = trial.suggest_float(
            "hierarchy_weight", 0.0, 2.0
        )
        params["core_weight"] = trial.suggest_float(
            "core_weight", 0.25, 4.0, log=True
        )
    return params


def _select_trial(study, tolerance=0.005):
    completed = [
        trial for trial in study.trials
        if trial.value is not None and trial.state == optuna.trial.TrialState.COMPLETE
    ]
    best_value = max(trial.value for trial in completed)
    nearly_best = [
        trial for trial in completed if trial.value >= best_value - tolerance
    ]
    return min(
        nearly_best,
        key=lambda trial: (
            trial.user_attrs["missed_tumors"],
            trial.user_attrs["empty_slice_false_positives"],
            trial.user_attrs["parameter_distance"],
            trial.user_attrs["dice_std"],
            -trial.value,
        ),
    )


def optimize_baseline(model, validation_ids, n_trials=60):
    cache = prepare_evidence_cache(model, validation_ids)

    def objective(trial):
        image_params = {
            "min_component_size": trial.suggest_int(
                "min_component_size", 5, 60
            ),
            "closing_size": trial.suggest_categorical(
                "closing_size", [1, 3, 5, 7]
            ),
            "max_expansion_distance": trial.suggest_int(
                "max_expansion_distance", 5, 80
            ),
        }
        probability_params = _suggest_probability_params(trial)
        result = evaluate_model(
            model, validation_ids, image_params, probability_params, cache
        )
        summary = result["summary"]
        _record_selection_diagnostics(trial, summary)
        return summary["dice_mean"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.enqueue_trial({
        **DEFAULT_IMAGE_PROCESSING_PARAMS,
        **DEFAULT_PROBABILITY_PARAMS,
    })
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    selected_trial = _select_trial(study)
    image_keys = {
        "min_component_size", "closing_size", "max_expansion_distance"
    }
    image_params = {
        key: selected_trial.params[key] for key in image_keys
    }
    probability_params = {
        key: value for key, value in selected_trial.params.items()
        if key not in image_keys
    }
    return image_params, probability_params, study, selected_trial


def optimize_advanced_model(
    model,
    validation_ids,
    frozen_image_params,
    n_trials=50,
    seed=RANDOM_SEED + 1,
):
    cache = prepare_evidence_cache(model, validation_ids)
    hierarchical = isinstance(model, HierarchicalGMMModel)

    def objective(trial):
        probability_params = _suggest_probability_params(trial, hierarchical)
        result = evaluate_model(
            model,
            validation_ids,
            frozen_image_params,
            probability_params,
            cache,
        )
        summary = result["summary"]
        _record_selection_diagnostics(trial, summary)
        return summary["dice_mean"]

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    initial = dict(DEFAULT_PROBABILITY_PARAMS)
    if hierarchical:
        initial.update({"hierarchy_weight": 1.0, "core_weight": 1.0})
    study.enqueue_trial(initial)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    selected_trial = _select_trial(study)
    return selected_trial.params, study, selected_trial


def save_selected_parameters(
    image_params,
    model_probability_params,
    validation_summaries,
    path=SELECTED_PARAMETERS_PATH,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "workflow_version": "central_slice_log_gmm_components_v2",
        "selection_rule": (
            "Maximize validation mean Dice; within 0.005 of the maximum, "
            "prefer fewer missed tumors, fewer empty-slice false positives, "
            "parameters closer to the predefined defaults, and lower Dice variability."
        ),
        "frozen_image_processing_params": image_params,
        "model_probability_params": model_probability_params,
        "validation_summaries": validation_summaries,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_selected_parameters(path=SELECTED_PARAMETERS_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))
