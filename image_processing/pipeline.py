# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
import numpy as np

from image_processing.candidate_components import (
    extract_candidate_components, posterior_edge_map,
)
from image_processing.component_classification import (
    classify_components, normalized_entropy,
)
from image_processing.seed_expansion import expand_components


def segment_posteriors(
    posteriors,
    brain_mask,
    image_params,
    probability_params,
    guidance=None,
):
    tumor_posterior = np.sum(posteriors[..., 1:], axis=-1)
    entropy = normalized_entropy(posteriors, brain_mask)
    base_posterior = (
        guidance.get("base_tumor_probability")
        if guidance is not None else None
    )
    candidate_posterior = (
        np.maximum(tumor_posterior, base_posterior)
        if base_posterior is not None else tumor_posterior
    )
    outer_probability = (
        guidance.get("outer_probability")
        if guidance is not None else None
    )
    core_probability = (
        guidance.get("core_probability")
        if guidance is not None else None
    )
    components, support = extract_candidate_components(
        tumor_posterior=candidate_posterior,
        brain_mask=brain_mask,
        candidate_threshold=probability_params["candidate_threshold"],
        small_min_component_size=image_params["small_min_component_size"],
        closing_size=image_params["closing_size"],
    )
    accepted_components, component_rows, slice_gate_passed = classify_components(
        components=components,
        tumor_posterior=tumor_posterior,
        entropy=entropy,
        threshold=probability_params["component_threshold"],
        normal_min_component_size=image_params["min_component_size"],
        small_min_component_size=image_params["small_min_component_size"],
        small_component_q95_threshold=probability_params[
            "small_component_q95_threshold"
        ],
        slice_gate_threshold=probability_params["slice_gate_threshold"],
        base_tumor_posterior=base_posterior,
        outer_probability=outer_probability,
        core_probability=core_probability,
        hierarchy_confirmation_threshold=(
            probability_params.get("hierarchy_confirmation_threshold")
            if outer_probability is not None else None
        ),
        protection_margin=(
            guidance.get("protection_margin", 0.0)
            if guidance is not None else 0.0
        ),
    )
    segmentation = expand_components(
        accepted_components=accepted_components,
        tumor_posterior=candidate_posterior,
        entropy=entropy,
        brain_mask=brain_mask,
        entropy_threshold=probability_params["entropy_expansion_threshold"],
        posterior_threshold=probability_params["posterior_expansion_threshold"],
        max_expansion_distance=image_params["max_expansion_distance"],
        outer_probability=outer_probability,
        outer_expansion_threshold=(
            probability_params.get("outer_expansion_threshold")
            if outer_probability is not None else None
        ),
    )
    hierarchy_probability = (
        guidance.get("hierarchy_probability")
        if guidance is not None else None
    )
    return {
        "prediction": segmentation,
        "posteriors": posteriors,
        "tumor_posterior": tumor_posterior,
        "base_tumor_posterior": base_posterior,
        "outer_probability": outer_probability,
        "core_probability": core_probability,
        "hierarchy_probability": hierarchy_probability,
        "entropy": entropy,
        "edge_map": posterior_edge_map(tumor_posterior, brain_mask),
        "candidate_support": support,
        "components": components,
        "accepted_components": accepted_components,
        "component_table": component_rows,
        "slice_gate_passed": bool(slice_gate_passed),
    }
