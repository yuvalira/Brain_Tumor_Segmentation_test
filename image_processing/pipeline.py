# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
import numpy as np

from image_processing.candidate_components import (
    extract_candidate_components, posterior_edge_map,
)
from image_processing.component_classification import (
    classify_components, normalized_entropy,
)
from image_processing.seed_expansion import expand_components


def segment_posteriors(posteriors, brain_mask, image_params, probability_params):
    tumor_posterior = np.sum(posteriors[..., 1:], axis=-1)
    entropy = normalized_entropy(posteriors, brain_mask)
    components, support = extract_candidate_components(
        tumor_posterior=tumor_posterior,
        brain_mask=brain_mask,
        candidate_threshold=probability_params["candidate_threshold"],
        min_component_size=image_params["min_component_size"],
        closing_size=image_params["closing_size"],
    )
    accepted_components, component_rows = classify_components(
        components=components,
        tumor_posterior=tumor_posterior,
        entropy=entropy,
        threshold=probability_params["component_threshold"],
    )
    segmentation = expand_components(
        accepted_components=accepted_components,
        tumor_posterior=tumor_posterior,
        entropy=entropy,
        brain_mask=brain_mask,
        entropy_threshold=probability_params["entropy_expansion_threshold"],
        posterior_threshold=probability_params["posterior_expansion_threshold"],
        max_expansion_distance=image_params["max_expansion_distance"],
    )
    return {
        "prediction": segmentation,
        "posteriors": posteriors,
        "tumor_posterior": tumor_posterior,
        "entropy": entropy,
        "edge_map": posterior_edge_map(tumor_posterior, brain_mask),
        "candidate_support": support,
        "components": components,
        "accepted_components": accepted_components,
        "component_table": component_rows,
    }
