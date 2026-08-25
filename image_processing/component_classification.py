# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
import numpy as np


def normalized_entropy(posteriors, brain_mask):
    probabilities = np.clip(posteriors, 1e-12, 1.0)
    entropy = -np.sum(probabilities * np.log(probabilities), axis=-1)
    entropy /= np.log(posteriors.shape[-1])
    entropy[~brain_mask] = 0.0
    return entropy


def entropy_weighted_component_score(component, tumor_posterior, entropy):
    weights = np.clip(1.0 - entropy[component], 0.0, 1.0)
    if weights.sum() <= 1e-12:
        return float(np.mean(tumor_posterior[component]))
    return float(np.sum(tumor_posterior[component] * weights) / weights.sum())


def classify_components(
    components,
    tumor_posterior,
    entropy,
    threshold,
    base_tumor_posterior=None,
    hierarchy_probability=None,
    protection_margin=0.0,
):
    accepted_components = []
    rows = []
    for component_index in range(components.shape[-1]):
        component = components[..., component_index]
        score = entropy_weighted_component_score(
            component, tumor_posterior, entropy
        )
        base_score = (
            entropy_weighted_component_score(
                component, base_tumor_posterior, entropy
            )
            if base_tumor_posterior is not None
            else score
        )
        protected = (
            base_tumor_posterior is not None
            and base_score >= threshold + protection_margin
        )
        accepted = protected or score >= threshold
        row = {
            "component": component_index,
            "area": int(component.sum()),
            "weighted_tumor_probability": score,
            "base_weighted_probability": base_score,
            "protected_by_base_fusion": bool(protected),
            "accepted": bool(accepted),
        }
        if hierarchy_probability is not None:
            row["mean_hierarchy_probability"] = float(
                np.mean(hierarchy_probability[component])
            )
        rows.append(row)
        if accepted:
            accepted_components.append(component)

    if not accepted_components:
        accepted_stack = np.zeros((*components.shape[:2], 0), dtype=bool)
    else:
        accepted_stack = np.dstack(accepted_components)
    return accepted_stack, rows
