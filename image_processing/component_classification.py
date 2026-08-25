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


def _mean_probability(component, probability):
    return float(np.mean(probability[component])) if probability is not None else 0.0


def classify_components(
    components,
    tumor_posterior,
    entropy,
    threshold,
    normal_min_component_size,
    small_min_component_size,
    small_component_q95_threshold,
    slice_gate_threshold,
    base_tumor_posterior=None,
    outer_probability=None,
    core_probability=None,
    hierarchy_confirmation_threshold=None,
    protection_margin=0.0,
):
    """Classify components, rescue confident small tumors, and gate empty slices."""
    accepted_components, rows = [], []
    for component_index in range(components.shape[-1]):
        component = components[..., component_index]
        area = int(component.sum())
        score = entropy_weighted_component_score(
            component, tumor_posterior, entropy
        )
        q95 = float(np.quantile(tumor_posterior[component], 0.95))
        base_score = (
            entropy_weighted_component_score(
                component, base_tumor_posterior, entropy
            )
            if base_tumor_posterior is not None else score
        )
        standard_size = area >= int(normal_min_component_size)
        confident_small = (
            area >= int(small_min_component_size)
            and q95 >= float(small_component_q95_threshold)
        )
        size_eligible = standard_size or confident_small
        protected = size_eligible and base_score >= threshold + protection_margin

        outer_score = _mean_probability(component, outer_probability)
        core_score = _mean_probability(component, core_probability)
        hierarchy_score = 0.75 * outer_score + 0.25 * core_score
        hierarchy_confirmed = (
            size_eligible
            and hierarchy_confirmation_threshold is not None
            and score >= threshold - 0.10
            and hierarchy_score >= hierarchy_confirmation_threshold
        )
        accepted = size_eligible and (
            protected or score >= threshold or hierarchy_confirmed
        )
        if standard_size:
            gate_score = 0.70 * score + 0.30 * q95
        else:
            gate_score = 0.35 * score + 0.65 * q95
        if hierarchy_confirmed:
            gate_score = max(gate_score, hierarchy_score)

        rows.append({
            "component": component_index,
            "area": area,
            "weighted_tumor_probability": score,
            "posterior_q95": q95,
            "base_weighted_probability": base_score,
            "standard_size": bool(standard_size),
            "confident_small_component": bool(confident_small),
            "protected_by_base_fusion": bool(protected),
            "mean_outer_probability": outer_score,
            "mean_core_probability": core_score,
            "hierarchy_confirmation_score": hierarchy_score,
            "confirmed_by_hierarchy": bool(hierarchy_confirmed),
            "slice_gate_score": gate_score,
            "accepted_before_slice_gate": bool(accepted),
        })
        if accepted:
            accepted_components.append(component)

    slice_gate_passed = any(
        row["accepted_before_slice_gate"]
        and row["slice_gate_score"] >= float(slice_gate_threshold)
        for row in rows
    )
    for row in rows:
        row["slice_gate_passed"] = bool(slice_gate_passed)
        row["accepted"] = bool(
            row["accepted_before_slice_gate"] and slice_gate_passed
        )

    if not accepted_components or not slice_gate_passed:
        accepted_stack = np.zeros((*components.shape[:2], 0), dtype=bool)
    else:
        accepted_stack = np.dstack(accepted_components)
    return accepted_stack, rows, slice_gate_passed
