# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
from dataclasses import dataclass

import numpy as np
from scipy.special import expit

from config import HIERARCHY_LOG_RATIO_CLIP


def _logit(probability):
    probability = np.clip(probability, 1e-8, 1.0 - 1e-8)
    return np.log(probability) - np.log1p(-probability)


def _calibrated_probability(log_odds, brain_mask, probability_params):
    offset = float(probability_params.get("log_odds_offset", 0.0))
    temperature = max(float(probability_params.get("temperature", 1.0)), 1e-3)
    probability = np.zeros(brain_mask.shape, dtype=np.float64)
    probability[brain_mask] = expit(
        (log_odds[brain_mask] + offset) / temperature
    )
    return probability


def _build_posteriors(tumor_probability, tumor_conditional, brain_mask):
    posteriors = np.zeros((*brain_mask.shape, 4), dtype=np.float64)
    posteriors[..., 0] = 1.0 - tumor_probability
    posteriors[..., 1:] = tumor_probability[..., None] * tumor_conditional
    posteriors[~brain_mask] = 0.0
    return posteriors


@dataclass
class BoundarySymmetryFusionModel:
    boundary_model: object
    symmetry_model: object
    name: str

    def prepare(self, data):
        return {
            "brain_mask": data.brain_mask,
            "boundary": self.boundary_model.prepare(data),
            "symmetry": self.symmetry_model.prepare(data),
        }

    def _base_maps(self, evidence, probability_params):
        weight = np.clip(
            float(probability_params.get("fusion_weight", 0.5)), 0.0, 1.0
        )
        boundary = evidence["boundary"]
        symmetry = evidence["symmetry"]
        boundary_probability = expit(boundary["tumor_log_odds"])
        symmetry_probability = expit(symmetry["tumor_log_odds"])
        base_probability = (
            weight * boundary_probability
            + (1.0 - weight) * symmetry_probability
        )
        base_log_odds = _logit(base_probability)
        conditional = (
            weight * boundary["tumor_conditional"]
            + (1.0 - weight) * symmetry["tumor_conditional"]
        )
        conditional_sum = conditional.sum(axis=-1, keepdims=True)
        conditional = np.divide(
            conditional,
            conditional_sum,
            out=np.zeros_like(conditional),
            where=conditional_sum > 1e-12,
        )
        return base_log_odds, conditional

    def posteriors_from_evidence(self, evidence, probability_params):
        brain_mask = evidence["brain_mask"]
        base_log_odds, conditional = self._base_maps(
            evidence, probability_params
        )
        probability = _calibrated_probability(
            base_log_odds, brain_mask, probability_params
        )
        return _build_posteriors(probability, conditional, brain_mask)

    def predict(self, data, probability_params):
        evidence = self.prepare(data)
        return self.posteriors_from_evidence(evidence, probability_params)


@dataclass
class ProtectedHierarchicalFusionModel(BoundarySymmetryFusionModel):
    outer_branch: object
    core_branch: object

    def prepare(self, data):
        evidence = super().prepare(data)
        evidence["outer_log_ratio"] = self.outer_branch.log_likelihood_ratio(data)
        evidence["core_log_ratio"] = self.core_branch.log_likelihood_ratio(data)
        return evidence

    def posteriors_from_evidence(self, evidence, probability_params):
        """Keep the fused posterior unchanged; hierarchy acts downstream."""
        return super().posteriors_from_evidence(evidence, probability_params)

    def segmentation_guidance(self, evidence, probability_params):
        if not probability_params.get("use_hierarchy", True):
            return None
        brain_mask = evidence["brain_mask"]
        base_log_odds, _ = self._base_maps(evidence, probability_params)
        outer_probability = np.zeros(brain_mask.shape, dtype=np.float64)
        core_probability = np.zeros(brain_mask.shape, dtype=np.float64)
        outer_probability[brain_mask] = expit(np.clip(
            evidence["outer_log_ratio"][brain_mask],
            -HIERARCHY_LOG_RATIO_CLIP,
            HIERARCHY_LOG_RATIO_CLIP,
        ))
        core_probability[brain_mask] = expit(np.clip(
            evidence["core_log_ratio"][brain_mask],
            -HIERARCHY_LOG_RATIO_CLIP,
            HIERARCHY_LOG_RATIO_CLIP,
        ))
        return {
            "base_tumor_probability": _calibrated_probability(
                base_log_odds, brain_mask, probability_params
            ),
            "outer_probability": outer_probability,
            "core_probability": core_probability,
            "hierarchy_probability": (
                0.75 * outer_probability + 0.25 * core_probability
            ),
            "protection_margin": 0.10,
        }
