# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
from dataclasses import dataclass

import joblib
import numpy as np
from scipy.special import expit
from sklearn.mixture import GaussianMixture

from config import (
    GMM_MAX_ITER, GMM_N_INIT, GMM_REG_COVAR, RANDOM_SEED,
)


@dataclass
class BinaryLogGMMBranch:
    healthy_gmm: GaussianMixture
    tumor_gmm: GaussianMixture
    channel_indices: tuple
    name: str

    @classmethod
    def fit(
        cls,
        samples,
        component_counts,
        channel_indices,
        name,
        seed=RANDOM_SEED,
    ):
        gmms = []
        for class_index, (values, components) in enumerate(
            zip(samples, component_counts)
        ):
            model = GaussianMixture(
                n_components=components,
                covariance_type="full",
                reg_covar=GMM_REG_COVAR,
                max_iter=GMM_MAX_ITER,
                n_init=GMM_N_INIT,
                init_params="kmeans",
                random_state=seed + class_index,
            )
            model.fit(values)
            gmms.append(model)
        return cls(gmms[0], gmms[1], tuple(channel_indices), name)

    def log_likelihood_ratio(self, data):
        features = data.image[..., list(self.channel_indices)]
        values = features[data.brain_mask]
        ratio = (
            self.tumor_gmm.score_samples(values)
            - self.healthy_gmm.score_samples(values)
        )
        result = np.zeros(data.brain_mask.shape, dtype=np.float64)
        result[data.brain_mask] = ratio
        return result

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


@dataclass
class HierarchicalGMMModel:
    base_model: object
    outer_branch: BinaryLogGMMBranch
    core_branch: BinaryLogGMMBranch
    name: str

    def prepare(self, data):
        evidence = self.base_model.prepare(data)
        evidence["outer_log_ratio"] = self.outer_branch.log_likelihood_ratio(data)
        evidence["core_log_ratio"] = self.core_branch.log_likelihood_ratio(data)
        return evidence

    def posteriors_from_evidence(self, evidence, probability_params):
        hierarchy_weight = float(
            probability_params.get("hierarchy_weight", 1.0)
        )
        core_weight = max(float(probability_params.get("core_weight", 1.0)), 1e-6)
        offset = float(probability_params.get("log_odds_offset", 0.0))
        temperature = max(float(probability_params.get("temperature", 1.0)), 1e-3)
        outer = evidence["outer_log_ratio"]
        core = evidence["core_log_ratio"]
        hierarchy = (
            np.logaddexp(outer, core + np.log(core_weight))
            - np.log1p(core_weight)
        )
        final_log_odds = (
            evidence["tumor_log_odds"]
            + hierarchy_weight * hierarchy
            + offset
        )

        brain_mask = evidence["brain_mask"]
        tumor_probability = np.zeros(brain_mask.shape, dtype=np.float64)
        tumor_probability[brain_mask] = expit(
            final_log_odds[brain_mask] / temperature
        )
        posteriors = np.zeros((*brain_mask.shape, 4), dtype=np.float64)
        posteriors[..., 0] = 1.0 - tumor_probability
        posteriors[..., 1:] = (
            tumor_probability[..., None] * evidence["tumor_conditional"]
        )
        posteriors[~brain_mask] = 0.0
        return posteriors

    def predict(self, data, probability_params):
        evidence = self.prepare(data)
        return self.posteriors_from_evidence(evidence, probability_params)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)
