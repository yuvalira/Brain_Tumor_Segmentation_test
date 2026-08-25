# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
from dataclasses import dataclass

import joblib
import numpy as np
from scipy.special import expit, logsumexp
from sklearn.mixture import GaussianMixture

from config import (
    GMM_MAX_ITER, GMM_N_INIT, GMM_REG_COVAR, RANDOM_SEED,
)
from data.preprocessing import build_features


@dataclass
class LogSpaceGMMClassifier:
    gmms: list
    priors: np.ndarray
    feature_kind: str
    name: str

    @classmethod
    def fit(
        cls,
        samples,
        priors,
        component_counts,
        feature_kind,
        name,
        seed=RANDOM_SEED,
    ):
        gmms = []
        for class_index, (class_samples, components) in enumerate(
            zip(samples, component_counts)
        ):
            if len(class_samples) < components:
                raise ValueError(
                    f"Class {class_index} has {len(class_samples)} samples "
                    f"but requires {components} components."
                )
            model = GaussianMixture(
                n_components=components,
                covariance_type="full",
                reg_covar=GMM_REG_COVAR,
                max_iter=GMM_MAX_ITER,
                n_init=GMM_N_INIT,
                init_params="kmeans",
                random_state=seed + class_index,
            )
            model.fit(class_samples)
            gmms.append(model)
        return cls(gmms, np.asarray(priors, dtype=np.float64), feature_kind, name)

    def prepare(self, data):
        features = build_features(data, self.feature_kind)
        height, width, dimensions = features.shape
        brain_values = features[data.brain_mask].reshape(-1, dimensions)
        log_likelihood = np.column_stack([
            gmm.score_samples(brain_values) for gmm in self.gmms
        ])
        log_priors = np.log(np.clip(self.priors, 1e-12, 1.0))
        log_joint = log_likelihood + log_priors
        tumor_joint = logsumexp(log_joint[:, 1:], axis=1)
        tumor_log_odds = tumor_joint - log_joint[:, 0]
        tumor_conditional = np.exp(
            log_joint[:, 1:] - logsumexp(log_joint[:, 1:], axis=1, keepdims=True)
        )

        shape = (height, width)
        odds_map = np.zeros(shape, dtype=np.float64)
        conditional_map = np.zeros((*shape, 3), dtype=np.float64)
        odds_map[data.brain_mask] = tumor_log_odds
        conditional_map[data.brain_mask] = tumor_conditional
        return {
            "brain_mask": data.brain_mask,
            "tumor_log_odds": odds_map,
            "tumor_conditional": conditional_map,
        }

    def posteriors_from_evidence(self, evidence, probability_params):
        offset = float(probability_params.get("log_odds_offset", 0.0))
        temperature = max(float(probability_params.get("temperature", 1.0)), 1e-3)
        brain_mask = evidence["brain_mask"]
        tumor_probability = np.zeros(brain_mask.shape, dtype=np.float64)
        tumor_probability[brain_mask] = expit(
            (evidence["tumor_log_odds"][brain_mask] + offset) / temperature
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
