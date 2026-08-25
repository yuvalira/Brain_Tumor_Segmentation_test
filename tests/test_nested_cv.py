import unittest
from unittest.mock import patch

import numpy as np

from data.splits import _nested_fold_indices, _validate_nested_folds
from evaluation.cross_validation import _select_combined_or_fallback
from models.fusion_model import ProtectedHierarchicalFusionModel


class NestedSplitTests(unittest.TestCase):
    def test_outer_test_folds_are_disjoint_and_cover_every_patient(self):
        volume_ids = np.arange(1, 101)
        strata = np.asarray([f"stratum_{index % 4}" for index in range(100)])

        folds = _nested_fold_indices(volume_ids, strata)
        _validate_nested_folds(folds, volume_ids)

        outer_ids = np.concatenate([fold["test"] for fold in folds])
        self.assertEqual(len(folds), 5)
        self.assertEqual(len(np.unique(outer_ids)), len(volume_ids))
        self.assertEqual(set(outer_ids), set(volume_ids))

    def test_train_validation_and_test_are_disjoint(self):
        volume_ids = np.arange(1, 101)
        strata = np.asarray([f"stratum_{index % 4}" for index in range(100)])

        for fold in _nested_fold_indices(volume_ids, strata):
            train = set(fold["train"])
            validation = set(fold["validation"])
            test = set(fold["test"])
            self.assertFalse(train & validation)
            self.assertFalse(train & test)
            self.assertFalse(validation & test)


class HierarchyFallbackTests(unittest.TestCase):
    def test_disabled_hierarchy_returns_no_guidance(self):
        guidance = ProtectedHierarchicalFusionModel.segmentation_guidance(
            None, {}, {"use_hierarchy": False}
        )
        self.assertIsNone(guidance)

    @patch("evaluation.cross_validation.evaluate_model")
    def test_combined_falls_back_when_hierarchy_does_not_improve_validation(
        self, evaluate_model
    ):
        evaluate_model.return_value = {"summary": {"dice_mean": 0.80}}
        params = {
            "Boundary + Symmetry": {"fusion_weight": 0.4},
            "Combined": {"fusion_weight": 0.7},
        }
        results = {
            "Boundary + Symmetry": {"summary": {"dice_mean": 0.80}},
            "Combined": {"summary": {"dice_mean": 0.79}},
        }

        _, selected, _ = _select_combined_or_fallback(
            {"Combined": object()}, [], {}, params, results
        )

        self.assertFalse(selected)
        self.assertEqual(params["Combined"]["fusion_weight"], 0.4)
        self.assertFalse(params["Combined"]["use_hierarchy"])

    def test_combined_keeps_hierarchy_after_meaningful_validation_gain(self):
        params = {
            "Boundary + Symmetry": {"fusion_weight": 0.4},
            "Combined": {"fusion_weight": 0.7},
        }
        results = {
            "Boundary + Symmetry": {"summary": {"dice_mean": 0.80}},
            "Combined": {"summary": {"dice_mean": 0.81}},
        }

        _, selected, _ = _select_combined_or_fallback(
            {}, [], {}, params, results
        )

        self.assertTrue(selected)
        self.assertTrue(params["Combined"]["use_hierarchy"])


if __name__ == "__main__":
    unittest.main()
