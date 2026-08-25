import unittest

import numpy as np

from image_processing.component_classification import classify_components

try:
    from image_processing.seed_expansion import expand_component
except ModuleNotFoundError as error:
    if error.name != "cv2":
        raise
    expand_component = None


class ComponentGateTests(unittest.TestCase):
    def _classify(self, tumor_posterior, component, **overrides):
        arguments = {
            "components": component[..., None],
            "tumor_posterior": tumor_posterior,
            "entropy": np.zeros_like(tumor_posterior),
            "brain_mask": np.ones_like(tumor_posterior, dtype=bool),
            "threshold": 0.50,
            "normal_min_component_size": 20,
            "small_min_component_size": 4,
            "small_component_q95_threshold": 0.90,
            "slice_gate_threshold": 0.55,
        }
        arguments.update(overrides)
        return classify_components(**arguments)

    def test_small_component_cannot_approve_weak_slice_by_itself(self):
        posterior = np.full((10, 10), 0.05)
        component = np.zeros((10, 10), dtype=bool)
        component[4:6, 4:6] = True
        posterior[component] = 0.95

        accepted, rows, gate_passed = self._classify(posterior, component)

        self.assertTrue(rows[0]["confident_small_component"])
        self.assertTrue(rows[0]["accepted_before_slice_gate"])
        self.assertFalse(gate_passed)
        self.assertEqual(accepted.shape[-1], 0)

    def test_broad_posterior_evidence_passes_slice_gate(self):
        posterior = np.full((10, 10), 0.05)
        component = np.zeros((10, 10), dtype=bool)
        component[2:7, 2:7] = True
        posterior[component] = 0.80

        accepted, rows, gate_passed = self._classify(posterior, component)

        self.assertTrue(gate_passed)
        self.assertEqual(rows[0]["slice_gate_source"], "posterior")
        self.assertEqual(accepted.shape[-1], 1)

    def test_hierarchy_can_confirm_a_small_component(self):
        posterior = np.full((10, 10), 0.05)
        component = np.zeros((10, 10), dtype=bool)
        component[4:6, 4:6] = True
        posterior[component] = 0.45
        outer = np.zeros_like(posterior)
        core = np.zeros_like(posterior)
        outer[component] = 0.80
        core[component] = 0.70

        accepted, rows, gate_passed = self._classify(
            posterior,
            component,
            outer_probability=outer,
            core_probability=core,
            hierarchy_confirmation_threshold=0.60,
        )

        self.assertTrue(rows[0]["confirmed_by_hierarchy"])
        self.assertTrue(gate_passed)
        self.assertEqual(rows[0]["slice_gate_source"], "hierarchy")
        self.assertEqual(accepted.shape[-1], 1)


class ExpansionTests(unittest.TestCase):
    @unittest.skipIf(expand_component is None, "OpenCV is not installed")
    def test_expansion_stays_inside_local_euclidean_band(self):
        seed = np.zeros((15, 15), dtype=bool)
        seed[7, 7] = True
        posterior = np.ones((15, 15))
        entropy = np.ones((15, 15))
        brain = np.ones((15, 15), dtype=bool)

        expanded = expand_component(
            seed,
            posterior,
            entropy,
            brain,
            entropy_threshold=0.10,
            posterior_threshold=0.20,
            max_expansion_distance=2,
        )

        rows, columns = np.where(expanded)
        distances = np.sqrt((rows - 7) ** 2 + (columns - 7) ** 2)
        self.assertLessEqual(float(distances.max()), 2.0)


if __name__ == "__main__":
    unittest.main()
