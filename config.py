# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = Path(os.getenv(
    "BRATS_DATA_DIR",
    PROJECT_ROOT.parent / "MRI_2026_datasets" / "Brats" /
    "BraTS2020_training_data" / "content" / "data",
))

SLICE_NUM = 80
TOTAL_VOLUMES = 369
RANDOM_SEED = 42
TRAIN_SIZE = 250
VALIDATION_SIZE = 50
TEST_SIZE = 69

MODALITY_NAMES = ("T1", "T1ce", "T2", "FLAIR")
TISSUE_NAMES = ("NCR_NET", "ED", "ET")
HEALTHY_COMPONENTS = 9
TUMOR_COMPONENTS = 4
MAX_HEALTHY_SAMPLES_PER_PATIENT = 12_000
MAX_TUMOR_SAMPLES_PER_CLASS_PER_PATIENT = 4_000
GMM_REG_COVAR = 1e-5
GMM_N_INIT = 3
GMM_MAX_ITER = 200

SYMMETRY_AXIS = 0
SYMMETRY_BLUR_SIGMA = 2.0
HIERARCHY_LOG_RATIO_CLIP = 3.0
HIERARCHY_STRONG_BASE_NEGATIVE_SCALE = 0.25
EPSILON = 1e-8

SPLITS_DIR = PROJECT_ROOT / "splits"
MODEL_DIR = PROJECT_ROOT / "saved_parameters" / "gmm_models"
SELECTED_PARAMETERS_PATH = PROJECT_ROOT / "saved_parameters" / "selected_parameters.json"
OUTPUT_DIR = PROJECT_ROOT / "output"
VALIDATION_OUTPUT_DIR = OUTPUT_DIR / "validation"
TEST_OUTPUT_DIR = OUTPUT_DIR / "test"
FIGURES_DIR = OUTPUT_DIR / "figures"
DIAGNOSTICS_DIR = OUTPUT_DIR / "diagnostics"

MODEL_NAMES = (
    "Raw (4D)",
    "Boundary distance (5D)",
    "Symmetry (8D)",
    "Boundary + Symmetry",
    "Combined",
)

DEFAULT_IMAGE_PROCESSING_PARAMS = {
    "min_component_size": 50,
    "small_min_component_size": 8,
    "closing_size": 3,
    "max_expansion_distance": 16,
}
DEFAULT_PROBABILITY_PARAMS = {
    "log_odds_offset": 0.0,
    "temperature": 1.0,
    "candidate_threshold": 0.25,
    "component_threshold": 0.50,
    "small_component_q95_threshold": 0.90,
    "slice_gate_threshold": 0.55,
    "entropy_expansion_threshold": 0.14,
    "posterior_expansion_threshold": 0.20,
}
