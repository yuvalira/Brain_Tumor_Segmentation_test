# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
import cv2
import numpy as np
from scipy.ndimage import sobel


def posterior_edge_map(tumor_posterior, brain_mask):
    gradient_x = sobel(tumor_posterior, axis=0)
    gradient_y = sobel(tumor_posterior, axis=1)
    edges = np.hypot(gradient_x, gradient_y)
    edges[~brain_mask] = 0.0
    return edges


def extract_candidate_components(
    tumor_posterior,
    brain_mask,
    candidate_threshold,
    small_min_component_size,
    closing_size,
):
    """Extract even small candidates; confidence is assessed downstream."""
    support = (tumor_posterior >= candidate_threshold) & brain_mask
    closing_size = max(int(closing_size), 1)
    if closing_size > 1:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (closing_size, closing_size)
        )
        support = cv2.morphologyEx(
            support.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        ).astype(bool)

    number, labels, statistics, _ = cv2.connectedComponentsWithStats(
        support.astype(np.uint8), connectivity=8
    )
    components = []
    for label_index in range(1, number):
        area = int(statistics[label_index, cv2.CC_STAT_AREA])
        if area >= int(small_min_component_size):
            components.append(labels == label_index)

    if not components:
        return np.zeros((*support.shape, 0), dtype=bool), support
    return np.dstack(components), support
