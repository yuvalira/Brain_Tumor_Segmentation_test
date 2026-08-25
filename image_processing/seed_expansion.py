# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
from collections import deque

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt


NEIGHBORS_8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def expand_component(
    seed,
    tumor_posterior,
    entropy,
    brain_mask,
    entropy_threshold,
    posterior_threshold,
    max_expansion_distance,
    outer_probability=None,
    outer_expansion_threshold=None,
):
    expanded = seed.copy()
    local_band = distance_transform_edt(~seed) <= float(max_expansion_distance)
    distance = np.full(seed.shape, -1, dtype=np.int32)
    eroded = cv2.erode(
        seed.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)),
    ).astype(bool)
    boundary = seed & ~eroded
    queue = deque(zip(*np.where(boundary)))
    distance[boundary] = 0
    height, width = seed.shape

    while queue:
        row, column = queue.popleft()
        current_distance = distance[row, column]
        if current_distance >= max_expansion_distance:
            continue
        for delta_row, delta_column in NEIGHBORS_8:
            new_row, new_column = row + delta_row, column + delta_column
            if not (0 <= new_row < height and 0 <= new_column < width):
                continue
            if expanded[new_row, new_column] or not brain_mask[new_row, new_column]:
                continue
            if not local_band[new_row, new_column]:
                continue
            posterior_growth = (
                entropy[new_row, new_column] >= entropy_threshold
                and tumor_posterior[new_row, new_column] >= posterior_threshold
            )
            outer_growth = (
                outer_probability is not None
                and outer_expansion_threshold is not None
                and outer_probability[new_row, new_column] >= outer_expansion_threshold
                and tumor_posterior[new_row, new_column] >= posterior_threshold
            )
            if posterior_growth or outer_growth:
                expanded[new_row, new_column] = True
                distance[new_row, new_column] = current_distance + 1
                queue.append((new_row, new_column))

    return cv2.morphologyEx(
        expanded.astype(np.uint8),
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ).astype(bool)


def expand_components(
    accepted_components,
    tumor_posterior,
    entropy,
    brain_mask,
    entropy_threshold,
    posterior_threshold,
    max_expansion_distance,
    outer_probability=None,
    outer_expansion_threshold=None,
):
    segmentation = np.zeros(brain_mask.shape, dtype=bool)
    for component_index in range(accepted_components.shape[-1]):
        segmentation |= expand_component(
            accepted_components[..., component_index],
            tumor_posterior,
            entropy,
            brain_mask,
            entropy_threshold,
            posterior_threshold,
            max_expansion_distance,
            outer_probability,
            outer_expansion_threshold,
        )
    return segmentation
