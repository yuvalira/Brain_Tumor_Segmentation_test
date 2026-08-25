# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
import numpy as np


def binary_segmentation_metrics(prediction, ground_truth):
    prediction = np.asarray(prediction, dtype=bool)
    ground_truth = np.asarray(ground_truth, dtype=bool)
    intersection = int(np.logical_and(prediction, ground_truth).sum())
    union = int(np.logical_or(prediction, ground_truth).sum())
    prediction_size = int(prediction.sum())
    ground_truth_size = int(ground_truth.sum())

    if prediction_size == 0 and ground_truth_size == 0:
        dice = iou = precision = recall = 1.0
    else:
        dice = (
            2.0 * intersection / (prediction_size + ground_truth_size)
            if prediction_size + ground_truth_size else 0.0
        )
        iou = intersection / union if union else 0.0
        precision = intersection / prediction_size if prediction_size else 0.0
        recall = intersection / ground_truth_size if ground_truth_size else 0.0

    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "intersection": intersection,
        "prediction_size": prediction_size,
        "ground_truth_size": ground_truth_size,
        "tumor_present": bool(ground_truth_size > 0),
        "missed_tumor": bool(ground_truth_size > 0 and intersection == 0),
        "empty_slice_false_positive": bool(
            ground_truth_size == 0 and prediction_size > 0
        ),
    }
