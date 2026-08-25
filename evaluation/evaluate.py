# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
from pathlib import Path

import numpy as np
import pandas as pd

from data.preprocessing import load_preprocessed_slice
from evaluation.metrics import binary_segmentation_metrics
from image_processing.pipeline import segment_posteriors


def prepare_evidence_cache(model, volume_ids):
    cache = {}
    for volume_id in volume_ids:
        data = load_preprocessed_slice(int(volume_id))
        cache[int(volume_id)] = {
            "data": data,
            "evidence": model.prepare(data),
        }
    return cache


def evaluate_model(
    model,
    volume_ids,
    image_params,
    probability_params,
    evidence_cache=None,
    return_details=False,
):
    rows = []
    details = {}
    for volume_id in map(int, volume_ids):
        if evidence_cache is None:
            data = load_preprocessed_slice(volume_id)
            evidence = model.prepare(data)
        else:
            data = evidence_cache[volume_id]["data"]
            evidence = evidence_cache[volume_id]["evidence"]
        posteriors = model.posteriors_from_evidence(evidence, probability_params)
        guidance = (
            model.segmentation_guidance(evidence, probability_params)
            if hasattr(model, "segmentation_guidance")
            else None
        )
        segmentation = segment_posteriors(
            posteriors,
            data.brain_mask,
            image_params,
            probability_params,
            guidance=guidance,
        )
        metrics = binary_segmentation_metrics(
            segmentation["prediction"], data.whole_tumor
        )
        rows.append({"volume_id": volume_id, "model": model.name, **metrics})
        if return_details:
            details[volume_id] = {
                "data": data,
                **segmentation,
                **metrics,
            }

    frame = pd.DataFrame(rows)
    tumor_rows = frame[frame["tumor_present"]]
    summary = {
        "model": model.name,
        "dice_mean": float(frame["dice"].mean()),
        "dice_std": float(frame["dice"].std(ddof=0)),
        "iou_mean": float(frame["iou"].mean()),
        "iou_std": float(frame["iou"].std(ddof=0)),
        "tumor_present_dice": float(tumor_rows["dice"].mean()),
        "precision": float(tumor_rows["precision"].mean()),
        "recall": float(tumor_rows["recall"].mean()),
        "missed_tumors": int(frame["missed_tumor"].sum()),
        "tumor_slices": int(frame["tumor_present"].sum()),
        "empty_slice_false_positives": int(
            frame["empty_slice_false_positive"].sum()
        ),
        "empty_slices": int((~frame["tumor_present"]).sum()),
    }
    return {
        "summary": summary,
        "per_volume": frame,
        "details": details if return_details else None,
    }


def save_evaluation(result, output_dir, prefix):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result["per_volume"].to_csv(
        output_dir / f"{prefix}_per_volume.csv", index=False
    )
    pd.DataFrame([result["summary"]]).to_csv(
        output_dir / f"{prefix}_summary.csv", index=False
    )
