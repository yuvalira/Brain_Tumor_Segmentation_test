# Authors: Yuval Ratzabi (ID: TODO), second student (ID: TODO)
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import pearsonr

from config import MODALITY_NAMES


def plot_metric_boxplots(per_model_frames, output_path=None):
    frame = pd.concat(per_model_frames, ignore_index=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    for axis, metric, label in zip(axes, ("dice", "iou"), ("Dice", "IoU")):
        sns.boxplot(data=frame, x="model", y=metric, ax=axis)
        axis.set_title(f"{label} distribution on the test set")
        axis.set_xlabel("")
        axis.set_ylabel(label)
        axis.tick_params(axis="x", rotation=20)
        axis.grid(alpha=0.25)
    figure.tight_layout()
    if output_path:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
    return figure


def plot_required_scatterplots(baseline_frame, proposed_frame, output_path=None):
    merged = baseline_frame.merge(
        proposed_frame, on="volume_id", suffixes=("_baseline", "_proposed")
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 5))
    for axis, metric, label in zip(axes, ("dice", "iou"), ("Dice", "IoU")):
        x = merged[f"{metric}_baseline"]
        y = merged[f"{metric}_proposed"]
        correlation = pearsonr(x, y).statistic if len(x) > 1 else np.nan
        axis.scatter(x, y, alpha=0.75)
        axis.plot([0, 1], [0, 1], "r--", linewidth=1)
        axis.set(
            xlim=(-0.03, 1.03),
            ylim=(-0.03, 1.03),
            xlabel=f"Baseline {label}",
            ylabel=f"Proposed {label}",
            title=f"Baseline vs proposed {label}\nPearson r = {correlation:.3f}",
        )
        axis.grid(alpha=0.25)
    figure.tight_layout()
    if output_path:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
    return figure


def choose_qualitative_examples(baseline_frame, proposed_frame):
    merged = baseline_frame.merge(
        proposed_frame, on="volume_id", suffixes=("_baseline", "_proposed")
    )
    tumor = merged[merged["tumor_present_baseline"]].copy()
    selected = {}
    used = set()

    def choose(order):
        for volume_id in order:
            if int(volume_id) not in used:
                used.add(int(volume_id))
                return int(volume_id)
        return None

    selected["Both performed well"] = choose(
        tumor.assign(score=tumor[["dice_baseline", "dice_proposed"]].min(axis=1))
        .sort_values("score", ascending=False)["volume_id"]
    )
    selected["Both performed poorly"] = choose(
        tumor.assign(score=tumor[["dice_baseline", "dice_proposed"]].max(axis=1))
        .sort_values("score")["volume_id"]
    )
    baseline_difference = tumor.assign(
        difference=tumor["dice_baseline"] - tumor["dice_proposed"]
    ).sort_values("difference", ascending=False)
    if baseline_difference.iloc[0]["difference"] > 0:
        selected["Baseline performed better"] = choose(
            baseline_difference["volume_id"]
        )
    else:
        selected["Baseline performed better"] = None

    proposed_difference = tumor.assign(
        difference=tumor["dice_proposed"] - tumor["dice_baseline"]
    ).sort_values("difference", ascending=False)
    if proposed_difference.iloc[0]["difference"] > 0:
        selected["Proposed model performed better"] = choose(
            proposed_difference["volume_id"]
        )
    else:
        selected["Proposed model performed better"] = None
    return selected


def plot_qualitative_examples(selected, baseline_details, proposed_details, output_path=None):
    available = [(label, volume) for label, volume in selected.items() if volume]
    figure, axes = plt.subplots(
        len(available), 7, figsize=(18, 3.1 * len(available)), squeeze=False
    )
    for row, (label, volume_id) in enumerate(available):
        baseline = baseline_details[volume_id]
        proposed = proposed_details[volume_id]
        data = baseline["data"]
        for modality_index, modality_name in enumerate(MODALITY_NAMES):
            axes[row, modality_index].imshow(
                np.where(data.brain_mask, data.image[..., modality_index], np.nan),
                cmap="gray",
            )
            axes[row, modality_index].set_title(modality_name)
        axes[row, 4].imshow(baseline["prediction"], cmap="gray")
        axes[row, 4].set_title(f"Baseline\nDice={baseline['dice']:.3f}")
        axes[row, 5].imshow(proposed["prediction"], cmap="gray")
        axes[row, 5].set_title(f"Proposed\nDice={proposed['dice']:.3f}")
        axes[row, 6].imshow(data.whole_tumor, cmap="gray")
        axes[row, 6].set_title("Ground truth")
        axes[row, 0].set_ylabel(f"{label}\nVolume {volume_id}")
        for axis in axes[row]:
            axis.axis("off")
    figure.tight_layout()
    if output_path:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
    return figure


def plot_pipeline_diagnostic(details, output_path=None):
    data = details["data"]
    base_posterior = details.get("base_tumor_posterior")
    hierarchy_probability = details.get("hierarchy_probability")
    if base_posterior is None:
        base_posterior = details["tumor_posterior"]
    if hierarchy_probability is None:
        hierarchy_probability = np.zeros(data.brain_mask.shape)
    accepted = (
        np.any(details["accepted_components"], axis=-1)
        if details["accepted_components"].shape[-1]
        else np.zeros(data.brain_mask.shape)
    )
    masks = [
        data.image[..., 3],
        base_posterior,
        hierarchy_probability,
        details["tumor_posterior"],
        details["entropy"],
        details["edge_map"],
        details["candidate_support"],
        accepted,
        details["prediction"],
        data.whole_tumor,
    ]
    titles = [
        "FLAIR",
        "Boundary-symmetry fusion",
        "Hierarchy probability",
        "Guided tumor posterior",
        "Entropy",
        "Sobel diagnostic",
        "Candidate support",
        "Accepted components",
        "Expanded mask",
        "Ground truth",
    ]
    figure, axes = plt.subplots(2, 5, figsize=(17, 7))
    for axis, image, title in zip(axes.flat, masks, titles):
        axis.imshow(image, cmap="gray")
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    if output_path:
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
    return figure
