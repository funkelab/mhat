from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import zarr


def _compute_ious(
    frame1: np.ndarray, frame2: np.ndarray
) -> list[tuple[int, int, float]]:
    """Compute label IOUs between two label arrays of the same shape. Ignores background
    (label 0).

    Args:
        frame1 (np.ndarray): Array with integer labels
        frame2 (np.ndarray): Array with integer labels

    Returns:
        list[tuple[int, int, float]]: List of tuples of label in frame 1, label in
            frame 2, and iou values. Labels that have no overlap are not included.
    """
    frame1 = frame1.flatten()
    frame2 = frame2.flatten()
    # get indices where both are not zero (ignore background)
    # this speeds up computation significantly
    non_zero_indices = np.logical_and(frame1, frame2)
    flattened_stacked = np.array([frame1[non_zero_indices], frame2[non_zero_indices]])

    values, counts = np.unique(flattened_stacked, axis=1, return_counts=True)
    frame1_values, frame1_counts = np.unique(frame1, return_counts=True)
    frame1_label_sizes = dict(zip(frame1_values, frame1_counts))
    frame2_values, frame2_counts = np.unique(frame2, return_counts=True)
    frame2_label_sizes = dict(zip(frame2_values, frame2_counts))
    ious: list[tuple[int, int, float]] = []
    for index in range(values.shape[1]):
        pair = values[:, index]
        intersection = counts[index]
        id1, id2 = pair
        union = frame1_label_sizes[id1] + frame2_label_sizes[id2] - intersection
        ious.append((id1, id2, intersection / union))
    return ious


def evaluate_masks(
    csv_filepath,
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
    iou_threshold: float = 0.5,
) -> list[dict]:
    results = []
    for t in range(gt_masks.shape[0]):
        ious = _compute_ious(gt_masks[t], pred_masks[t])
        ious = [iou for iou in ious if iou[2] >= iou_threshold]

        gt_ids = set(np.unique(gt_masks[t])) - {0}
        pred_ids = set(np.unique(pred_masks[t])) - {0}

        matched_gt_ids = {iou[0] for iou in ious}
        matched_pred_ids = {iou[1] for iou in ious}

        false_positives = len(pred_ids - matched_pred_ids)
        false_negatives = len(gt_ids - matched_gt_ids)
        true_positives = len(matched_gt_ids)

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0
        )
        recall = (
            true_positives / (true_positives + false_negatives)
            if (true_positives + false_negatives) > 0
            else 0
        )
        f1_score = (
            2 * (precision * recall) / (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        results.append(
            {
                "time_frame": t,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "false_negatives": false_negatives,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
            }
        )

    outfile = csv_filepath
    fields = [
        "time_frame",
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1_score",
    ]
    with open(outfile, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    return results


if __name__ == "__main__":
    for file_num in range(1, 101):
        zarr_dir = Path(
            f"/nrs/funke/data/darts/synthetic_data/test1/{file_num}/data.zarr"
        )
        root = zarr.open(zarr_dir, "a")
        gt_mask = root["mask"][:]
        pred_mask = root["pred_mask_0.15"][:]
        csv_fp = zarr_dir.parent / "new_evaluation.csv"

        results = evaluate_masks(zarr_dir, csv_fp, gt_mask, pred_mask)

        f1 = [result["f1_score"] for result in results]
        time = [result["time_frame"] for result in results]

        # plt.figure(figsize=(10,6))
        # plt.hist(f1)
        # plt.title(f"F1 Plot of Video {file_num}")
        # plt.xlabel("F1 Score")
        # plt.ylabel("Frequency")

        # save = (zarr_dir.parent / f"f1_histogram_plt_{file_num}.png")
        # plt.savefig(save)
        # plt.close()
