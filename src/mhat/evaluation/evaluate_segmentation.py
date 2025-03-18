from __future__ import annotations

import csv

import numpy as np

from .compute_iou import compute_iou


def get_matches(gt_mask, pred_mask, threshold):
    """Get matches for one time point.

    Args:
        gt_mask (_type_): _description_
        pred_mask (_type_): _description_
        iou_threshold (_type_): _description_
    """
    ious = compute_iou(gt_mask, pred_mask)
    gt_to_pred = {}
    pred_to_gt = {}
    for gt, pred, iou in ious:
        if iou > threshold:
            gt_matches = gt_to_pred.get(gt, [])
            gt_matches.append(pred)
            gt_to_pred[gt] = gt_matches

            pred_matches = pred_to_gt.get(pred, [])
            pred_matches.append(gt)
            pred_to_gt[pred] = pred_matches
    return gt_to_pred, pred_to_gt


def evaluate_segmentation(
    gt_masks: np.ndarray,
    pred_masks: np.ndarray,
    iou_threshold: float = 0.2,
) -> list[dict]:
    """Match the ground truth and predicted segmentation_metrics with IOU
    greater than iou_threshold. Then compute the following segmentation metrics
    per frame:
        TP - gt masks with one matched prediction mask
        FP - prediction masks with no match
        FN - gt masks with no match
        SPLIT - gt masks with multiple matched prediction masks
        MERGE - prediction masks with multiple matched gt masks

    NOTE: If IOU >=0.5, there will be no splits or merges

    Args:
        gt_masks (np.ndarray): _description_
        pred_masks (np.ndarray): _description_
        iou_threshold (float, optional): _description_. Defaults to 0.2.

    Returns:
        list[dict]: _description_
    """

    results = []
    for t in range(gt_masks.shape[0]):
        gt_to_pred, pred_to_gt = get_matches(gt_masks[t], pred_masks[t], iou_threshold)

        gt_ids = set(np.unique(gt_masks[t])) - {0}
        pred_ids = set(np.unique(pred_masks[t])) - {0}

        gt_tps = []
        pred_tps = []
        gt_fns = []
        pred_fps = []
        gt_splits = []
        pred_splits = []
        gt_merges = []
        pred_merges = []

        for gt in gt_ids:
            if len(gt_to_pred.get(gt, [])) == 0:
                gt_fns.append(gt)
            else:
                preds = gt_to_pred[gt]
                if len(preds) > 1:
                    gt_splits.append(gt)
                    pred_splits.extend(preds)
                else:
                    # add for now, will remove if it is a merge
                    gt_tps.append(gt)

        for pred in pred_ids:
            if len(pred_to_gt.get(pred, [])) == 0:
                pred_fps.append(pred)
            else:
                gts = pred_to_gt[pred]
                if len(gts) > 1:
                    pred_merges.append(pred)
                    gt_merges.extend(gts)
                    for gt in gts:
                        if gt in gt_tps:
                            gt_tps.remove(gt)
                else:
                    if pred not in pred_splits:
                        pred_tps.append(pred)

        # sanity check
        assert len(gt_tps) == len(pred_tps)
        # correct items, counting each merge or split as one correct match
        accuracy_numerator = len(gt_tps) + len(pred_merges) + len(gt_splits)
        # total items, counting each merge or split as multiple
        accuracy_denominator = (
            len(gt_tps)
            + len(gt_merges)
            + len(pred_splits)
            + len(pred_fps)
            + len(gt_fns)
        )
        results.append(
            {
                "frame": t,
                "TP": len(gt_tps),
                "FP": len(pred_fps),
                "FN": len(gt_fns),
                "merge": len(gt_merges),  # number of GT items merged
                "split": len(gt_splits),  # number of GT items split
                "GT": len(gt_ids),
                "accuracy": accuracy_numerator / accuracy_denominator,
            }
        )

    return results


def save_seg_results(seg_results, outfile):
    fieldnames = ["frame", "TP", "FP", "FN", "merge", "split", "GT", "accuracy"]
    sum_row = {"frame": "all"}
    for field in fieldnames:
        sum_row[field] = 0
    with open(outfile, "w") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for frame_result in seg_results:
            writer.writerow(frame_result)
            for field in fieldnames:
                sum_row[field] += frame_result[field]
        writer.writerow(sum_row)
