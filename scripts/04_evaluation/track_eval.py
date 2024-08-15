import csv
import json
from pathlib import Path

import networkx as nx
import traccuracy
import zarr
from darts_utils.tracking import utils
from evaluate import evaluate_masks
from traccuracy import run_metrics
from traccuracy.matchers import IOUMatcher
from traccuracy.metrics import CTCMetrics, DivisionMetrics


def get_metrics(gt_graph, labels, pred_graph, pred_segmentation):
    """Calculate metrics for linked tracks by comparing to ground truth.

    Args:
        gt_graph (networkx.DiGraph): Ground truth graph.
        labels (np.ndarray): Ground truth detections.
        pred_graph (networkx.DiGraph): Predicted graph.
        pred_segmentation (np.ndarray): Predicted dense segmentation.

    Returns:
        results (dict): Dictionary of metric results.
    """

    gt_graph = traccuracy.TrackingGraph(
        graph=gt_graph,
        frame_key="time",
        label_key="label",
        location_keys=("x", "y"),
        segmentation=labels,
    )

    pred_graph = traccuracy.TrackingGraph(
        graph=pred_graph,
        frame_key="time",
        label_key="label",
        location_keys=("x", "y"),
        segmentation=pred_segmentation,
    )

    results = run_metrics(
        gt_data=gt_graph,
        pred_data=pred_graph,
        matcher=IOUMatcher(iou_threshold=0.5),
        metrics=[CTCMetrics(), DivisionMetrics(max_frame_buffer=2)],
    )

    return results


def load_prediction(csv_path):
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        graph = nx.DiGraph()
        for row in reader:
            node_id = int(float(row["id"]))
            attrs = {
                "time": int(float(row["time"])),
                "x": int(float(row["x"])),
                "y": int(float(row["y"])),
                "label": node_id,
            }
            parent_id = int(float(row["parent_id"]))
            graph.add_node(node_id, **attrs)
            if parent_id != -1:
                graph.add_edge(parent_id, node_id)
    return graph


if __name__ in "__main__":
    dt = "2024-08-09_15-40-36"

    for vid_num in range(1, 101):
        base_path = Path(f"/nrs/funke/data/darts/synthetic_data/test1/{vid_num}")
        dt_path = Path(f"/nrs/funke/data/darts/synthetic_data/test1/{vid_num}/{dt}")
        zarr_path = base_path / "data.zarr"
        gt_csv_path = base_path / "gt_tracks.csv"
        output_csv_path = dt_path / "multihypo_pred_tracks.csv"

        zarr_root = zarr.open(zarr_path)
        gt_labels = zarr_root["mask"][:]

        hypo_mask = zarr_root[f"{dt}_multihypo_pred_mask"][:]

        # run evaluation for the hypo mask and gt mask
        csv_filepath = dt_path / "mask_eval.csv"
        results = evaluate_masks(csv_filepath, gt_labels, hypo_mask)

        gt_tracks = utils.read_gt_tracks(zarr_path, gt_csv_path)
        pred_tracks = load_prediction(output_csv_path)

        eval = get_metrics(gt_tracks, gt_labels, pred_tracks, hypo_mask)
        output_file = dt_path / "track_metrics.json"
        with open(output_file, "w") as f:
            json.dump(eval, f, indent=4)
