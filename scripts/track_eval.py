import traccuracy
from traccuracy import run_metrics
from traccuracy.metrics import CTCMetrics, DivisionMetrics
from traccuracy.matchers import CTCMatcher
from darts_utils.tracking import utils
import zarr
import csv
import networkx as nx
from pathlib import Path

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
        matcher=CTCMatcher(),
        metrics=[CTCMetrics(), DivisionMetrics()],
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
                "label": node_id
            }
            parent_id = int(float(row["parent_id"]))
            graph.add_node(node_id, **attrs)
            if parent_id != -1:
                graph.add_edge(parent_id, node_id)
    return graph

if __name__ in "__main__":
    vid_num = 5

    base_path = Path(f"/nrs/funke/data/darts/synthetic_data/test1/{vid_num}")
    zarr_path = base_path / "data.zarr"
    gt_csv_path = base_path / "gt_tracks.csv"
    output_csv_path = base_path / "pred_tracks.csv"

    zarr_root = zarr.open(zarr_path)
    seg_mask = zarr_root["pred_mask_0.15"][:]
    gt_labels = zarr_root["mask"][:]


    gt_tracks = utils.read_gt_tracks(zarr_path, gt_csv_path)
    pred_tracks = load_prediction(output_csv_path)

    eval = get_metrics(gt_tracks, gt_labels, pred_tracks, seg_mask)
    print(eval)