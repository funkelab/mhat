from traccuracy import TrackingGraph, run_metrics
from traccuracy.matchers import IOUMatcher
from traccuracy.metrics import CTCMetrics, DivisionMetrics


def evaluate_tracking(
    gt_graph, gt_segmentation, pred_graph, pred_segmentation, iou_threshold
):
    """Calculate metrics for linked tracks by comparing to ground truth.

    Args:
        gt_graph (networkx.DiGraph): Ground truth graph.
        labels (np.ndarray): Ground truth detections.
        pred_graph (networkx.DiGraph): Predicted graph.
        pred_segmentation (np.ndarray): Predicted dense segmentation.

    Returns:
        results (dict): Dictionary of metric results.
    """

    gt_graph = TrackingGraph(
        graph=gt_graph,
        frame_key="time",
        label_key="label",
        location_keys=("x", "y"),
        segmentation=gt_segmentation,
    )

    pred_graph = TrackingGraph(
        graph=pred_graph,
        frame_key="time",
        label_key="label",
        location_keys=("x", "y"),
        segmentation=pred_segmentation,
    )

    results = run_metrics(
        gt_data=gt_graph,
        pred_data=pred_graph,
        matcher=IOUMatcher(iou_threshold=iou_threshold, one_to_one=True),
        metrics=[CTCMetrics(), DivisionMetrics(max_frame_buffer=3)],
    )
    results[0]["gt_edges"] = gt_graph.graph.number_of_edges()

    return results
