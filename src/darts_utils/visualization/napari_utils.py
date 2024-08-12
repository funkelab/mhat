from pathlib import Path

import motile_plugin
import napari
import numpy as np
import zarr
from darts_utils.tracking.utils import (
    load_prediction,
    read_gt_tracks,
    relabel_segmentation,
)
from motile_toolbox.visualization.napari_utils import assign_tracklet_ids


def crop_tracks(tracks, start_frame, end_frame, frame_attr="time"):
    if start_frame is None:
        if end_frame is None:
            return tracks
        start_frame = 0
    if end_frame is None:
        end_frame = max([data[frame_attr] for node, data in tracks.nodes(data=True)])
    keep_nodes = [
        node
        for node, data in tracks.nodes(data=True)
        if data[frame_attr] >= start_frame and data[frame_attr] < end_frame
    ]
    subgraph = tracks.subgraph(keep_nodes)
    if start_frame is not None:
        for node in subgraph.nodes():
            subgraph.nodes[node][frame_attr] -= start_frame
    return subgraph


def view_run(
    data_path: Path, experiment: str, run_name: str, start_frame=None, end_frame=None
):
    zarr_name = "data.zarr"
    raw_group = "phase"
    gt_seg_group = "mask"
    input_seg_group = "fragments"
    pred_seg_group = f"{experiment}_pred_mask"

    gt_tracks_name = "gt_tracks.csv"
    pred_tracks_name = f"{experiment}/pred_tracks.csv"

    zarr_root = zarr.open(data_path / zarr_name)
    raw_data = zarr_root[raw_group][start_frame:end_frame]
    pred_seg = zarr_root[pred_seg_group][start_frame:end_frame]
    fragments = zarr_root[input_seg_group][start_frame:end_frame]

    pred_tracks = load_prediction(data_path / pred_tracks_name)
    pred_tracks = crop_tracks(pred_tracks, start_frame, end_frame)
    assign_tracklet_ids(pred_tracks)

    pred_seg = relabel_segmentation(pred_tracks, pred_seg)

    run = motile_plugin.backend.motile_run.MotileRun(
        run_name=run_name,
        solver_params=None,
        output_segmentation=np.expand_dims(pred_seg, axis=1),
        tracks=pred_tracks,
    )

    viewer = napari.Viewer()
    viewer.add_image(raw_data, name="phase")
    viewer.add_labels(fragments, name="fragments")

    if gt_seg_group in zarr_root:
        gt_seg = zarr_root[gt_seg_group][start_frame:end_frame]
        gt_tracks = read_gt_tracks(data_path / zarr_name, data_path / gt_tracks_name)
        gt_tracks = crop_tracks(gt_tracks, start_frame, end_frame)
        assign_tracklet_ids(gt_tracks)
        viewer.add_labels(gt_seg, name="gt_seg")

    widget = motile_plugin.widgets.TreeWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Lineage View", area="bottom")
    widget.view_controller.update_napari_layers(run)
    napari.run()
