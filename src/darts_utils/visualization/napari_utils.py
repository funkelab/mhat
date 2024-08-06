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


def view_run(data_path: Path, experiment: str, run_name: str):
    zarr_name = "data.zarr"
    raw_group = "phase"
    gt_seg_group = "mask"
    input_seg_group = "fragments"
    pred_seg_group = f"{experiment}_multihypo_pred_mask"

    gt_tracks_name = "gt_tracks.csv"
    pred_tracks_name = f"{experiment}/multihypo_pred_tracks.csv"

    zarr_root = zarr.open(data_path / zarr_name)
    raw_data = zarr_root[raw_group][:]
    gt_seg = zarr_root[gt_seg_group][:]
    pred_seg = zarr_root[pred_seg_group][:]
    fragments = zarr_root[input_seg_group][:]

    pred_tracks = load_prediction(data_path / pred_tracks_name)
    assign_tracklet_ids(pred_tracks)
    gt_tracks = read_gt_tracks(data_path / zarr_name, data_path / gt_tracks_name)
    assign_tracklet_ids(gt_tracks)

    pred_seg = relabel_segmentation(pred_tracks, pred_seg)

    run = motile_plugin.backend.motile_run.MotileRun(
        run_name=run_name,
        solver_params=None,
        output_segmentation=np.expand_dims(pred_seg, axis=1),
        tracks=pred_tracks,
    )

    viewer = napari.Viewer()
    viewer.add_image(raw_data, name="phase")
    viewer.add_labels(gt_seg, name="gt_seg")
    viewer.add_labels(fragments, name="fragments")
    widget = motile_plugin.widgets.TreeWidget(viewer)
    viewer.window.add_dock_widget(widget, name="Lineage View", area="bottom")

    widget.view_controller.update_napari_layers(run)
    napari.run()
