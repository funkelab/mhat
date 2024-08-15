import argparse
import json
from pathlib import Path

import toml
import zarr
from darts_utils.evaluation.evaluate_segmentation import (
    evaluate_segmentation,
    save_seg_results,
)
from darts_utils.evaluation.evaluate_tracking import evaluate_tracking
from darts_utils.tracking.tracks_io import load_tracks_from_csv, read_gt_tracks


def run_evaluation(config, data_dir: Path):
    """
    Args:
        config (_type_): _description_
        data_dir (_type_): directory containing data.zarr and {exp_name}/pred_tracks.csv
    """
    exp_name: str = config["exp_name"]
    zarr_path = data_dir / "data.zarr"
    assert zarr_path.is_dir(), "data.zarr is missing from {zarr_path}"
    zarr_root = zarr.open(zarr_path)
    pred_seg_group = exp_name + "_pred_mask"
    gt_seg_group = "mask"
    for group in [gt_seg_group, pred_seg_group]:
        assert group in zarr_root, f"Group {group} missing from zarr {zarr_path}"

    exp_results_path = data_dir / exp_name
    assert exp_results_path.is_dir(), f"experiment dir {exp_results_path} is missing"
    pred_tracks_csv = exp_results_path / "pred_tracks.csv"
    assert pred_tracks_csv.is_file(), f"Tracks file {pred_tracks_csv} is missing"
    pred_tracks = load_tracks_from_csv(pred_tracks_csv)

    gt_tracks_file = data_dir / "gt_tracks.csv"
    assert gt_tracks_file.is_file(), f"GT tracks file {gt_tracks_file} is missing"
    gt_tracks = read_gt_tracks(zarr_path, gt_tracks_file)

    # evaluate segmentation
    gt_mask = zarr_root[gt_seg_group][:]
    pred_mask = zarr_root[pred_seg_group][:]
    seg_results = evaluate_segmentation(gt_mask, pred_mask, config["iou_threshold"])

    # evaluate tracking
    tracking_results = evaluate_tracking(
        gt_tracks, gt_mask, pred_tracks, pred_mask, config["iou_threshold"]
    )
    return seg_results, tracking_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()

    config = toml.load(args.config)
    input_base_dir = Path(config["input_base_dir"])
    output_base_dir = Path(config["output_base_dir"])
    dataset: str = config["dataset"]
    assert input_base_dir.is_dir()
    assert output_base_dir.is_dir()

    data_dir = input_base_dir / dataset
    assert data_dir.is_dir()

    exp_name: str = config["exp_name"]

    for video_dir in data_dir.iterdir():
        assert video_dir.is_dir()
        vid_name = video_dir.stem
        seg_results, track_results = run_evaluation(config, video_dir)
        out_dir = output_base_dir / exp_name / dataset / vid_name
        out_dir.mkdir(exist_ok=True, parents=True)
        segfile = out_dir / "segmentation_results.csv"
        save_seg_results(seg_results, segfile)
        tracksfile = out_dir / "tracking_metrics.json"
        with open(tracksfile, "w") as f:
            json.dump(track_results, f)
