import json
import re
from pathlib import Path

import pandas as pd
import zarr


def check_video_dir(dir: Path) -> bool:
    stem = dir.stem
    vid_re = re.compile("\d+")
    return vid_re.match(stem) is not None


def load_segmentation_results(
    ds_results_dir: Path, exp_name: str
) -> pd.DataFrame | None:
    per_frame_seg_results = None
    for video_dir in ds_results_dir.iterdir():
        if check_video_dir(video_dir):
            vid_name = video_dir.stem
            vid_df = pd.read_csv(video_dir / exp_name / "segmentation_metrics.csv")
            vid_df.drop(vid_df.tail(1).index, inplace=True)  # drop last row
            vid_df["video"] = vid_name
            if per_frame_seg_results is None:
                per_frame_seg_results = vid_df
            else:
                per_frame_seg_results = pd.concat([per_frame_seg_results, vid_df])
    return per_frame_seg_results


def load_tracking_results(ds_results_dir: Path, exp_name: str) -> pd.DataFrame:
    results = {}
    for video_dir in ds_results_dir.iterdir():
        if check_video_dir(video_dir):
            vid_name = video_dir.stem
            res_file = video_dir / exp_name / "tracking_metrics.json"
            assert res_file.is_file()
            with open(res_file) as f:
                tracking_results = json.load(f)
            results[vid_name] = tracking_results

    return convert_tracking_results_to_df(results)


def convert_tracking_results_to_df(tracking_results) -> pd.DataFrame:
    for vid, result in tracking_results.items():
        print(result)
    ctc_columns = ["fp_edges", "fn_edges", "TRA"]
    div_columns = [
        "True Positive Divisions",
        "False Positive Divisions",
        "False Negative Divisions",
        "Division F1",
    ]
    div_columns_renamed = ["tp_div", "fp_div", "fn_div", "div_f1"]

    tracking_results_filtered = {}
    tracking_results_filtered["video"] = list(tracking_results.keys())

    gt_edges = [
        tracking_results[video][0]["gt_edges"]
        for video in tracking_results_filtered["video"]
    ]
    tracking_results_filtered["gt_edges"] = gt_edges

    for column in ctc_columns:
        vals = [results[0]["results"][column] for results in tracking_results.values()]
        tracking_results_filtered[column] = vals

    for buffer in [0, 1, 2, 3]:
        for column, new_column in zip(div_columns, div_columns_renamed):
            vals = [
                results[1]["results"][f"Frame Buffer {buffer}"][column]
                for results in tracking_results.values()
            ]
            tracking_results_filtered[f"{new_column}_fb{buffer}"] = vals
    return pd.DataFrame(tracking_results_filtered)


def get_cell_lengths(ds_input_dir: Path) -> pd.Series:
    cell_lengths = {}
    for subdir in ds_input_dir.iterdir():
        if check_video_dir(subdir):
            vid_name = subdir.stem
            zarr_path = subdir / "data.zarr"
            assert zarr_path.is_dir(), f"zarr {zarr_path} does not exist"
            zarr_root = zarr.open(zarr_path)
            cell_lengths[vid_name] = zarr_root.attrs["simulation"]["cell_max_length"]
    return pd.Series(cell_lengths)
