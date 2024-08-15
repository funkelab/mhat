# ---
# jupyter:
#   jupytext:
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: 09-tracking
#     language: python
#     name: python3
# ---

# %%
# %reload_ext autoreload
# %autoreload 2
from pathlib import Path

import pandas as pd
import toml
from darts_utils.evaluation.eval_io import (
    get_cell_lengths,
    load_segmentation_results,
    load_tracking_results,
)
from itables import init_notebook_mode

# %%
init_notebook_mode(all_interactive=True)


# %%
config: dict[str, str] = toml.load("eval_config.toml")
exp_name = config["exp_name"]
config

# %%
results_base_dir = Path(config["output_base_dir"]) / exp_name
ds_dir = results_base_dir / config["dataset"]

# %%
per_frame_seg_results = load_segmentation_results(ds_dir)
per_frame_seg_results

# %%
per_video_seg_results = (
    per_frame_seg_results.groupby("video", as_index=True).sum().drop(columns=["frame"])
)
per_video_seg_results

# %%
per_video_seg_results["perc_correct"] = (
    per_video_seg_results["TP"] / per_video_seg_results["GT"]
)
per_video_seg_results

# %%
input_base_dir = Path(config["input_base_dir"]) / config["dataset"]
cell_lengths_dict = get_cell_lengths(input_base_dir)
per_video_seg_results["max_cell_length"] = pd.Series(cell_lengths_dict)
per_video_seg_results

# %%
from darts_utils.visualization.napari_utils import view_run

# %%
video = 9
vid_dir = input_base_dir / str(video)
view_run(vid_dir, exp_name, "with_area")

# %%
tracking_results = load_tracking_results(ds_dir)
tracking_results["1"]
ctc_columns = ["fp_nodes", "fn_nodes", "fp_edges", "fn_edges", "TRA"]
div_columns = [
    "True Positive Divisions",
    "False Positive Divisions",
    "False Negative Divisions",
    "Division F1",
]
div_columns_renamed = ["tp_div", "fp_div", "fn_div", "div_f1"]

tracking_results_filtered = {}
tracking_results_filtered["video"] = list(tracking_results.keys())

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
tracking_results_filtered

# %%
tracking_df = pd.DataFrame(tracking_results_filtered)
tracking_df
