import argparse
import csv
import datetime
from pathlib import Path

import motile
import numpy as np
import toml
import zarr
from darts_utils.tracking import create_multihypo_graph, solve_with_motile, utils


def save_solution_graph(solution_graph, csv_path):
    with open(csv_path, "w") as f:
        writer = csv.DictWriter(f, fieldnames=["time", "x", "y", "id", "parent_id"])
        writer.writeheader()
        for node, data in solution_graph.nodes(data=True):
            parents = list(solution_graph.predecessors(node))
            if len(parents) == 1:
                parent_id = parents[0]
            elif len(parents) == 0:
                parent_id = -1
            else:
                raise ValueError(f"Node {node} has too many parents! {parents}")
            row = {
                "time": data["time"],
                "x": data["x"],
                "y": data["y"],
                "id": node,
                "parent_id": parent_id,
            }
            writer.writerow(row)


def get_solution_seg(fragments, merge_history, solution_graph):
    solution_seg = np.zeros_like(fragments)

    merge_dict = {}
    for merge in merge_history:
        a, b, c, score = merge
        a = int(a)
        b = int(b)
        c = int(c)
        children = [a, b]
        if a in merge_dict:
            children.extend(merge_dict[a])
        if b in merge_dict:
            children.extend(merge_dict[b])
        merge_dict[c] = children

    frag_ids = set(np.unique(fragments))
    frag_ids.remove(0)

    for node in solution_graph.nodes():
        if node in merge_dict:
            children = merge_dict[node]
        else:
            assert node in frag_ids, f"Node {node} not in merge dict or frag ids"
            children = [node]

        for child in children:
            assert np.all(
                [solution_seg[fragments == child] == 0]
            ), f"Child {child} fragment is already selected"
            solution_seg[fragments == child] = node

    return solution_seg


def run_tracking(config, video_base_path: Path):
    current_datetime = datetime.datetime.now()
    datetime_str = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")
    print(datetime_str)
    base_path = Path(video_base_path)
    exp_path = base_path / datetime_str
    exp_path.mkdir()
    zarr_path = base_path / "data.zarr"
    merge_history_csv_path = base_path / "merge_history.csv"
    config_filepath = exp_path / "config.toml"
    output_filepath = exp_path / "pred_tracks.csv"

    with open(config_filepath, "w") as config_file:
        toml.dump(config, config_file)

    seg_group = "fragments"
    output_seg_group = f"{datetime_str}_pred_mask"

    max_edge_distance = config["max_edge_distance"]

    zarr_root = zarr.open(zarr_path)
    fragments = zarr_root[seg_group][:]
    max_node_id = np.max(fragments)

    merge_history = create_multihypo_graph.load_merge_history(merge_history_csv_path)
    merge_history = create_multihypo_graph.renumber_merge_history(
        merge_history, max_node_id
    )
    cand_graph, exclusion_sets = create_multihypo_graph.get_nodes(
        fragments,
        merge_history,
        min_score=config["min_merge_score"],
        max_score=config["max_merge_score"],
    )

    utils.add_cand_edges(cand_graph, max_edge_distance)
    print("Edges before hyperedges: ", cand_graph.number_of_edges())
    cand_graph = utils.add_hyper_elements(cand_graph)
    print("Edges after hyperedges: ", cand_graph.number_of_edges())
    utils.add_appear_ignore_attr(cand_graph)
    utils.add_disappear(cand_graph)
    track_graph = motile.TrackGraph(cand_graph, frame_attribute="time")
    utils.add_drift_dist_attr(track_graph, drift=config["drift"])
    utils.add_area_diff_attr(track_graph)

    solution_graph = solve_with_motile(config, track_graph, exclusion_sets)

    save_solution_graph(solution_graph, output_filepath)
    solution_seg = get_solution_seg(fragments, merge_history, solution_graph)
    zarr_root[output_seg_group] = solution_seg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument(
        "data_dir",
        help="directory containing the data.zarr and other dataset specific files",
    )
    args = parser.parse_args()
    config = toml.load(args.config)

    run_tracking(config, args.data_dir)
