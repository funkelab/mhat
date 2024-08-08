import csv
import toml
#from configs import config
from pathlib import Path
import os
import datetime
import argparse
import zarr
from darts_utils.tracking import solve_with_motile, utils
from darts_utils.tracking import create_multihypo_graph
import numpy as np
import motile
from motile_toolbox.candidate_graph import graph_to_nx


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
        a,b, c, score = merge
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
            assert np.all([solution_seg[fragments == child] == 0])
            solution_seg[fragments == child] = node

    return solution_seg
        



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    config = toml.load(args.config)

    current_datetime = datetime.datetime.now()
    datetime_str = current_datetime.strftime('%Y-%m-%d_%H-%M-%S')

    for vid_num in range(1, 3):

        base_path = Path(f"/nrs/funke/data/darts/synthetic_data/validation1/{vid_num}")
        make_path = os.mkdir(f"/nrs/funke/data/darts/synthetic_data/validation1/{vid_num}/{datetime_str}")
        dt_base_path = Path(f"/nrs/funke/data/darts/synthetic_data/validation1/{vid_num}/{datetime_str}")
        zarr_path = base_path / "data.zarr"
        gt_csv_path = base_path / "gt_tracks.csv"
        merge_history_csv_path = base_path / "merge_history.csv"
        config_filepath = dt_base_path / f"config.toml"
        output_filepath = dt_base_path / f"multihypo_pred_tracks.csv"
        #output_csv_path = base_path / "multihypo_pred_tracks.csv"

        with open(config_filepath, "w") as config_file:
            config = toml.load(args.config)
            toml.dump(config, config_file)

        seg_group = "fragments"
        #output_seg_group = "multihypo_pred_mask"
        seg_group_dt = f"{datetime_str}_multihypo_pred_mask"
        
        max_edge_distance = 50

        zarr_root = zarr.open(zarr_path)
        fragments = zarr_root[seg_group][:]
        max_node_id = np.max(fragments)

        gt_tracks = utils.read_gt_tracks(zarr_path, gt_csv_path)
        merge_history = create_multihypo_graph.load_merge_history(merge_history_csv_path)
        merge_history = create_multihypo_graph.renumber_merge_history(merge_history, max_node_id)
        cand_graph, exclusion_sets = create_multihypo_graph.get_nodes(
            fragments, merge_history, min_score=0.1, max_score=0.3
            )
        cand_graph = utils.add_hyper_elements(cand_graph)

        utils.add_cand_edges(cand_graph, max_edge_distance)
        utils.add_appear_ignore_attr(cand_graph)
        utils.add_disappear(cand_graph)
        track_graph = motile.TrackGraph(cand_graph, frame_attribute="time")
        utils.add_drift_dist_attr(track_graph)

        solution_graph = solve_with_motile(config, track_graph, exclusion_sets)
        # print(
        #     f"Our gt graph has {gt_tracks.number_of_nodes()} nodes and {gt_tracks.number_of_edges()} edges"
        # )
        # print(
        #     f"Our solution graph has {solution_graph.number_of_nodes()} nodes and {solution_graph.number_of_edges()} edges"
        # )
        # print(
        #     f"Candidate graph has {cand_graph.number_of_nodes()} nodes and {cand_graph.number_of_edges()} edges"
        # )


        save_solution_graph(solution_graph, output_filepath)
        solution_seg = get_solution_seg(fragments, merge_history, solution_graph)
        zarr_root[seg_group_dt] = solution_seg
