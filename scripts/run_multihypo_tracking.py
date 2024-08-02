import csv
from pathlib import Path

import zarr
from darts_utils.tracking import utils
from darts_utils.tracking import create_multihypo_graph
import numpy as np
import motile
from motile_toolbox.candidate_graph import graph_to_nx

def solve_with_motile(graph, exclusion_sets):
    """Set up and solve the network flow problem.

    Args:
        graph (motile.TrackGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """
    cand_trackgraph = motile.TrackGraph(graph, frame_attribute="time")
    solver = motile.Solver(cand_trackgraph)

    solver.add_cost(motile.costs.EdgeSelection(weight=1.0, attribute="drift_dist", constant = -50.0))

    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(2))

    solver.add_cost(motile.costs.NodeSelection(weight=-5, attribute="cohesion", constant=2), name="cohesion")
    solver.add_cost(motile.costs.NodeSelection(
        weight=-5, attribute="adhesion", constant=2,
    ), name="adhesion")
    
    solver.add_cost(motile.costs.Appear(constant=50.0, ignore_attribute = "ignore_appear"))
    solver.add_cost(motile.costs.Disappear(constant=1000.0, ignore_attribute = "ignore_disappear"))

    solver.add_constraint(motile.constraints.ExclusiveNodes(exclusion_sets))

    solver.solve()
    solution_graph = graph_to_nx(solver.get_selected_subgraph())
    return solution_graph


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
                "x": data["pos"][0],
                "y": data["pos"][1],
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
    vid_num = 2

    base_path = Path(f"/Volumes/funke/data/darts/synthetic_data/validation1/{vid_num}")
    zarr_path = base_path / "data.zarr"
    gt_csv_path = base_path / "gt_tracks.csv"
    merge_history_csv_path = base_path / "merge_history.csv"
    output_csv_path = base_path / "multihypo_pred_tracks.csv"

    seg_group = "fragments"
    output_seg_group = "multihypo_pred_mask"
    
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

    utils.add_cand_edges(cand_graph, max_edge_distance)
    utils.add_appear_ignore_attr(cand_graph)
    utils.add_disappear(cand_graph)
    utils.add_drift_dist_attr(cand_graph)

    solution_graph = solve_with_motile(cand_graph, exclusion_sets)
    print(
        f"Our gt graph has {gt_tracks.number_of_nodes()} nodes and {gt_tracks.number_of_edges()} edges"
    )
    print(
        f"Our solution graph has {solution_graph.number_of_nodes()} nodes and {solution_graph.number_of_edges()} edges"
    )
    print(
        f"Candidate graph has {cand_graph.number_of_nodes()} nodes and {cand_graph.number_of_edges()} edges"
    )

    save_solution_graph(solution_graph, output_csv_path)
    solution_seg = get_solution_seg(fragments, merge_history, solution_graph)
    zarr_root[output_seg_group] = solution_seg
