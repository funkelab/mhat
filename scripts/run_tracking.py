import csv
from pathlib import Path

import zarr
from darts_utils.tracking import solve_with_motile, utils


def save_solution(solution_graph, csv_path):
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


if __name__ == "__main__":
    vid_num = 2

    base_path = Path(f"/Volumes/funke/data/darts/synthetic_data/test1/{vid_num}")
    zarr_path = base_path / "data.zarr"
    gt_csv_path = base_path / "gt_tracks.csv"
    output_csv_path = base_path / "pred_tracks.csv"

    seg_group = "pred_mask_0.15"
    max_edge_distance = 50

    zarr_root = zarr.open(zarr_path)
    seg = zarr_root[seg_group][:]

    gt_tracks = utils.read_gt_tracks(zarr_path, gt_csv_path)
    cand_graph = utils.nodes_from_segmentation(seg)
    utils.add_cand_edges(cand_graph, max_edge_distance)

    solution_graph = solve_with_motile(cand_graph)
    print("Solution")
    print(
        f"Our gt graph has {gt_tracks.number_of_nodes()} nodes and {gt_tracks.number_of_edges()} edges"
    )
    print(
        f"Our solution graph has {solution_graph.number_of_nodes()} nodes and {solution_graph.number_of_edges()} edges"
    )

    save_solution(solution_graph, output_csv_path)
