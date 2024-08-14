from __future__ import annotations

from csv import DictReader
from itertools import combinations
from typing import Any, Iterable

import motile
import networkx as nx
import numpy as np
import scipy
import skimage
import zarr


def nodes_from_segmentation(
    segmentation: np.ndarray, size_threshold: int | None = None
) -> nx.DiGraph:
    """Extract candidate nodes from a segmentation.

    Args:
        segmentation (np.ndarray): A numpy array with integer labels and dimensions
            (t, y, x).

        size_threshold (int): A minimum area for candidate nodes. Nodes smaller
            than this area will not be added to the graph.

    Returns:
        nx.DiGraph: A candidate graph with only nodes.
    """
    cand_graph = nx.DiGraph()
    for t in range(len(segmentation)):
        seg_frame = segmentation[t]
        props = skimage.measure.regionprops(seg_frame)
        for regionprop in props:
            if size_threshold and regionprop.area < size_threshold:
                continue
            node_id = int(regionprop.label)
            attrs = {
                "time": t,
                "x": float(regionprop.centroid[0]),
                "y": float(regionprop.centroid[1]),
                "label": node_id,
                "area": regionprop.area,
            }
            cand_graph.add_node(node_id, **attrs)

    return cand_graph


def read_gt_tracks(mask_zarr, tracks_file) -> nx.DiGraph:
    """Get the ground truth tracks as a networkx graph

    Args:
        mask_zarr (str | Path): Path to the zarr containing the ground truth mask
        tracks_file (str | Path): path to csv containing ground truth links

    Returns:
        nx.DiGraph: _description_
    """
    data_root = zarr.open(mask_zarr)
    gt_seg = data_root["mask"][:]
    gt_tracks = nodes_from_segmentation(gt_seg)

    with open(tracks_file) as f:
        reader = DictReader(f)
        for row in reader:
            i_d = int(float(row["id"]))
            assert i_d in gt_tracks.nodes, f"node {i_d} not in graph"
            parent_id = int(float(row["parent_id"]))
            if parent_id != -1:
                assert (
                    parent_id in gt_tracks.nodes
                ), f"parent id {parent_id} not in graph"
                gt_tracks.add_edge(parent_id, i_d)
    return gt_tracks


def _compute_node_frame_dict(cand_graph: nx.DiGraph) -> dict[int, list[Any]]:
    """Compute dictionary from time frames to node ids for candidate graph.

    Args:
        cand_graph (nx.DiGraph): A networkx graph

    Returns:
        dict[int, list[Any]]: A mapping from time frames to lists of node ids.
    """
    node_frame_dict: dict[int, list[Any]] = {}
    for node, data in cand_graph.nodes(data=True):
        t = data["time"]
        if t not in node_frame_dict:
            node_frame_dict[t] = []
        node_frame_dict[t].append(node)
    return node_frame_dict


def create_kdtree(
    cand_graph: nx.DiGraph, node_ids: Iterable[Any]
) -> scipy.spatial.KDTree:
    positions = [
        [cand_graph.nodes[node]["x"], cand_graph.nodes[node]["y"]] for node in node_ids
    ]
    return scipy.spatial.KDTree(positions)


def add_cand_edges(
    cand_graph: nx.DiGraph,
    max_edge_distance: float,
) -> None:
    """Add candidate edges to a candidate graph by connecting all nodes in adjacent
    frames that are closer than max_edge_distance. Also adds attributes to the edges.

    Args:
        cand_graph (nx.DiGraph): Candidate graph with only nodes populated. Will
            be modified in-place to add edges.
        max_edge_distance (float): Maximum distance that objects can travel between
            frames. All nodes within this distance in adjacent frames will by connected
            with a candidate edge.
        node_frame_dict (dict[int, list[Any]] | None, optional): A mapping from frames
            to node ids. If not provided, it will be computed from cand_graph. Defaults
            to None.
    """
    node_frame_dict = _compute_node_frame_dict(cand_graph)

    frames = sorted(node_frame_dict.keys())
    prev_node_ids = node_frame_dict[frames[0]]
    prev_kdtree = create_kdtree(cand_graph, prev_node_ids)
    for frame in frames:
        if frame + 1 not in node_frame_dict:
            continue
        next_node_ids = node_frame_dict[frame + 1]
        next_kdtree = create_kdtree(cand_graph, next_node_ids)

        matched_indices = prev_kdtree.query_ball_tree(next_kdtree, max_edge_distance)

        for prev_node_id, next_node_indices in zip(prev_node_ids, matched_indices):
            for next_node_index in next_node_indices:
                next_node_id = next_node_ids[next_node_index]
                cand_graph.add_edge(prev_node_id, next_node_id)

        prev_node_ids = next_node_ids
        prev_kdtree = next_kdtree


def relabel_segmentation(
    solution_nx_graph: nx.DiGraph,
    segmentation: np.ndarray,
) -> np.ndarray:
    """Relabel a segmentation based on tracking results so that nodes in same
    track share the same id. IDs do change at division.

    Args:
        solution_nx_graph (nx.DiGraph): Networkx graph with the solution to use
            for relabeling. Nodes not in graph will be removed from seg.
        segmentation (np.ndarray): Original (potentially multi-hypothesis)
            segmentation with dimensions (t,h,[z],y,x), where h is 1 for single
            input segmentation.

    Returns:
        np.ndarray: Relabeled segmentation array where nodes in same track share same
            id with shape (t,1,[z],y,x)
    """
    tracked_masks = np.zeros_like(segmentation)
    id_counter = 1
    parent_nodes = [n for (n, d) in solution_nx_graph.out_degree() if d > 1]
    soln_copy = solution_nx_graph.copy()
    for parent_node in parent_nodes:
        out_edges = solution_nx_graph.out_edges(parent_node)
        soln_copy.remove_edges_from(out_edges)
    for node_set in nx.weakly_connected_components(soln_copy):
        for node in node_set:
            time_frame = solution_nx_graph.nodes[node]["time"]
            previous_seg_id = node
            previous_seg_mask = segmentation[time_frame] == previous_seg_id
            tracked_masks[time_frame][previous_seg_mask] = id_counter
        id_counter += 1
    return tracked_masks


def add_appear_ignore_attr(cand_graph):
    for node_id, attrs in cand_graph.nodes(data=True):
        if "time" not in attrs:
            continue  # skip hypernodes
        if attrs.get("time") == 0:
            cand_graph.nodes[node_id]["ignore_appear"] = True


def add_disappear(cand_graph):
    for node_id, attrs in cand_graph.nodes(data=True):
        if "time" not in attrs:
            continue  # skip hypernodes
        if attrs.get("time") == 99 or attrs.get("x") > 380:
            cand_graph.nodes[node_id]["ignore_disappear"] = True


def add_drift_dist_attr(cand_graph: motile.TrackGraph, drift=10):
    for edge in cand_graph.edges:
        if cand_graph.is_hyperedge(edge):
            us, vs = edge
            u = us[0]  # assume always one "source" node
            v1, v2 = vs  # assume always two "target" nodes
            pos_u = drift + cand_graph.nodes[u]["x"]
            pos_v = (cand_graph.nodes[v1]["x"] + cand_graph.nodes[v2]["x"]) / 2
        else:
            u, v = edge
            pos_u = drift + cand_graph.nodes[u]["x"]
            pos_v = cand_graph.nodes[v]["x"]

        drift_dist = abs(pos_u - pos_v)
        cand_graph.edges[edge]["drift_dist"] = drift_dist


def add_area_diff_attr(cand_graph: motile.TrackGraph):
    for edge in cand_graph.edges:
        if cand_graph.is_hyperedge(edge):
            us, vs = edge
            u = us[0]  # assume always one "source" node
            v1, v2 = vs  # assume always two "target" nodes
            area_u = cand_graph.nodes[u]["area"]
            area_v = cand_graph.nodes[v1]["area"] + cand_graph.nodes[v2]["area"]
        else:
            u, v = edge
            area_u = cand_graph.nodes[u]["area"]
            area_v = cand_graph.nodes[v]["area"]

        area_diff = np.abs(area_u - area_v)
        cand_graph.edges[edge]["area_diff"] = area_diff


def load_prediction(csv_path):
    with open(csv_path) as f:
        reader = DictReader(f)
        graph = nx.DiGraph()
        for row in reader:
            node_id = int(float(row["id"]))
            attrs = {
                "time": int(float(row["time"])),
                "pos": [int(float(row["x"])), int(float(row["y"]))],
            }
            parent_id = int(float(row["parent_id"]))
            graph.add_node(node_id, **attrs)
            if parent_id != -1:
                graph.add_edge(parent_id, node_id)
    return graph


def add_division_hyperedges(candidate_graph: nx.DiGraph) -> nx.DiGraph:
    """Add hyper edges representing specific divisions to the graph

    Args:
        candidate_graph (nx.DiGraph): A candidate graph already populated with
            normal nodes and edges.

    Returns:
        nx.DiGraph: The candidate graph with additional hypernodes for each
            possible division
    """
    nodes_original = list(candidate_graph.nodes)
    for node in nodes_original:
        successors = candidate_graph.successors(node)
        pairs = list(combinations(successors, 2))
        for pair in pairs:
            hypernode = str(node) + "_" + str(pair[0]) + "_" + str(pair[1])
            candidate_graph.add_node(hypernode)
            candidate_graph.add_edge(
                node,
                hypernode,
            )
            candidate_graph.add_edge(
                hypernode,
                pair[0],
            )
            candidate_graph.add_edge(
                hypernode,
                pair[1],
            )
    return candidate_graph
