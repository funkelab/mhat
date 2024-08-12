from pathlib import Path
import networkx as nx
import numpy as np
from typing import List
from .utils import nodes_from_segmentation
import csv

def load_merge_history(merge_path: Path) -> np.ndarray:
    """_summary_

    Args:
        merge_path (Path): _description_

    Returns:
        List[tuple, ...]: _description_
    """
    merge_history = []
    with open(merge_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            a = int(row["a"])
            b = int(row["b"])
            c = int(row["c"])
            score = float(row["score"])
            merge_history.append([a, b, c, score])

    merge_history = np.array(merge_history)
    return merge_history

def renumber_merge_history(merge_history: np.ndarray, max_node_id: int) -> np.ndarray:

    # renumber all the merges to be new ids
    for idx in range(merge_history.shape[0]):
        max_node_id += 1
        row = merge_history[idx]
        c = row[2]
        merge_history[idx][2] = max_node_id
        # replace all instances of c after this row and later with new node id
        if idx < merge_history.shape[0]:
            merge_history[idx + 1:][merge_history[idx + 1:] == c] = max_node_id

    return merge_history


def get_nodes(
    fragments: np.ndarray,
    merge_history: np.ndarray,
    min_score:float = 0.0,
    max_score:float = 0.5
) -> tuple[nx.DiGraph, list[tuple, ...]]:
    """_summary_

    Args:
        fragments (np.ndarray): _description_
        merge_history (List[tuple, ...]): Already renumbered so all cs are unique
        min_score (float, optional): _description_. Defaults to 0.0.
        max_score (float, optional): _description_. Defaults to 0.5.

    Returns:
        tuple[nx.DiGraph, list[tuple, ...]]: returns a networkx graph with all
        the nodes added, and a list of exclusion sets for nodes in the graph
        (nodes that cannot be selected together)
    """
    # create a dictionary from node_ids to last merge scores used to create the node
    last_scores = {}
    # create a dictionary from node_ids to next merge scores used to merge the node
    next_scores = {}
    
    fragments = fragments.copy()

    graph : nx.DiGraph | None = None
    conflict_sets = {} # map from parent node to conflict sets with that node

    for index, merge in enumerate(merge_history):
        a, b, c, score = merge
        a = int(a)
        b = int(b)
        c = int(c)

        if score >= min_score and graph is None:
            # get the initial fragments we want to populate the cand graph with
            graph = nodes_from_segmentation(fragments)

        # merge the fragments and add to history
        fragments[fragments == a] = c
        fragments[fragments == b] = c
        assert c not in last_scores
        last_scores[c] = score
        assert a not in next_scores
        assert b not in next_scores
        next_scores[a] = score
        next_scores[b] = score

        if score >= min_score and score < max_score:
            # add the new node to the graph
            new_seg_only = np.zeros_like(fragments)
            new_seg_only[fragments == c] = c
            node_graph = nodes_from_segmentation(new_seg_only)
            assert node_graph.number_of_nodes() == 1
            graph.add_nodes_from(node_graph.nodes(data=True))

            # add conflicting segs to conflict sets
            conflicts = []
            if a in conflict_sets:
                for cs in conflict_sets[a]:
                    cs.append(c)
                    conflicts.append(cs)
                del conflict_sets[a]
            else:
                conflicts.append([a, c])
            if b in conflict_sets:
                for cs in conflict_sets[b]:
                    cs.append(c)
                    conflicts.append(cs)
                del conflict_sets[b]
            else:
                conflicts.append([b, c])
            if len(conflicts) > 0:
                conflict_sets[c] = conflicts

    for node in graph.nodes():
        cohesion_score = 1 - last_scores.get(node, 0.0)
        adhesion_score = next_scores.get(node, 1.0)
        graph.nodes[node]["cohesion"] = cohesion_score
        graph.nodes[node]["adhesion"] = adhesion_score

    exclusion_sets = []
    for conflicts in conflict_sets.values():
        # filter out elements that didn't make it into the graph
        for conflict_set in conflicts:
            conflict_set = [node for node in conflict_set if node in graph.nodes]
            exclusion_sets.append(conflict_set)

    return graph, exclusion_sets
