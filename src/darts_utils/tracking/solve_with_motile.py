import motile
from motile_toolbox.candidate_graph import graph_to_nx


def solve_with_motile(graph):
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

    solver.add_cost(motile.costs.Appear(constant=10.0, ignore_attribute = "ignore_appear"))
    solver.add_cost(motile.costs.Disappear(constant=2.0, ignore_attribute = "ignore_disappear"))

    #solver.add_cost(motile.costs.Split(constant = 1000.0))

    solver.solve()
    solution_graph = graph_to_nx(solver.get_selected_subgraph())
    return solution_graph
