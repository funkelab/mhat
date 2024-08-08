import motile
from motile_toolbox.candidate_graph import graph_to_nx
from darts_utils.tracking import utils


def solve_with_motile(config, graph, exclusion_sets):
    """Set up and solve the network flow problem.

    Args:
        graph (motile.TrackGraph): The candidate graph.

    Returns:
        nx.DiGraph: The networkx digraph with the selected solution tracks
    """
    solver = motile.Solver(graph)

    solver.add_cost(motile.costs.EdgeSelection(
        weight= config["edge_selection_weight"], 
        attribute="drift_dist", 
        constant = config["edge_selection_constant"]
    ))

    solver.add_constraint(motile.constraints.MaxParents(1))
    solver.add_constraint(motile.constraints.MaxChildren(1))

    solver.add_cost(utils.HyperAreaSplit(
        weight=config["hyper_weight"],
        constant=config["hyper_constant"],
        area_attribute= "area"
    ))

    solver.add_cost(motile.costs.NodeSelection(
        weight=config["cohesion_weight"], 
        attribute="cohesion", 
        constant=config["cohesion_constant"]
    ), name="cohesion")
    
    solver.add_cost(motile.costs.NodeSelection(
        weight=config["adhesion_weight"], 
        attribute="adhesion", 
        constant=config["adhesion_constant"],
    ), name="adhesion")
    
    solver.add_cost(motile.costs.Appear(
        constant=config["appear_constant"], 
        ignore_attribute = "ignore_appear"
    ))
    solver.add_cost(motile.costs.Disappear(
        constant=config["disappear_constant"], 
        ignore_attribute = "ignore_disappear"
    ))

    solver.add_constraint(motile.constraints.ExclusiveNodes(exclusion_sets))

    solver.solve()
    solution_graph = graph_to_nx(solver.get_selected_subgraph())
    return solution_graph
