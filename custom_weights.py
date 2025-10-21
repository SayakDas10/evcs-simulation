import numpy as np
import networkx as nx
import json

def _calculate_weight(u, v, data, seeds):
    """
    Calculates a custom weight for a single edge.
    This is our cost function too.
    """
    length = data.get('length', 0)

    with open("evcs_phase2_results/evcs_config.json") as f:
        evcs_stat = json.load(f)

    if v in seeds:
        evcs_num = np.where(seeds==v)[0][0]
        stats = evcs_stat[f"EVCS_{evcs_num}" if evcs_num > 9 else f"EVCS_0{evcs_num}"]
        mu = stats["mu"]
        return length + length * (1/mu)
    else:
        return length

def apply_custom_weights(graph, seeds):
    """
    Applies custom weights to every edge in the graph.

    Args:
        graph (networkx.MultiDiGraph): The input graph from OSMnx.

    Returns:
        tuple: A tuple containing the modified graph and the name of the new weight attribute.
    """
    print("Applying custom weights to all edges...")
    
    # Create a dictionary of custom weights for all edges
    custom_weights = {
        (u, v, k): _calculate_weight(u, v, d, seeds)
        for u, v, k, d in graph.edges(keys=True, data=True)
    }

    # Add this new 'custom_weight' attribute to every edge
    weight_attribute_name = 'custom_weight'
    nx.set_edge_attributes(graph, custom_weights, weight_attribute_name)
    
    return graph, weight_attribute_name
