import numpy as np
import networkx as nx
import json

def _calculate_weight(u, v, data, seed_stats_lookup, alphas):
    """
    Calculates a custom weight for a single edge (u, v).
    This is our cost function.

    The cost of an edge is:
    1. The scaled travel cost (length * alpha[0])
    2. PLUS an 'initial cost' if the source 'u' is a seed node (EVCS).
       This 'initial cost' represents the queueing/service time at that station.
    """
    
    length = data.get('length', 0)
    cost = alphas[0] * length

    if u in seed_stats_lookup:
        stats = seed_stats_lookup[u]
        L_emp = stats.get("L_emp", 0)
        Lq_emp = stats.get("Lq_emp", 0)
        W_emp = stats.get("W_emp", 0)
        Wq_emp = stats.get("Wq_emp", 0)
        
        cost += alphas[1] * Lq_emp + alphas[2] * L_emp + alphas[3] * Wq_emp + alphas[4] * W_emp
    
    return cost

def apply_custom_weights(graph, seeds):
    """
    Applies custom weights to every edge in the graph.

    The custom weight `cost(u, v)` is calculated as:
    cost = (alpha[0] * length) + (QueueingCost if u is a seed)

    This model assumes the total path cost from a seed 's' to a node 'n' is:
    TotalCost = QueueingCost(s) + TravelCost(s -> n)

    Dijkstra's algorithm will correctly find the 'n' that minimizes this
    value, as the QueueingCost is applied as a fixed cost on the first
    edge leaving any seed.

    Args:
        graph (networkx.MultiDiGraph): The input graph from OSMnx.
        seeds (np.array): An array of node IDs for the seed locations (EVCS).

    Returns:
        tuple: A tuple containing the modified graph and the name of the new weight attribute.
    """
    print("Applying custom weights to all edges...")
    
    alphas = [0.2, 0.2, 0.2, 0.2, 0.2]
    
    try:
        with open("evcs_phase2_results/evcs_stats.json") as f:
            evcs_stat = json.load(f)
    except FileNotFoundError:
        print("Error: evcs_phase2_results/evcs_stats.json not found.")
        print("Please ensure the stats file is in the correct location.")
        weight_attribute_name = 'length'
        print(f"Warning: Falling back to using '{weight_attribute_name}' for weights.")
        return graph, weight_attribute_name
    except json.JSONDecodeError:
        print("Error: Could not decode evcs_phase2_results/evcs_stats.json.")
        weight_attribute_name = 'length'
        print(f"Warning: Falling back to using '{weight_attribute_name}' for weights.")
        return graph, weight_attribute_name

    seed_stats_lookup = {}
    for i, seed_node_id in enumerate(seeds):
        key = f"EVCS_{i:02d}"
        
        if key in evcs_stat:
            seed_stats_lookup[seed_node_id] = evcs_stat[key]
        else:
            alt_key = f"EVCS_{i}" if i > 9 else f"EVCS_0{i}"
            if alt_key in evcs_stat:
                seed_stats_lookup[seed_node_id] = evcs_stat[alt_key]
            else:
                print(f"Warning: No stats found for seed {i} (key '{key}' or '{alt_key}')")
                seed_stats_lookup[seed_node_id] = {} 
    
    print(f"Loaded stats for {len(seed_stats_lookup)} seed nodes.")

    custom_weights = {
        (u, v, k): _calculate_weight(u, v, d, seed_stats_lookup, alphas)
        for u, v, k, d in graph.edges(keys=True, data=True)
    }

    weight_attribute_name = 'custom_weight'
    nx.set_edge_attributes(graph, custom_weights, weight_attribute_name)
    
    print("Custom weights applied.")
    return graph, weight_attribute_name
