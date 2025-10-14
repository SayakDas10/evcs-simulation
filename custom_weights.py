import networkx as nx

def _calculate_weight(u, v, data):
    """
    Calculates a custom weight for a single edge.
    This is our cost function too.
    """
    length = data.get('length', 0)
    highway_type = data.get('highway', 'unknown')

    if highway_type == 'residential':
        return length * 2.0
    elif highway_type in ['motorway', 'trunk', 'primary']:
        return length * 0.8
    else:
        return length

def apply_custom_weights(graph):
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
        (u, v, k): _calculate_weight(u, v, d)
        for u, v, k, d in graph.edges(keys=True, data=True)
    }

    # Add this new 'custom_weight' attribute to every edge
    weight_attribute_name = 'custom_weight'
    nx.set_edge_attributes(graph, custom_weights, weight_attribute_name)
    
    return graph, weight_attribute_name
