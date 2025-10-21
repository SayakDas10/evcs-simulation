#!/usr/bin/env -S uv run --script

import os
import osmnx as ox
import numpy as np
import matplotlib.pyplot as plt

import custom_weights
import voronoi
np.random.seed(1234)

def main():
    place = "Bangalore, India"
    graph_filepath = "bangalore.graphml"
    
    if os.path.exists(graph_filepath):
        print(f"Loading graph from file: {graph_filepath}")
        G = ox.load_graphml(graph_filepath)
    else:
        print("Graph file not found. Downloading from OSMnx...")
        G = ox.graph_from_place(place, network_type="drive")
        print(f"Saving graph to file: {graph_filepath}")
        ox.save_graphml(G, filepath=graph_filepath)

    nodes, edges = ox.graph_to_gdfs(G)

    seed_nodes = np.random.choice(nodes.index, size=20, replace=False)
    print(f"Selected {len(seed_nodes)} seed nodes")

    G, weight_attr = custom_weights.apply_custom_weights(G, seed_nodes)

    projected_crs = nodes.estimate_utm_crs()
    local_epsg = projected_crs.to_epsg()
    # regions_gdf = voronoi.generate_voronoi_regions(G, nodes, seed_nodes, weight_attribute=weight_attr, crs_epsg=local_epsg) # Ours
    regions_gdf = voronoi.generate_euclidean_voronoi_regions(nodes, seed_nodes, crs_epsg=local_epsg) # Euclidean
    print("Voronoi generation complete.")

    print("Plotting results...")
    fig, ax = plt.subplots(figsize=(12, 12))
    edges.plot(ax=ax, color="lightgray", linewidth=0.3)
    regions_gdf.plot(ax=ax, column="seed", cmap="tab20", alpha=0.6, legend=False)
    nodes.loc[seed_nodes].plot(ax=ax, color="red", markersize=20, label="EVCS Locations")
    
    plt.title(f"Graph-based Voronoi Tessellation on {place} Road Network", fontsize=14)
    plt.legend()
    plt.axis("off")
    ax.set_xlim(77.55, 77.65)
    ax.set_ylim(12.90, 13.00)
    plt.savefig("network1.png", dpi=300, bbox_inches="tight")
    plt.show()

if __name__ == "__main__":
    main()
