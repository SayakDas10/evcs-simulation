import networkx as nx
import geopandas as gpd
import pandas as pd
from shapely.ops import unary_union

def generate_voronoi_regions(G, nodes_gdf, seed_nodes, weight_attribute, crs_epsg):
    """
    Computes network-distance-based Voronoi regions.

    Args:
        G (networkx.MultiDiGraph): The graph.
        nodes_gdf (gpd.GeoDataFrame): GeoDataFrame of graph nodes.
        seed_nodes (list): A list of node IDs to use as Voronoi seeds.
        weight_attribute (str): The edge attribute to use for distance calculation.
        crs_epsg (int): The EPSG code for the projected CRS to use for buffering.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the polygonal Voronoi regions.
    """
    print(f"Generating Voronoi regions using '{weight_attribute}'...")

    distances, paths = nx.multi_source_dijkstra(G, sources=list(seed_nodes), weight=weight_attribute)

    # Assign each node to its closest seed
    assignments = []
    for node, dist in distances.items():
        path = paths[node]
        closest_seed = path[0]
        assignments.append((node, closest_seed, dist))

    region_df = pd.DataFrame(assignments, columns=["node", "seed", "distance"])
    region_gdf = nodes_gdf.merge(region_df, left_index=True, right_on="node")

    # Create polygonal regions by buffering and merging nodes
    region_gdf_proj = region_gdf.to_crs(epsg=crs_epsg)

    regions = []
    for seed, group in region_gdf_proj.groupby("seed"):
        # Buffer each node point by 30 meters and merge them into a single polygon
        merged = unary_union(group.geometry.buffer(30))
        regions.append({"seed": seed, "geometry": merged})

    # Create the final GeoDataFrame
    regions_gdf = gpd.GeoDataFrame(regions, crs=region_gdf_proj.crs)
    regions_gdf = regions_gdf.to_crs(epsg=4326) # Reproject back to WGS84

    return regions_gdf

def generate_euclidean_voronoi_regions(nodes_gdf, seed_nodes, crs_epsg):
    """
    Computes Euclidean-distance-based Voronoi regions.

    Args:
        nodes_gdf (gpd.GeoDataFrame): GeoDataFrame of graph nodes (in WGS84, epsg:4326).
        seed_nodes (list): A list of node IDs to use as Voronoi seeds.
        crs_epsg (int): The EPSG code for the projected CRS to use for buffering
                          and accurate distance calculation.

    Returns:
        gpd.GeoDataFrame: A GeoDataFrame containing the polygonal Voronoi regions.
    """
    print("Generating Euclidean Voronoi regions...")

    nodes_proj = nodes_gdf.to_crs(epsg=crs_epsg)
    
    seeds_proj = nodes_proj.loc[seed_nodes].copy()
    seeds_proj['seed'] = seeds_proj.index 
    assigned_nodes_proj = gpd.sjoin_nearest(nodes_proj, seeds_proj[['geometry', 'seed']], how='left', distance_col='distance')

    regions = []
    for seed, group in assigned_nodes_proj.groupby("seed"):
        merged = unary_union(group.geometry.buffer(30))
        regions.append({"seed": seed, "geometry": merged})

    regions_gdf = gpd.GeoDataFrame(regions, crs=assigned_nodes_proj.crs)
    regions_gdf = regions_gdf.to_crs(epsg=4326) 

    return regions_gdf
