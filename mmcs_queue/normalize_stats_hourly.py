import json
import os
from collections import defaultdict

def normalize_hourly_evcs_json(json_path):
    """
    Normalize hourly EVCS stats for keys: L, Lq, W, Wq.
    For each hour (0–23), find max L and W across all EVCS entries,
    then normalize all L/Lq/W/Wq values by their respective hourly max.
    """
    # Load JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Store max L and W for each hour
    L_hourly_max = defaultdict(float)
    W_hourly_max = defaultdict(float)

    # Pass 1: find hourly maxima
    for evcs_id, stats in data.items():
        if "hourly_results" not in stats:
            continue
        for hour_str, vals in stats["hourly_results"].items():
            hour = int(hour_str)
            L_hourly_max[hour] = max(L_hourly_max[hour], vals.get("L", 0))
            W_hourly_max[hour] = max(W_hourly_max[hour], vals.get("W", 0))

    # Avoid division by zero
    for hour in range(24):
        L_hourly_max[hour] = L_hourly_max.get(hour, 1.0) or 1.0
        W_hourly_max[hour] = W_hourly_max.get(hour, 1.0) or 1.0

    # Pass 2: normalize all hourly values
    for evcs_id, stats in data.items():
        if "hourly_results" not in stats:
            continue
        for hour_str, vals in stats["hourly_results"].items():
            hour = int(hour_str)
            # Normalize L and Lq
            for key in ["L", "Lq"]:
                if key in vals:
                    vals[key + "_norm"] = vals[key] / L_hourly_max[hour]
            # Normalize W and Wq
            for key in ["W", "Wq"]:
                if key in vals:
                    vals[key + "_norm"] = vals[key] / W_hourly_max[hour]

    # Save to new file
    new_path = json_path.replace(".json", "_normalized.json")
    with open(new_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Hourly normalization complete. File saved as: {new_path}")
    print(f"Computed hourly maxima for {len(L_hourly_max)} hours")

# Example usage
if __name__ == "__main__":
    json_path = "final_results/mmck_hourly_evcs_stats.json"  # path to your JSON file
    normalize_hourly_evcs_json(json_path)
