import json
import os

def normalize_evcs_json(json_path):
    # Load JSON file
    with open(json_path, "r") as f:
        data = json.load(f)

    # Collect all L and W values to find max
    L_vals = []
    W_vals = []
    for evcs_id, stats in data.items():
        # prefer theoretical, fallback to empirical
        L_vals.append(stats.get("L_theo", stats.get("L_emp", 0)))
        W_vals.append(stats.get("W_theo", stats.get("W_emp", 0)))

    L_max = max(L_vals) if L_vals else 1.0
    W_max = max(W_vals) if W_vals else 1.0

    # Normalize L, Lq, W, Wq for all EVCS
    for evcs_id, stats in data.items():
        for key in ["L_emp", "Lq_emp", "L_theo", "Lq_theo"]:
            if key in stats:
                stats[key + "_norm"] = stats[key] / L_max
        for key in ["W_emp", "Wq_emp", "W_theo", "Wq_theo"]:
            if key in stats:
                stats[key + "_norm"] = stats[key] / W_max

    # Save back to the same JSON path
    json_path = json_path.replace(".json", "_normalized.json")
    with open(json_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"✅ Normalization complete. File updated: {json_path}")
    print(f"Max L = {L_max:.4f}, Max W = {W_max:.4f}")

# Example usage
if __name__ == "__main__":
    json_path = "evcs_phase2_results/evcs_stats.json"  # path to your JSON file
    normalize_evcs_json(json_path)
