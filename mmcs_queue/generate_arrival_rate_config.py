import pandas as pd
import numpy as np
import json
import random
import os

def generate_random_rotations(csv_path, output_json_path, num_configs=10):
    """
    Read hourly arrival rates from CSV, generate rotated variants,
    and save as JSON with keys config_1, config_2, ..., config_N.
    """

    # Load CSV
    df = pd.read_csv(csv_path)

    if "Hour" not in df.columns or "Arrival Rate" not in df.columns:
        raise ValueError("CSV must contain columns: 'hour' and 'arrival_rate'")

    hours = df["Hour"].tolist()
    base_arrivals = df["Arrival Rate"].tolist()

    configs = {}

    for i in range(1, num_configs + 1):
        # Choose random rotation (0–6 hours)
        shift = random.randint(0, 6)
        rotated_arrivals = np.roll(base_arrivals, shift)

        configs[f"config_{i}"] = {
            "hour": hours,
            "arrival_rate": rotated_arrivals.tolist(),
            #"rotation_shift": shift
        }

    # Save to JSON
    with open(output_json_path, "w") as f:
        json.dump(configs, f, indent=4)

    print(f"✅ Generated {num_configs} configurations with random rotations (0–6h).")
    print(f"📄 Saved at: {os.path.abspath(output_json_path)}")


# Example usage
if __name__ == "__main__":
    csv_path = "final_results/hourly_arrival_rate.csv"         # Input CSV file path
    output_json_path = "final_results/arrival_rate_configs.json"  # Output JSON file
    generate_random_rotations(csv_path, output_json_path)
