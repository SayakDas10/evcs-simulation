import json
import pandas as pd
import numpy as np
import math
import random
import os

# ================================================================
# M/M/c/K Queue Functions
# ================================================================
def compute_p0(tau, c, K):
    rho = tau / c
    sum1 = sum((tau**n) / math.factorial(n) for n in range(0, c + 1))
    sum2 = 0.0
    if K > c:
        if abs(rho - 1.0) < 1e-12:
            geom = (K - c)
        else:
            geom = (rho * (1 - rho ** (K - c))) / (1 - rho)
        sum2 = (tau ** c) / math.factorial(c) * geom
    denom = sum1 + sum2
    return 1.0 / denom


def compute_pn(n, p0, tau, c):
    if n <= c:
        return (tau ** n) / math.factorial(n) * p0
    else:
        rho = tau / c
        return (tau ** c) / math.factorial(c) * (rho ** (n - c)) * p0


def mmck_metrics(arrival_rate, mu, c, K):
    tau = arrival_rate / mu
    rho = tau / c
    p0 = compute_p0(tau, c, K)
    p = [compute_pn(n, p0, tau, c) for n in range(0, K + 1)]

    L = sum(n * p[n] for n in range(0, K + 1))
    Lq = sum((n - c) * p[n] for n in range(c + 1, K + 1)) if K > c else 0.0
    p_loss = p[K]
    lambda_eff = arrival_rate * (1.0 - p_loss)

    if lambda_eff > 1e-12:
        W = L / lambda_eff
        Wq = Lq / lambda_eff
    else:
        W = float('inf')
        Wq = float('inf')

    utilization = lambda_eff / (c * mu)
    return {
        "arrival_rate": arrival_rate,
        "mu": mu,
        "c": c,
        "K": K,
        "tau": tau,
        "rho": rho,
        "p0": p0,
        "p_loss": p_loss,
        "lambda_eff": lambda_eff,
        "L": L,
        "Lq": Lq,
        "W": W,
        "Wq": Wq,
        "utilization": utilization
    }

# ================================================================
# Main Processing Logic
# ================================================================
def run_mmck_for_evcs(evcs_config_path, arrival_config_path, output_json_path):
    # Load EVCS configuration
    with open(evcs_config_path, "r") as f:
        evcs_data = json.load(f)

    # Load hourly arrival rate profiles
    with open(arrival_config_path, "r") as f:
        arrival_profiles = json.load(f)

    evcs_results = {}

    for evcs_id, config in evcs_data.items():
        # Randomly select one of the arrival rate configs
        selected_key = random.choice(list(arrival_profiles.keys()))
        selected_profile = arrival_profiles[selected_key]

        hours = selected_profile["hour"]
        hourly_arrivals = selected_profile["arrival_rate"]

        # Extract base parameters for EVCS
        base_lambda = config["lambda"]
        base_mu = config["mu"]
        c = config["c"]
        K = config["K"]

        hourly_results = {}
        for hr, arr_factor in zip(hours, hourly_arrivals):
            # Scale lambda and mu by arrival rate factor to keep rho constant
            lam = base_lambda * arr_factor
            mu = base_mu * arr_factor

            # Compute queue metrics
            metrics = mmck_metrics(lam, mu, c, K)
            hourly_results[int(hr)] = metrics

        evcs_results[evcs_id] = {
            "zone_type": config["zone_type"],
            "charger_type": config["charger_type"],
            "arrival_profile": selected_key,
            "hourly_results": hourly_results
        }

    # Save output JSON
    with open(output_json_path, "w") as f:
        json.dump(evcs_results, f, indent=4)

    print(f"✅ M/M/c/K hourly analysis complete for {len(evcs_results)} EVCS.")
    print(f"📄 Results saved to: {os.path.abspath(output_json_path)}")


# ================================================================
# Example Usage
# ================================================================
if __name__ == "__main__":
    evcs_config_path = "evcs_phase2_results/evcs_config.json"          # contains 50 EVCS configs
    arrival_config_path = "final_results/arrival_rate_configs.json"  # contains 10 hourly profiles
    output_json_path = "final_results/mmck_hourly_evcs_stats.json"

    run_mmck_for_evcs(evcs_config_path, arrival_config_path, output_json_path)
