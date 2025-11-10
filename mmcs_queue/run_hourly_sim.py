import pandas as pd
import numpy as np
from mmck_sim import run_mmck_hourly

def simulate_100_days(csv_path, mu, c, K, output_excel="MMCK_100days_results.xlsx"):
    # Load hourly arrival rates (EV/hour) and convert to per minute
    df = pd.read_csv(csv_path)
    lam_hourly = df["EV_arrivals_per_hour"].values / 60.0  # per minute

    sim_minutes = 24 * 60  # 1440 minutes per day

    arrivals_all = pd.DataFrame()
    queue_all = pd.DataFrame()
    util_all = pd.DataFrame()

    for day in range(1, 101):
        arrivals, queue_len, utilization = run_mmck_hourly(lam_hourly, mu, c, K, sim_minutes)
        arrivals_all[f"Day_{day}"] = arrivals
        queue_all[f"Day_{day}"] = queue_len
        util_all[f"Day_{day}"] = utilization

    # Add time column (in hours)
    time_index = np.arange(sim_minutes) / 60
    arrivals_all.insert(0, "Time (h)", time_index)
    queue_all.insert(0, "Time (h)", time_index)
    util_all.insert(0, "Time (h)", time_index)

    # Write to Excel
    with pd.ExcelWriter(output_excel) as writer:
        arrivals_all.to_excel(writer, sheet_name="Arrivals_vs_Time", index=False)
        queue_all.to_excel(writer, sheet_name="Queue_vs_Time", index=False)
        util_all.to_excel(writer, sheet_name="Utilization_vs_Time", index=False)

    print(f"Simulation complete. Results saved to {output_excel}")


if __name__ == "__main__":
    csv_path = "dynamic_price/ev_arrival_rate_24h.csv"
    mu = 0.001
    c = 5
    K = 20
    output_excel = "final_results/MMCK_100days_results_3.xlsx"
    simulate_100_days(csv_path=csv_path,
                      mu=mu,
                      c=c,
                      K=K,
                      output_excel=output_excel)

