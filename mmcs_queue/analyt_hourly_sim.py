import pandas as pd
import math

def compute_p0(tau, c, K):
    """
    Computes p0 for M/M/c/K queue:
      τ = λ/μ
      ρ = λ / (c·μ) = τ / c
    p0 = [ Σ_{n=0}^{c} τ^n / n!  + (τ^c / c!) · Σ_{n=c+1}^{K} ρ^{(n-c)} ]⁻¹
    """
    rho = tau / c
    sum1 = sum((tau**n) / math.factorial(n) for n in range(0, c+1))
    sum2 = 0.0
    if K > c:
        # geometric series Σ_{n=c+1}^{K} ρ^(n-c) = Σ_{m=1}^{K-c} ρ^m = (ρ (1-ρ^(K-c))) / (1-ρ) if ρ≠1
        if abs(rho - 1.0) < 1e-12:
            geom = (K - c)
        else:
            geom = (rho * (1 - rho**(K - c))) / (1 - rho)
        sum2 = (tau**c) / math.factorial(c) * geom
    denom = sum1 + sum2
    return 1.0 / denom

def compute_pn(n, p0, tau, c):
    """
    Computes p_n for M/M/c/K queue:
      For n ≤ c: p_n = (τ^n / n!) · p0
      For n > c:  p_n = (τ^c / c!) · (ρ^(n-c)) · p0
    with ρ = τ / c
    """
    if n <= c:
        return (tau**n) / math.factorial(n) * p0
    else:
        rho = tau / c
        return (tau**c) / math.factorial(c) * (rho**(n - c)) * p0

def mmck_metrics(arrival_rate, mu, c, K):
    """
    Given λ (arrival_rate), μ (service rate per server), c (number of servers),
    K (system capacity including servers and waiting places),
    compute L, Lq, W, Wq, utilization, and blocking probability (p_loss).
    Returns dict of metrics.
    """
    tau = arrival_rate / mu
    rho = arrival_rate / (mu * c)
    p0 = compute_p0(tau, c, K)
    # compute p_n for n = 0..K
    p = [compute_pn(n, p0, tau, c) for n in range(0, K+1)]
    # L = Σ_{n=0}^K n · p_n
    L = sum(n * p[n] for n in range(0, K+1))
    # Lq = Σ_{n=c+1}^K (n-c) · p_n (only queued waiting)
    Lq = sum((n - c) * p[n] for n in range(c+1, K+1)) if K > c else 0.0
    # blocking prob = p_K
    p_loss = p[K]
    # effective arrival rate
    lambda_eff = arrival_rate * (1.0 - p_loss)
    # W = L / λ_eff,  Wq = Lq / λ_eff
    if lambda_eff > 1e-12:
        W = L / lambda_eff
        Wq = Lq / lambda_eff
    else:
        W = float('inf')
        Wq = float('inf')
    # utilization (fraction of busy servers on average) = λ_eff / (c · μ)
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

def process_csv(input_csv_path, output_csv_path, mu, c, K, arrival_rate_col="Arrival Rate"):
    """
    Read input CSV with at least the column 'arrival_rate', compute metrics for each row,
    and write output CSV with the computed metrics.
    """
    df = pd.read_csv(input_csv_path)
    results = []
    for idx, row in df.iterrows():
        lam = float(row[arrival_rate_col])
        metrics = mmck_metrics(lam, mu, c, K)
        # optionally record additional context from row (e.g., hour)
        for col in row.index:
            if col not in metrics:
                metrics[col] = row[col]
        results.append(metrics)
    df_out = pd.DataFrame(results)
    df_out.to_csv(output_csv_path, index=False)
    print(f"Results written to {output_csv_path}")
    return df_out

if __name__ == "__main__":
    # --- user sets parameters here ---
    input_csv  = "final_results/hourly_arrival_rate.csv"     # CSV path with arrival_rate column (veh/hr)
    output_csv = "final_results/mmck_output_hourly.csv"
    mu = 4.0      # service rate per charger (veh/hr) → e.g., average service time = 10 minutes
    c  = 4        # number of chargers (servers)
    K  = 12       # total system capacity (including servers + waiting queue spots)

    # process
    df_results = process_csv(input_csv, output_csv, mu, c, K)
    print(df_results.head())
