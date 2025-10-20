import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
import os

SAVE_DIR = "/home/saksham/samsad/MLSpace/MTech Sem3/EV-charging-impact/ev_phase2_results"

# --- THEORETICAL FORMULA FUNCTIONS for M/M/c/K ---

def calculate_mmck_performance(lam, mu, c, K):
    """
    Calculates the exact theoretical performance metrics for an M/M/c/K system.
    """
    if lam <= 0:
        return 0, 0, 0, 0, 0
    
    A = lam / mu
    c = int(c)
    K = int(K)
    rho = A / c

    # Calculate P0 (probability of an empty system)
    sum_part1 = sum(A**n / math.factorial(n) for n in range(c))
    if abs(rho - 1.0) < 1e-9: # Handle rho being very close to 1
        sum_part2 = (A**c / math.factorial(c)) * (K - c + 1)
    else:
        sum_part2 = (A**c / math.factorial(c)) * (1 - rho**(K - c + 1)) / (1 - rho)
    
    p0 = 1.0 / (sum_part1 + sum_part2) if (sum_part1 + sum_part2) > 0 else 0.0

    # 1. Loss Probability (PK)
    pb = p0 * (A**K / (math.factorial(c) * (c**(K - c))))

    # 2. Average number in queue (Lq)
    if abs(rho - 1.0) < 1e-9:
        lq_factor = ((K - c) * (K - c + 1)) / 2
        lq = p0 * (A**c / math.factorial(c)) * lq_factor
    else:
        lq_factor1 = (rho * (1 - rho**(K - c + 1))) / ((1 - rho)**2)
        lq_factor2 = ((K - c + 1) * rho**(K - c + 1)) / (1 - rho)
        lq = p0 * (A**c / math.factorial(c)) * (lq_factor1 - lq_factor2)
        
    # 3. Effective arrival rate and other metrics via Little's Law
    effective_lambda = lam * (1 - pb)
    
    wq = lq / effective_lambda if effective_lambda > 0 else 0
    w = wq + (1 / mu)
    l = effective_lambda * w
    
    return pb, l, w, lq, wq

# --- SIMULATION AND PLOTTING ---

def run_simulation_run(lam, mu, c, K, q_size, sim_time):
    """Performs a single event-driven simulation for an M/M/c/K system."""
    current_time = 0.0
    servers = [0.0] * c
    num_in_system = 0
    
    total_wait_time, total_system_time = 0.0, 0.0
    total_arrivals, total_dropped, total_entered = 0, 0, 0
    area_queue_length, area_system_length = 0.0, 0.0
    last_event_time = 0.0

    arrival_times = []
    t = 0
    while t < sim_time:
        t += np.random.exponential(scale=1/lam)
        arrival_times.append(t)
    
    queue = []
    
    while arrival_times or any(s > current_time for s in servers):
        next_departure_time = min((s for s in servers if s > current_time), default=float('inf'))
        next_arrival_time = arrival_times[0] if arrival_times else float('inf')
        current_time = min(next_arrival_time, next_departure_time)
        
        time_since_last_event = current_time - last_event_time
        area_queue_length += len(queue) * time_since_last_event
        area_system_length += num_in_system * time_since_last_event
        last_event_time = current_time

        if current_time == next_arrival_time:
            arrival_time = arrival_times.pop(0)
            total_arrivals += 1

            if num_in_system < K:
                num_in_system += 1
                total_entered += 1

                free_server_idx = next((i for i, s_time in enumerate(servers) if s_time <= current_time), -1)
                if free_server_idx != -1:
                    service_time = np.random.exponential(scale=1/mu) # M/M/c/K specific
                    servers[free_server_idx] = current_time + service_time
                    total_system_time += service_time
                else:
                    queue.append(arrival_time)
            else:
                total_dropped += 1
        else: # Departure
            num_in_system -= 1
            if queue:
                arrival_time_from_queue = queue.pop(0)
                service_time = np.random.exponential(scale=1/mu) # M/M/c/K specific
                departing_server_idx = servers.index(next_departure_time)
                servers[departing_server_idx] = current_time + service_time
                
                wait_time = current_time - arrival_time_from_queue
                total_wait_time += wait_time
                total_system_time += wait_time + service_time
        
        if not arrival_times and num_in_system == 0: break

    Pb_emp = total_dropped / total_arrivals if total_arrivals > 0 else 0
    Wq_emp = total_wait_time / total_entered if total_entered > 0 else 0
    W_emp = total_system_time / total_entered if total_entered > 0 else 0
    Lq_emp = area_queue_length / current_time if current_time > 0 else 0
    L_emp = area_system_length / current_time if current_time > 0 else 0
    
    return Pb_emp, L_emp, W_emp, Lq_emp, Wq_emp

def plot_mmck_results(results, c, q_size):
    """Generates and saves plots for the M/M/c/K simulation."""
    lam_range = [r['lambda'] for r in results]
    dist_name = "MMcK_System"
    
    # --- Plot 1: Loss Probability (Primary Metric) ---
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    fig1.suptitle(f'Loss Probability for M/M/c/K (c={c}, Q={q_size})', fontsize=16)
    
    pb_theo = [r['Pb_theo'] for r in results]
    pb_emp = [r['Pb_emp'] for r in results]
    
    ax1.plot(lam_range, pb_theo, 'r-', label='Theoretical $P_B$', linewidth=2)
    ax1.plot(lam_range, pb_emp, 'bo--', label='Empirical $P_B$', markersize=5)
    ax1.set_title('Loss Probability vs. Arrival Rate')
    ax1.set_xlabel('Arrival Rate (λ)')
    ax1.set_ylabel('Loss Probability ($P_B$)')
    ax1.legend()
    ax1.grid(True, linestyle=':')
    ax1.set_ylim(bottom=0)
    
    filename1 = os.path.join(SAVE_DIR, f"{dist_name}_Loss_Probability_c{c}_K{q_size}.png")
    plt.savefig(filename1)
    print(f"-> Loss plot saved as {filename1}")
    plt.close(fig1)

    # --- Plot 2: Other Performance Metrics ---
    fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle(f'Performance Metrics for M/M/c/K (c={c}, Q={q_size})', fontsize=16)
    axes = axes.flatten()

    metrics = [('L', 'System Length'), ('W', 'System Time'), ('Lq', 'Queue Length'), ('Wq', 'Queue Time')]
    for i, (metric, title) in enumerate(metrics):
        ax = axes[i]
        theo_vals = [r[f'{metric}_theo'] for r in results]
        emp_vals = [r[f'{metric}_emp'] for r in results]
        
        ax.plot(lam_range, theo_vals, 'r-', label=f'Theoretical {metric}', linewidth=2)
        ax.plot(lam_range, emp_vals, 'bo--', label=f'Empirical {metric}', markersize=5)
        ax.set_title(f'Average {title}')
        ax.set_xlabel('Arrival Rate (λ)')
        ax.set_ylabel(f'{title} ({metric})')
        ax.legend()
        ax.grid(True, linestyle=':')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    filename2 = os.path.join(SAVE_DIR, f"{dist_name}_Performance_Metrics_c{c}_K{q_size}.png")
    plt.savefig(filename2)
    print(f"-> Metrics plot saved as {filename2}")
    plt.close(fig2)



def run_mmck(mu, c, q_size, sim_time = 100000):
    """Main driver for the M/M/c/K EV charging simulation study."""
    # --- System Parameters ---
    MU = mu          # Average service rate
    C = c            # Number of charging stations (servers)
    Q_SIZE = q_size        # Number of waiting spots in the queue
    K_CAPACITY = C + Q_SIZE # Total system capacity
    SIM_TIME = sim_time # Single long simulation run
    
    # Lambda range between 0 and 1, ensuring rho < 1
    LAMBDA_RANGE = np.linspace(0.05, 0.95, 10)

    print(f"\n--- Running Analysis for: M/M/{C}/K System ---")
    results = []
    for lam in LAMBDA_RANGE:
        print(f"  Simulating λ = {lam:.2f}...")
        
        pb_emp, l_emp, w_emp, lq_emp, wq_emp = run_simulation_run(
            lam, MU, C, K_CAPACITY, Q_SIZE, SIM_TIME
        )
        
        pb_theo, l_theo, w_theo, lq_theo, wq_theo = calculate_mmck_performance(
            lam, MU, C, K_CAPACITY
        )
        
        results.append({
            'lambda': lam,
            'Pb_emp': pb_emp, 'L_emp': l_emp, 'W_emp': w_emp, 'Lq_emp': lq_emp, 'Wq_emp': wq_emp,
            'Pb_theo': pb_theo, 'L_theo': l_theo, 'W_theo': w_theo, 'Lq_theo': lq_theo, 'Wq_theo': wq_theo
        })
        
    plot_mmck_results(results, C, Q_SIZE)
    return results

def excel_writer(results, configs,  excel_path):
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        for result, (mu, c, K) in zip(results, configs):
            df = pd.DataFrame(result)

            # Sheet name describes the configuration
            sheet_name = f"mu{mu}_c{c}_K{K}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"All results saved to {excel_path}")


def main():

    results = []
    configs = []
    # Config_1
    mu = 0.4                # service rate
    c = 2                 # number of servers
    K = 10                # total system capacity
    configs.append((mu, c, K))
    result = run_mmck(mu=mu, c=c, q_size=K)
    results.append(result)

    mu = 0.4                # service rate
    c = 3                 # number of servers
    K = 20                # total system capacity
    configs.append((mu, c, K))
    result = run_mmck(mu=mu, c=c, q_size=K)
    results.append(result)

    mu = 0.3                # service rate
    c = 4                 # number of servers
    K = 30                # total system capacity
    configs.append((mu, c, K))
    result = run_mmck(mu=mu, c=c, q_size=K)
    results.append(result)
   

    mu = 0.25                # service rate
    c = 5                 # number of servers
    K = 30                # total system capacity
    configs.append((mu, c, K))
    result = run_mmck(mu=mu, c=c, q_size=K)
    results.append(result)

    excel_path = os.path.join(SAVE_DIR, "mmck_model_validation.xlsx")
    #excel_writer(results, configs, excel_path)
    

def vary_K_for_loss_prob():
    mu = 0.3                # service rate
    c = 4                 # number of servers
    lam = 0.8
    SIM_TIME = 100000

    K_arr = list(range(2, 21, 2))
    loss_probs = []
    for K in K_arr:
        K_CAPACITY = c + K
        pb_emp, l_emp, w_emp, lq_emp, wq_emp = run_simulation_run(
            lam, mu, c, K_CAPACITY, K, SIM_TIME
        )
        
        pb_theo, l_theo, w_theo, lq_theo, wq_theo = calculate_mmck_performance(
            lam, mu, c, K_CAPACITY
        )

        loss_probs.append({
            "K": K,
            "pb_emp": pb_emp,
            "Pb_theo": pb_theo
        })
    
    excel_path = os.path.join(SAVE_DIR, "mmck_loss_probs.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        df = pd.DataFrame(loss_probs)
        df.to_excel(writer, index = False)

    


if __name__ == "__main__":
    main()
    #vary_K_for_loss_prob()