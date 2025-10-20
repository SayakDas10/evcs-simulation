import numpy as np
import os
import pandas as pd
from mmck_sim import run_simulation_run, calculate_mmck_performance
from utils import plot_mmck_results, excel_writer

SAVE_DIR = "/home/saksham/samsad/MLSpace/MTech Sem3/MnS_Project/evcs-simulation/evcs_phase2_results"

def run_mmck(mu, c, q_size, sim_time = 100000, plot_results = True):
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
    
    if plot_results:
        plot_mmck_results(results, C, Q_SIZE, save_dir=SAVE_DIR)
    return results


def run_model_verif(plot_results = True, write_to_excel = True):

    results = []
    configs = []
    # Config_1
    mu = 0.4                # service rate
    c = 2                 # number of servers
    K = 10                # total system capacity
    configs.append((mu, c, K))
    result = run_mmck(mu=mu, c=c, q_size=K, plot_results=plot_results)
    results.append(result)

    mu = 0.4                # service rate
    c = 3                 # number of servers
    K = 20                # total system capacity
    configs.append((mu, c, K))
    result = run_mmck(mu=mu, c=c, q_size=K, plot_results=plot_results)
    results.append(result)

    mu = 0.3                # service rate
    c = 4                 # number of servers
    K = 30                # total system capacity
    configs.append((mu, c, K))
    result = run_mmck(mu=mu, c=c, q_size=K, plot_results=plot_results)
    results.append(result)
   

    mu = 0.25                # service rate
    c = 5                 # number of servers
    K = 30                # total system capacity
    configs.append((mu, c, K))
    result = run_mmck(mu=mu, c=c, q_size=K, plot_results=plot_results)
    results.append(result)

    if write_to_excel:
        excel_path = os.path.join(SAVE_DIR, "mmck_model_validation.xlsx")
        excel_writer(results, configs, excel_path)
    

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
    run_model_verif()
    #vary_K_for_loss_prob()