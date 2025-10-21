import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from mmck_sim import run_simulation_run, calculate_mmck_performance, run_simulation_run_new
from utils import plot_mmck_results, excel_writer, compute_electrical_loading

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
    mu = 0.4               
    c = 2                 
    K = 10                
    configs.append((mu, c, K))
    result = run_mmck(mu=mu, c=c, q_size=K, plot_results=plot_results)
    results.append(result)

    mu = 0.4               
    c = 3                 
    K = 20                
    configs.append((mu, c, K))
    result = run_mmck(mu=mu, c=c, q_size=K, plot_results=plot_results)
    results.append(result)

    mu = 0.3               
    c = 4                 
    K = 30                
    configs.append((mu, c, K))
    result = run_mmck(mu=mu, c=c, q_size=K, plot_results=plot_results)
    results.append(result)
   

    mu = 0.25                
    c = 5                 
    K = 30                
    configs.append((mu, c, K))
    result = run_mmck(mu=mu, c=c, q_size=K, plot_results=plot_results)
    results.append(result)

    if write_to_excel:
        excel_path = os.path.join(SAVE_DIR, "mmck_model_validation.xlsx")
        excel_writer(results, configs, excel_path)
    

def vary_K_for_loss_prob():
    mu = 0.3                 # service rate
    c_list = [3, 6, 10]      # number of servers to test
    lam_arr = [0.8, 1.2, 1.8]
    SIM_TIME = 100000

    K_arr = list(range(2, 21, 2))  # buffer capacities to test
    excel_path = os.path.join(SAVE_DIR, "mmck_loss_probs.xlsx")

    
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        for lam, c in zip(lam_arr, c_list):
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
                    "pb_theo": pb_theo
                })

            
            df = pd.DataFrame(loss_probs)
            sheet_name = f"c_{c}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Saved results for all c values to: {excel_path}")



def analyze_and_save_results(results, save_path="mmck_system_dynamics.xlsx"):
    """
    Plots and saves M/M/c/K simulation analysis results with an average queue length bar.
    
    Args:
        results (dict): Output of run_simulation_run()
        save_path (str): Output Excel file path (default: 'mmck_results.xlsx')
    """
    
    avg_queue_length = results["Lq_emp"]
    queue_time_df = pd.DataFrame(results["queue_length_time"], columns=["Time", "Queue_Length"])
    
    plt.figure(figsize=(8, 4))
    plt.plot(queue_time_df["Time"], queue_time_df["Queue_Length"], lw=1.5, color='steelblue')
    plt.axhline(y=avg_queue_length, color='red', linestyle='--', linewidth=1.5,
                label=f'Avg Queue Length = {avg_queue_length:.2f}')
    plt.title("Queue Length vs Time")
    plt.xlabel("Time")
    plt.ylabel("Queue Length")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    dist_df = pd.DataFrame(list(results["queue_length_distribution"].items()), 
                           columns=["Queue_Length", "Probability"])
    
    plt.figure(figsize=(6, 4))
    plt.bar(dist_df["Queue_Length"], dist_df["Probability"], 
            color='skyblue', edgecolor='black', label="Distribution")
    plt.axvline(x=avg_queue_length, color='red', linestyle='--', linewidth=1.5,
                label=f'Avg Queue Length = {avg_queue_length:.2f}')
    plt.title("Queue Length Distribution")
    plt.xlabel("Queue Length")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    util_df = pd.DataFrame({
        "Server_ID": [f"Server_{i+1}" for i in range(len(results["server_utilization"]))],
        "Utilization": results["server_utilization"]
    })
    
    plt.figure(figsize=(6, 4))
    plt.bar(util_df["Server_ID"], util_df["Utilization"], 
            color='lightgreen', edgecolor='black')
    plt.title("Server Utilization")
    plt.xlabel("Server")
    plt.ylabel("Utilization Fraction")
    plt.ylim(0, 1)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()
    
    save_path = os.path.join(SAVE_DIR, save_path)
    with pd.ExcelWriter(save_path, engine="openpyxl") as writer:
        queue_time_df.to_excel(writer, sheet_name="Queue_vs_Time", index=False)
        dist_df.to_excel(writer, sheet_name="Queue_Distribution", index=False)
        util_df.to_excel(writer, sheet_name="Server_Utilization", index=False)
        
        summary = {
            "Metric": ["Pb_emp", "L_emp", "W_emp", "Lq_emp", "Wq_emp", "Avg_Utilization"],
            "Value": [
                results["Pb_emp"], results["L_emp"], results["W_emp"],
                results["Lq_emp"], results["Wq_emp"], results["avg_utilization"]
            ]
        }
        summary_df = pd.DataFrame(summary)
        summary_df.to_excel(writer, sheet_name="Summary_Metrics", index=False)
    
    print(f"Results saved successfully to: {save_path}")

def vary_c_for_svr_utilization():
    mu = 0.2               
    K = 20                
    lam = 1.0
    SIM_TIME = 100000

    c_arr = np.arange(2, 21, 2)
    utiliz = []
    for c in c_arr:
        K_CAPACITY = c + K
        sim_results = run_simulation_run_new(
            lam, mu, c, K_CAPACITY, K, SIM_TIME
        )

        utiliz.append({
            "c":c,
            "svr_utilz_emp": np.mean(sim_results["server_utilization"])

        })
    
    excel_path = os.path.join(SAVE_DIR, "mmck_server_utilz.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        df = pd.DataFrame(utiliz)
        df.to_excel(writer, index = False)

    svr_utilz_emp = [x["svr_utilz_emp"] for x in utiliz]

    plt.plot(c_arr, svr_utilz_emp) 
    plt.show()

def compare_electrical_loading():
    q_size = 20        
    lam = 0.85
    low_svc_times = []
    med_svc_times = []
    high_svc_times = []

    charger_type = "low"
    for c in range(1, 21):
                    
        rho = 0.5
        mu = lam/(rho * c)
        results = run_simulation_run_new(lam=lam, mu=mu, c=c, q_size=q_size, K=c+q_size, sim_time=10000)
        low_svc_times.append(np.mean(results["svc_times"]))

        charger_type = "medium"
                    
        rho = 0.7
        mu = lam/(rho * c)
        results = run_simulation_run_new(lam=lam, mu=mu, c=c, q_size=q_size, K=c+q_size, sim_time=10000)
        med_svc_times.append(np.mean(results["svc_times"]))

        charger_type = "high"
                   
        rho = 0.9
        mu = lam/(rho * c)
        results = run_simulation_run_new(lam=lam, mu=mu, c=c, q_size=q_size, K=c+q_size, sim_time=10000)
        high_svc_times.append(np.mean(results["svc_times"]))


    low_svc_load = list(map(lambda x: compute_electrical_loading(x, "low"), low_svc_times))
    med_svc_load = list(map(lambda x: compute_electrical_loading(x, "medium"), med_svc_times))
    high_svc_load = list(map(lambda x: compute_electrical_loading(x, "high"), high_svc_times))


    excel_result = {
        "vehicle": list(range(1, 21)),
        "low_svc_load": low_svc_load,
        "med_svc_load": med_svc_load,
        "high_svc_load": high_svc_load
    }

    excel_path = os.path.join(SAVE_DIR, "mmck_electrical_load.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        df = pd.DataFrame(excel_result)
        df.to_excel(writer, index = False)



if __name__ == "__main__":
    #run_model_verif()
    vary_K_for_loss_prob()
    # mu = 0.5          # Average service rate
    # c = 2            # Number of charging stations (servers)
    # q_size = 20        # Number of waiting spots in the queue
    # lam = 0.85
    # results = run_simulation_run_new(lam=lam, mu=mu, c=c, q_size=q_size, K=c+q_size, sim_time=10000)
    # analyze_and_save_results(results)
    vary_c_for_svr_utilization()

    compare_electrical_loading()

