import numpy as np
import json
import os
from mmck_sim import run_simulation_run, calculate_mmck_performance

SAVE_DIR = "/home/saksham/samsad/MLSpace/MTech Sem3/EV-charging-impact/ev_phase2_results"

def load_json(json_path):
    with open(json_path, "r") as f:
        evcs_config = json.load(f)
    return evcs_config

def save_json(stats_dict, json_path):
    with open(json_path, "w") as f:
        json.dump(stats_dict, f, indent=4)

def generate_stats(evcs_config):
    stats_dict = {}

    for evcs_id in evcs_config:
        config = evcs_config[evcs_id]
        stats_dict[evcs_id] = config

        print(f"Generating EVCS Stats for - {evcs_id}, config: {config} ")
        lam = config["lambda"]
        MU = config["mu"]          # Average service rate
        C = config["c"]            # Number of charging stations (servers)
        Q_SIZE = config["K"]        # Number of waiting spots in the queue
        K_CAPACITY = C + Q_SIZE # Total system capacity
        SIM_TIME = 100000 # Single long simulation run

        pb_emp, l_emp, w_emp, lq_emp, wq_emp = run_simulation_run(
            lam, MU, C, K_CAPACITY, Q_SIZE, SIM_TIME
        )
        
        pb_theo, l_theo, w_theo, lq_theo, wq_theo = calculate_mmck_performance(
            lam, MU, C, K_CAPACITY
        )
        
        results = {
            'lambda': lam,
            'Pb_emp': pb_emp, 'L_emp': l_emp, 'W_emp': w_emp, 'Lq_emp': lq_emp, 'Wq_emp': wq_emp,
            'Pb_theo': pb_theo, 'L_theo': l_theo, 'W_theo': w_theo, 'Lq_theo': lq_theo, 'Wq_theo': wq_theo
        }
        
        for key, val in results.items():
            stats_dict[evcs_id][key] = val
    
    return stats_dict

if __name__ == "__main__":
    evcs_config_json_path = "/home/saksham/samsad/MLSpace/MTech Sem3/EV-charging-impact/ev_phase_2_results/evcs_config.json"
    evcs_config_dict = load_json(evcs_config_json_path)
    evcs_stats_dict = generate_stats(evcs_config_dict)
    save_json(stats_dict=evcs_stats_dict, 
              json_path=os.path.join(os.path.dirname(evcs_config_json_path), "evcs_stats.json"))