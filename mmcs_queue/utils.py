import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


def plot_mmck_results(results, c, q_size, save_dir = None):
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
    
    if save_dir:
        filename1 = os.path.join(save_dir, f"{dist_name}_Loss_Probability_c{c}_K{q_size}.png")
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

    if save_dir:
        filename2 = os.path.join(save_dir, f"{dist_name}_Performance_Metrics_c{c}_K{q_size}.png")
        plt.savefig(filename2)
        print(f"-> Metrics plot saved as {filename2}")
    plt.close(fig2)


def excel_writer(results, configs,  excel_path):
    with pd.ExcelWriter(excel_path, engine="openpyxl", mode="w") as writer:
        for result, (mu, c, K) in zip(results, configs):
            df = pd.DataFrame(result)

            # Sheet name describes the configuration
            sheet_name = f"mu{mu}_c{c}_K{K}"
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"All results saved to {excel_path}")