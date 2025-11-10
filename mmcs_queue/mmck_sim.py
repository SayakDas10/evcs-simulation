import numpy as np
import math
from collections import Counter
from utils import compute_electrical_loading
import pandas as pd

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

def run_simulation_run(lam, mu, c, K, q_size, sim_time = 100000):
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



def run_simulation_run_new(lam, mu, c, K, q_size, sim_time = 100000, get_stats_dict = False):
    """
    Event-driven simulation for M/M/c/K queue.
    Returns empirical performance metrics + queue length evolution + distribution + utilization.
    """
    current_time = 0.0
    servers = [0.0] * c
    num_in_system = 0
    
    total_wait_time, total_system_time = 0.0, 0.0
    total_arrivals, total_dropped, total_entered = 0, 0, 0
    area_queue_length, area_system_length = 0.0, 0.0
    last_event_time = 0.0

    arrival_times = []
    service_times_record = []
    t = 0
    while t < sim_time:
        t += np.random.exponential(scale=1/lam)
        arrival_times.append(t)
    
    queue = []
    queue_lengths = []        
    queue_length_counts = Counter()  

    busy_time_per_server = [0.0] * c  # To calculate utilization

    while arrival_times or any(s > current_time for s in servers):
        next_departure_time = min((s for s in servers if s > current_time), default=float('inf'))
        next_arrival_time = arrival_times[0] if arrival_times else float('inf')
        current_time = min(next_arrival_time, next_departure_time)
        
        time_since_last_event = current_time - last_event_time
        area_queue_length += len(queue) * time_since_last_event
        area_system_length += num_in_system * time_since_last_event
        last_event_time = current_time

        # Record queue length over time
        queue_lengths.append((current_time, len(queue)))
        queue_length_counts[len(queue)] += 1

        if current_time == next_arrival_time:  # ARRIVAL
            arrival_time = arrival_times.pop(0)
            total_arrivals += 1

            if num_in_system < K:
                num_in_system += 1
                total_entered += 1

                free_server_idx = next((i for i, s_time in enumerate(servers) if s_time <= current_time), -1)
                if free_server_idx != -1:
                    service_time = np.random.exponential(scale=1/mu)
                    service_times_record.append(service_time)
                    servers[free_server_idx] = current_time + service_time
                    busy_time_per_server[free_server_idx] += service_time
                    total_system_time += service_time
                else:
                    queue.append(arrival_time)
            else:
                total_dropped += 1

        else:  # DEPARTURE
            num_in_system -= 1
            if queue:
                arrival_time_from_queue = queue.pop(0)
                service_time = np.random.exponential(scale=1/mu)
                service_times_record.append(service_time)
                departing_server_idx = servers.index(next_departure_time)
                servers[departing_server_idx] = current_time + service_time
                busy_time_per_server[departing_server_idx] += service_time
                
                wait_time = current_time - arrival_time_from_queue
                total_wait_time += wait_time
                total_system_time += wait_time + service_time
            else:
                departing_server_idx = servers.index(next_departure_time)
                servers[departing_server_idx] = 0.0

        if not arrival_times and num_in_system == 0:
            break

    # === Empirical metrics ===
    Pb_emp = total_dropped / total_arrivals if total_arrivals > 0 else 0
    Wq_emp = total_wait_time / total_entered if total_entered > 0 else 0
    W_emp = total_system_time / total_entered if total_entered > 0 else 0
    Lq_emp = area_queue_length / current_time if current_time > 0 else 0
    L_emp = area_system_length / current_time if current_time > 0 else 0

    # === Queue length distribution (normalized) ===
    total_obs = sum(queue_length_counts.values())
    q_len_distribution = {k: v / total_obs for k, v in sorted(queue_length_counts.items())}

    # === Server utilization ===
    utilization = [busy_time / current_time for busy_time in busy_time_per_server]
    avg_utilization = np.mean(utilization)

    stats_dict = {
        "Pb_emp": Pb_emp,
        "L_emp": L_emp,
        "W_emp": W_emp,
        "Lq_emp": Lq_emp,
        "Wq_emp": Wq_emp,
        "queue_length_time": queue_lengths,
        "queue_length_distribution": q_len_distribution,
        "server_utilization": utilization,
        "avg_utilization": avg_utilization,
        "svc_times": service_times_record
    }

    return stats_dict


def run_mmck_hourly(lam_hourly, mu, c, K, sim_minutes=1440):
    """
    Cycle-driven M/M/c/K simulator for 24h period with minute-level steps.
    lam_hourly: array of 24 hourly arrival rates (per minute)
    mu: service rate per minute
    c: number of servers
    K: system capacity
    sim_minutes: total simulation time (default = 1440 minutes for 24h)
    """
    queue = []
    servers = [0.0] * c
    num_in_system = 0

    arrivals_record = np.zeros(sim_minutes)
    queue_len_record = np.zeros(sim_minutes)
    utilization_record = np.zeros(sim_minutes)
    
    current_time = 0.0

    for t in range(sim_minutes):
        hour_idx = int(t // 60)
        lam_t = lam_hourly[hour_idx]  # arrivals/minute at current hour
        
        # Generate arrivals during this minute (Poisson)
        arrivals_this_min = np.random.poisson(lam_t)
        arrivals_record[t] = arrivals_this_min
        
        for _ in range(arrivals_this_min):
            if num_in_system < K:
                num_in_system += 1
                free_server = next((i for i, s in enumerate(servers) if s <= current_time), -1)
                if free_server != -1:
                    service_time = np.random.exponential(1/mu)
                    servers[free_server] = current_time + service_time
                else:
                    queue.append(current_time)
        
        # Check for completed services
        for i in range(c):
            if servers[i] <= current_time:
                if queue:
                    arrival_from_queue = queue.pop(0)
                    service_time = np.random.exponential(1/mu)
                    servers[i] = current_time + service_time
                else:
                    servers[i] = current_time
        
        # Record stats
        queue_len_record[t] = len(queue)
        busy_servers = sum(1 for s in servers if s > current_time)
        utilization_record[t] = busy_servers / c
        
        current_time += 1  # advance 1 minute

    return arrivals_record, queue_len_record, utilization_record