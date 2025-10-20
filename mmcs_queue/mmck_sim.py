import numpy as np
import math

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
