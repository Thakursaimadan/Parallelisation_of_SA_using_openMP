import subprocess
import os
import time
import matplotlib.pyplot as plt
import numpy as np

# List of thread counts to test.
thread_counts = [1, 2, 4, 6, 8, 10, 12, 16, 20, 32, 64]
execution_times = []

# Path to the compiled executable
executable = "./sa_parallel"

for threads in thread_counts:
    print(f"Running with {threads} thread(s)...")
    # Set the environment variable for OpenMP
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    
    start_time = time.perf_counter()
    # Run the executable; we assume it prints its progress to stdout.
    subprocess.run([executable], env=env)
    end_time = time.perf_counter()
    
    elapsed = end_time - start_time
    print(f"Threads: {threads}, Execution Time: {elapsed:.2f} seconds")
    execution_times.append(elapsed)

# Convert lists to numpy arrays for easier calculations.
thread_counts_np = np.array(thread_counts)
execution_times_np = np.array(execution_times)

# Calculate speedup: speedup(n) = T(1)/T(n)
T1 = execution_times_np[0]
speedups = T1 / execution_times_np

# Calculate estimated parallel fraction (p) using Amdahl's law:
# S(n) = 1 / ((1-p) + p/n)  =>  p = (1 - 1/S(n)) / (1 - 1/n)  for n > 1.
parallel_fraction = []
for i, n in enumerate(thread_counts):
    if n == 1:
        parallel_fraction.append(0.0)
    else:
        sp = speedups[i]
        p_est = (1 - 1/sp) / (1 - 1/n)
        parallel_fraction.append(p_est)
parallel_fraction = np.array(parallel_fraction)

# Print out the computed speedup and parallel fraction for each thread count.
print("\nThread Count | Execution Time (s) | Speedup  | Estimated Parallel Fraction")
print("--------------------------------------------------------------------------")
for i, n in enumerate(thread_counts):
    print(f"{n:12d} | {execution_times[i]:18.2f} | {speedups[i]:7.2f} | {parallel_fraction[i]:26.2f}")

# Plot Execution Time vs. Threads.
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plt.plot(thread_counts, execution_times, marker='o', linestyle='-', color='b')
plt.xlabel('Number of Threads')
plt.ylabel('Execution Time (seconds)')
plt.title('Execution Time vs. Number of Threads')
plt.grid(True)
plt.xscale('log', base=2)
plt.xticks(thread_counts, thread_counts)

# Plot Speedup and Parallel Fraction vs. Threads.
plt.subplot(2, 1, 2)
plt.plot(thread_counts, speedups, marker='s', linestyle='-', color='g', label='Speedup')
plt.plot(thread_counts, parallel_fraction, marker='^', linestyle='--', color='r', label='Estimated Parallel Fraction')
plt.xlabel('Number of Threads')
plt.ylabel('Value')
plt.title('Speedup and Estimated Parallel Fraction vs. Number of Threads')
plt.grid(True)
plt.xscale('log', base=2)
plt.xticks(thread_counts, thread_counts)
plt.legend()

plt.tight_layout()
plt.show()
