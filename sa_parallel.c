#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>

#define DIM 1000         // Reduced dimensionality (1,000 dimensions)
#define NB_OF_RUNS 20     // Reduced number of independent runs
#define MAX_ITER 100000  // Reduced maximum iterations per simulated annealing run

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Simulated annealing parameters
#define T_INITIAL 100.0   // Initial temperature
#define T_MIN 1e-8        // Minimum temperature (stopping criterion)
#define COOL_RATE 0.99995 // Cooling rate per iteration

// Domain boundaries for the Rastrigin function
#define LOWER_BOUND -5.12
#define UPPER_BOUND 5.12

// The Rastrigin function: f(x) = 10*n + sum_{i=0}^{n-1}[ x_i^2 - 10*cos(2*pi*x_i) ]
double rastrigin(double *x, int n) {
    double sum = 0.0;
    // Parallelize the sum across dimensions.
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < n; i++) {
        double xi = x[i];
        sum += xi * xi - 10.0 * cos(2.0 * M_PI * xi);
    }
    return 10.0 * n + sum;
}

// Utility function to copy one vector to another.
void copy_vector(double *dest, double *src, int n) {
    for (int i = 0; i < n; i++) {
        dest[i] = src[i];
    }
}

// Initialize a vector with random values uniformly drawn from [LOWER_BOUND, UPPER_BOUND].
void random_vector(double *x, int n, unsigned int *seed) {
    for (int i = 0; i < n; i++) {
        double r = (double) rand_r(seed) / (double) RAND_MAX;
        x[i] = LOWER_BOUND + r * (UPPER_BOUND - LOWER_BOUND);
    }
}

// A single simulated annealing run which returns the best cost found.
double simulated_annealing_run() {
    double *x = malloc(DIM * sizeof(double));
    double *x_new = malloc(DIM * sizeof(double));
    // Create a thread-local seed based on current time and thread id.
    unsigned int seed = (unsigned int) time(NULL) ^ (omp_get_thread_num() + 1);

    // Initialize the candidate solution randomly.
    random_vector(x, DIM, &seed);
    double current_cost = rastrigin(x, DIM);
    double best_cost = current_cost;
    double T = T_INITIAL;

    // Print the initial cost for this run.
    printf("Thread %d: Starting run with initial cost = %f\n", omp_get_thread_num(), current_cost);

    // Main simulated annealing loop.
    for (long iter = 0; iter < MAX_ITER && T > T_MIN; iter++) {
        // Copy current solution to x_new.
        copy_vector(x_new, x, DIM);
        // Choose one random dimension to perturb.
        int idx = rand_r(&seed) % DIM;
        // Perturb the selected element by a small random value in [-0.1, 0.1].
        double delta = ((double) rand_r(&seed) / RAND_MAX) * 0.2 - 0.1;
        x_new[idx] += delta;
        // Clamp to the allowed bounds.
        if (x_new[idx] < LOWER_BOUND) x_new[idx] = LOWER_BOUND;
        if (x_new[idx] > UPPER_BOUND) x_new[idx] = UPPER_BOUND;

        double new_cost = rastrigin(x_new, DIM);
        double diff = new_cost - current_cost;
        // Accept the move if it improves the cost, or with a Boltzmann probability.
        if (diff < 0 || exp(-diff / T) > ((double) rand_r(&seed) / RAND_MAX)) {
            copy_vector(x, x_new, DIM);
            current_cost = new_cost;
            if (current_cost < best_cost) {
                best_cost = current_cost;
            }
        }
        // Cool the temperature.
        T *= COOL_RATE;

        // Print progress every 10% of iterations.
        if (iter % (MAX_ITER / 10) == 0) {
            printf("Thread %d: Iteration %ld, Current Best Cost = %f, Temperature = %e\n", 
                   omp_get_thread_num(), iter, best_cost, T);
        }
    }
    // Final print for this run.
    printf("Thread %d: Finished run with best cost = %f\n", omp_get_thread_num(), best_cost);

    free(x);
    free(x_new);
    return best_cost;
}

int main() {
    double best_overall = 1e30;
    double start_time = omp_get_wtime();

    printf("Starting simulated annealing with %d runs, DIM=%d, MAX_ITER=%d\n", NB_OF_RUNS, DIM, MAX_ITER);

    // Run many independent simulated annealing runs in parallel.
    #pragma omp parallel for reduction(min:best_overall)
    for (int i = 0; i < NB_OF_RUNS; i++) {
        double cost = simulated_annealing_run();
        if (cost < best_overall)
            best_overall = cost;
    }
    double end_time = omp_get_wtime();

    printf("Best cost found: %f\n", best_overall);
    printf("Total execution time: %f seconds\n", end_time - start_time);
    return 0;
}


