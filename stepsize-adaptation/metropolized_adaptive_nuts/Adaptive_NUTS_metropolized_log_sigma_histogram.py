import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import Fixed_step_size_NUTS_log_sigma_histogram as fn
import step_size_adapt_NUTS_metropolized as nuts_b_prime_transform
import time

def plot_average_acceptance_ratios(y_values, positions, filename):
    x_range = 12
    x_bins = np.linspace(-x_range, x_range, x_range * 10 + 1)
    bin_indices = np.digitize(positions, x_bins)

    bin_means = []
    bin_midpoints = []

    for i in range(1, len(x_bins)):
        bin_mask = (bin_indices == i)
        if np.any(bin_mask):
            bin_mean = np.mean(y_values[bin_mask])
            bin_midpoint = (x_bins[i - 1] + x_bins[i]) / 2
            bin_means.append(bin_mean)
            bin_midpoints.append(bin_midpoint)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(bin_midpoints, bin_means, marker='o', linestyle='-')
    plt.xlabel('$\log(\sigma)$ Values')
    plt.ylabel('Average Acceptance Ratio')
    plt.title('Average Ratio Value Dependent on Binned $\log(\sigma)$ Value')
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


if __name__ == '__main__':
    start_time = time.time()
    current_directory = os.path.dirname(os.path.abspath(__file__))
    destination_directory = f'{current_directory}/results/NUTS_b_prime_transform'

    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    stepsize = 1 / 2
    nuts_depth = 10
    max_step_size_search_depth = 10
    min_acceptance_rate = 0.7
    distribution = 'funnel'
    N = 150_000
    filename = f"stepsize_{stepsize}_NUTS_depth_{nuts_depth}_stepdepth_{max_step_size_search_depth}_min_accept_{min_acceptance_rate}_N_{N}"

    sampler_constructor_adjust = lambda model, rng, theta0: nuts_b_prime_transform.StepAdaptNUTSMetro(model,
                                                                                                      rng,
                                                                                                      theta0,
                                                                                                      np.zeros(
                                                                                                        model.param_unc_num()
                                                                                                      ),
                                                                                                      min_acceptance_rate,
                                                                                                      stepsize,
                                                                                                      max_step_size_search_depth,
                                                                                                      nuts_depth)


    sampler, draws = fn.funnel_test(sampler_constructor_adjust,
                                    N,
                                    destination_directory,
                                    filename)

    print(f"For sampler name {sampler._name} and N = {N} we have: ")
    print(f"Number of rejections: {sampler._no_return_rejections}")
    print(f'Average number of halvings = {np.mean(sampler._adapted_step_sizes)}')

    plot_average_acceptance_ratios(
        np.array(sampler.acceptance_ratios()),
        draws[:-1, 0],
        f'{destination_directory}/Acceptance_ratios_step_size_{stepsize}_NUTS_depth_{nuts_depth}_N_{N}_min_accept{min_acceptance_rate}.png'
    )
    end_time = time.time()
    print(f"Time taken: {end_time - start_time}")

