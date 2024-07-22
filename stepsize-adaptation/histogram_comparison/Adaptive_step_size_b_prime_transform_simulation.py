import matplotlib.pyplot as plt
import numpy as np
import Fixed_step_size_NUTS_simulation as fn
import step_size_adapt_NUTS_b_prime_transform as nuts_b_prime_transform

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
    destination_path = '../results/NUTS_b_prime_transform'
    if not os.path.exists(destination_path):
        os.makedirs(destination_path)

    stepsize = 1 / 4
    nuts_depth = 10
    max_step_size_search_depth = 10
    min_acceptance_rate = 0.7
    funnel = 'funnel'

    log_sig_initial = 'random'
    if not log_sig_initial == 'random':
        theta_initial = np.zeros(11)
        theta_initial[0] = log_sig_initial

    sampler_constructor_adjust = lambda model, rng, theta0: nuts_b_prime_transform.NUTSBprimeTransform(model,
                                                                                                       rng,
                                                                                                       theta0,
                                                                                                       np.zeros(
                                                                                                           model.param_unc_num()),
                                                                                                       min_acceptance_rate,
                                                                                                       stepsize,
                                                                                                       max_step_size_search_depth,
                                                                                                       nuts_depth)

for N in [50_000]:
    sampler, draws = fn.funnel_test(sampler_constructor_adjust,
                                    N,
                                    destination_path,
                                    f"Step_size_{stepsize}_NUTS_depth_{nuts_depth}_N_{N}_min_acceptance_{min_acceptance_rate}_initial_{log_sig_initial}",
                                    funnel)
    print(f"For sampler name {sampler._name} and N = {N} we have: ")
    print(f"Number of rejections: {sampler._no_return_rejections}")
    print(f'Average number of halvings = {np.mean(sampler._adapted_step_sizes)}')
    plot_average_acceptance_ratios(
        np.array(sampler.acceptance_ratios()),
        draws[:-1, 0],
        f'{destination_path}/Acceptance_ratios_step_size_{stepsize}_NUTS_depth_{nuts_depth}_N_{N}_min_accept{min_acceptance_rate}.png'
    )
    print(draws)
