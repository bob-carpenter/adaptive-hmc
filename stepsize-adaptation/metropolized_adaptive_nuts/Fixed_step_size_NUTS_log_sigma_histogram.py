import os

import bridgestan as bs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotnine as pn
import scipy as sp

import step_size_fixed_NUTS as vn


def create_model_stan_and_json(stan_name, json_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(os.path.dirname(current_dir))
    program_path = f"{project_dir}/stan/" + stan_name + ".stan"
    data_path = f"{project_dir}/stan/" + json_name + ".json"
    model_bs = bs.StanModel(model_lib=program_path, data=data_path)
    return model_bs


def funnel_test(sampler_constructor, N, destination_path, title):
    seed = 12909067
    model = create_model_stan_and_json("funnel", 'funnel')
    D = model.param_unc_num()
    draw_contours = (D == 2)
    rng = np.random.default_rng(seed)
    theta0 = np.zeros(D)
    # theta0[0] = -7.5
    theta0[0] = rng.normal(0, 3)
    theta0[1:D] = rng.normal(scale=np.exp(theta0[0] / 2), size=(D - 1))

    #    theta0 = np.array([-2.0, 1, 1, 1, 1, 1, 1, 1, 1, 1,1])
    # Initialized at a sample from the funnel
    sampler = sampler_constructor(model, rng, theta0)
    draws = sampler.sample_constrained(N)
    plot_draws_and_log_density(draws[:, :2], model.log_density, title, f"{destination_path}/{title}_dimenison_{D}.png",
                               contour=draw_contours)
    plot_hist_log_sigma(draws[:, 0], f"Draws of $\log(\sigma)$ h = {sampler._stepsize}, N = {N}",
                        f"{destination_path}/histogram_log_sigma_{title}_dimenison_{D}.png")
    plot_hist_x_marginal(draws[:, 1], "Histogram of Draws of X",
                         f"{destination_path}/histogram_x_{title}_dimenison_{D}.png")
    plot_hist_log_sigma_pyplot(draws[:, 0], f"Draws of log sigma h = {sampler._stepsize}, N = {N}",
                               f"{destination_path}/histogram_log_sigma_{title}_dimenison_{D}")

    return sampler, draws


def plot_draws_and_log_density(draws, log_density, title, filename, contour=True):
    plt.figure(figsize=(10, 8))
    n_samples = draws.shape[0]
    if contour:

        limits = np.max(np.abs(draws), axis=0)
        x_lim = limits[0] + 1
        y_lim = limits[1] + 1
        # x_lim = 15
        # y_lim = 15
        x = np.linspace(-x_lim, x_lim, 1000)
        y = np.linspace(-y_lim, y_lim, 1000)
        X, Y = np.meshgrid(x, y)

        plt.scatter(draws[:, 0], draws[:, 1], c=np.arange(n_samples), cmap='plasma', s=1)
        plt.colorbar(label='Iteration')
        Z = np.zeros(X.shape)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                Z[i, j] = log_density(np.array([X[i, j], Y[i, j]]))
        log_density_over_draws = np.array([log_density(draw) for draw in draws])
        print(f"Max log density: {np.max(log_density_over_draws)}")
        print(f"Min log density: {np.min(log_density_over_draws)}")
        contour_levels = np.percentile(log_density_over_draws, np.arange(0, 101, 15))
        print(f"Contour levels: {contour_levels}")

        plt.contour(X, Y, Z, levels=contour_levels, cmap='viridis')

    else:
        plt.scatter(draws[:, 1], draws[:, 0], c=np.arange(n_samples), cmap='plasma', s=1)
        plt.colorbar(label='Iteration')

    plt.title(title)
    plt.xlabel("X")
    plt.ylabel("$\log(\sigma)$")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def plot_hist_log_sigma(draws, title, filename):
    x_df = pd.DataFrame({'log(sigma)': draws})
    plot = (
            pn.ggplot(mapping=pn.aes(x='log(sigma)'), data=x_df)
            + pn.geom_histogram(pn.aes(y='..density..', x='log(sigma)'), bins=31, color='black', fill='white')
            + pn.stat_function(fun=lambda x: sp.stats.norm.pdf(x, 0, 3), color='red')
            + pn.scale_x_continuous(limits=(-9, 9), breaks=(-9, -6, -3, 0, 3, 6, 9))
            + pn.labs(x=r'$\log(\sigma)$')  # pn.labs(title=title, x=r'$\log(\sigma)$')
            + pn.theme(
        panel_background=pn.element_rect(fill='white', color='black'),  # Adds border around the panel
        panel_grid_major=pn.element_line(color='black', size=0.5),
        panel_grid_minor=pn.element_line(color='black', size=0.25),
        axis_ticks_major_x=pn.element_line(color='black'),  # Ensure x-axis ticks are visible
        axis_ticks_major_y=pn.element_line(color='black'),  # Ensure y-axis ticks are visible
        axis_line=pn.element_line(color='black'),  # Make axis lines visible on all sides
        # plot_background=pn.element_rect(color='black', fill='None')  # Optional: adds outer border to the entire plot area
    )
    )
    plot.save(filename)
    plot.show()


def plot_hist_log_sigma_pyplot(draws, title, filename):
    plt.hist(draws, bins=31, density=True, color='white', edgecolor='black')

    x = np.linspace(-9, 9, 100)
    plt.plot(x, sp.stats.norm.pdf(x, 0, 3), color='red')
    plt.title(title)
    plt.xlabel("$\log(\sigma)$")
    plt.ylabel("Density")
    plt.savefig(f"{filename}-pyplot.png")
    plt.show()


def plot_hist_x_marginal(draws, title, filename):
    x_df = pd.DataFrame({'X': draws})
    plot = (
            pn.ggplot(mapping=pn.aes(x='X'), data=x_df)
            + pn.geom_histogram(pn.aes(y='..density..', x='X'), bins=31, color='black', fill='white')
            # + pn.stat_function(fun=lambda x: sp.stats.norm.pdf(x, 0, 3), color='red')
            + pn.scale_x_continuous(limits=(-9, 9), breaks=(-9, -6, -3, 0, 3, 6, 9))
            + pn.theme(
        panel_background=pn.element_rect(fill='white'),  # Removes grey background
        panel_grid_major=pn.element_line(color='black', size=0.5),  # Major grid lines to black
        panel_grid_minor=pn.element_line(color='black', size=0.25),  # Minor grid lines to black
        axis_line=pn.element_line(color='black')  # Axis lines to black
    )
    )
    plot.save(filename)
    plot.show()


def plot_funnel_density(x_lim, y_lim, filename):
    model = create_model_stan_and_json("funnel", "one_dimensional_funnel")
    log_density = model.log_density
    x = np.linspace(-x_lim, x_lim, 1000)
    y = np.linspace(-y_lim, y_lim, 1000)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = log_density(np.array([X[i, j], Y[i, j]]))
    contour_levels = np.percentile(Z, np.arange(0, 101, 15))
    plt.figure(figsize=(10, 8))
    plt.contour(X, Y, Z.T, levels=contour_levels, cmap='viridis')
    plt.title("Density of Funnel")
    plt.xlabel("X")
    plt.ylabel("$\log(\sigma)$")
    plt.grid(True)
    plt.savefig(filename)
    plt.show()


def funnel_exact_samples(N, d):
    draws = np.empty((N, d + 1))
    for i in range(N):
        z = np.random.normal(0, 3)
        x = np.random.normal(0, np.exp(z / 2))
        draws[i, 0] = z
        draws[i, 1:] = x
    return draws


def plot_funnel_marginals_exact_samples(N, d, filename):
    model = create_model_stan_and_json("funnel", "one_dimensional_funnel")
    draw_contours = (d == 1)
    draws = funnel_exact_samples(N, d)
    plot_draws_and_log_density(draws[:, :2], model.log_density, "Exact Samples from Funnel", filename,
                               contour=draw_contours)
    plot_hist_log_sigma(draws[:, 0], "Histogram of Draws of log_sigma ", f"{filename}_histogram_log_sigma")
    plot_hist_x_marginal(draws[:, 1], "Histogram of Draws of X", f"{filename}_histogram_x")
    plot_hist_log_sigma_pyplot(draws[:, 0], "Histogram of Draws of log_sigma", f"{filename}_histogram_log_sigma")
    return draws


if __name__ == '__main__':
    current_directory = os.path.dirname(os.path.abspath(__file__))
    destination_directory = f"{current_directory}/results/Fixed_step_size_NUTS_simulation"
    if not os.path.exists(destination_directory):
        os.makedirs(destination_directory)

    step_size = 1 / 4
    nuts_depth = 10
    N = 150_000
    filename = f"TEST_NUTS_with_step_size_{step_size}_and_NUTS_depth_{nuts_depth}_N_{N}"

    sampler_constructor = lambda model, rng, theta0: vn.FixedStepSizeNUTS(model,
                                                                          rng,
                                                                          theta0,
                                                                          np.zeros(model.param_unc_num()),
                                                                          step_size,
                                                                          nuts_depth)
    draws = funnel_test(sampler_constructor,
                        N,
                        destination_directory,
                        filename)
