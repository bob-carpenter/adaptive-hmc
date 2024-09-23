import os
import plotnine as pn
import pandas as pd
import scipy as sp
import numpy as np
import step_size_fixed_NUTS as vn
import Fixed_step_size_NUTS_log_sigma_histogram as fn

def plot_hist_log_sigma(draws, title, filename):
    '''
    Plots histogram of draws of log(sigma) variable in funnel model
    Args:
        draws: Draws of log(sigma) variable
        title: Plot title
        filename: Destination filename
    Returns:
        None
    '''
    x_df = pd.DataFrame({'log(sigma)': draws})
    plot = (
            pn.ggplot(mapping=pn.aes(x='log(sigma)'), data=x_df)
            + pn.geom_histogram(pn.aes(y='..density..', x='log(sigma)'), bins=62, color='black', fill='white')
            + pn.stat_function(fun=lambda x: sp.stats.norm.pdf(x, 0, 1), color='red')
            + pn.scale_x_continuous(limits=(-9, 9), breaks=(-9, -6, -3, 0, 3, 6, 9))
            + pn.labs(x=r'$\omega$')  # pn.labs(title=title, x=r'$\omega$')
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

    if not os.path.exists(f'../metropolized_adaptive_nuts/results/Fixed_step_size_NUTS_simulation/standard_normal'):
        os.makedirs(f'../metropolized_adaptive_nuts/results/Fixed_step_size_NUTS_simulation/standard_normal')
    plot.save(f'results/Fixed_step_size_NUTS_simulation/standard_normal/{filename}')
    plot.show()

'''
This script 
'''
model = fn.create_model_stan_and_json("one_dimensional_standard_normal", 'one_dimensional_standard_normal')
D = model.param_unc_num()
print(f"Dimension: {D}")
seed = 12909067
rng = np.random.default_rng(seed)
N = 40000
stepsize = 0.1

sampler = vn.FixedStepSizeNUTS(model,
                           rng,
                           np.zeros(D),
                            np.zeros(D),
                           stepsize,
                           10)
draws = sampler.sample_constrained(N)
print(2**np.array(sampler.observed_heights))
mean_path_length = np.mean(stepsize*(2**np.array(sampler.observed_heights) - 1))/np.pi
print(f"Mean path length: {mean_path_length}")

plot_hist_log_sigma(draws[:, 0],
                          f"Draws of $\omega$ h = {sampler._stepsize}, N = {N}",
                    "Fixed_step_size_standard_normal.png")
