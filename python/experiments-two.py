import one_way_multinomial as ows
import cmdstanpy as csp
import numpy as np
import bridgestan as bs
import plotnine as pn
import pandas as pd
import logging
import traceback
import warnings

def learning_curve_plot():
    seed = 189236576
    stepsize = 0.25
    D = 2
    program_path, data_path = "../stan/normal.stan", "../stan/normal.json"
    model_bs = bs.StanModel(model_lib=program_path, data=data_path)
    rng = np.random.default_rng(seed)
    theta0 = rng.normal(loc=0, scale=1, size=D)  # draw from stationary distribution
    sampler = ows.OneWaySampler(
        model=model_bs, stepsize=stepsize, rng=rng, steps=3, theta=theta0
    )
    N = 10_000
    draws = sampler.sample(N)
    param_draws = np.stack([draw[0] for draw in draws])
    means = np.mean(param_draws, axis=0)
    std_devs = np.std(param_draws, axis=0)
    print(f"{means=}  {std_devs=}")
    
    cumsum_draws = np.cumsum(draws, axis=0)
    divisors = np.arange(1, draws.shape[0] + 1).reshape(-1, 1)
    abs_err = np.abs(cumsum_draws) / divisors
    avg_abs_err = np.mean(abs_err, axis=1)
    
    draws_sq = draws ** 2
    cumsum_draws_sq = np.cumsum(draws_sq, axis=0)
    abs_err_sq = np.abs(cumsum_draws_sq / divisors - 1)  # E[ChiSquare(1)] = 1
    avg_abs_err_sq = np.mean(abs_err_sq, axis=1)

    errs = np.concatenate([avg_abs_err, avg_abs_err_sq])
    estimands = np.concatenate([np.array(["theta"] * N), np.array(["theta**2"] * N)])
    iteration = np.arange(1, len(avg_abs_err) + 1)
    iterations = np.concatenate([iteration, iteration])
    df = pd.DataFrame(
        {"iteration": iterations, "|error|": errs, "estimand": estimands}
    )
    lines_df = pd.DataFrame(
        {
            "estimand": np.array(["theta", "theta**2"]),
            "x": np.array([10, 10]),
            "y": np.array([1 / np.sqrt(10), np.sqrt(2) / np.sqrt(10)]),
            "xend": np.array([N, N]),
            "yend": np.array([1 / np.sqrt(N), np.sqrt(2) / np.sqrt(N)]),
        }
    )
    plot = (
        pn.ggplot(df, pn.aes(x="iteration", y="|error|"))
        + pn.geom_line()
        + pn.scale_x_log10(limits=(10, N))
        + pn.scale_y_log10()
        + pn.geom_segment(
            data=lines_df,
            mapping=pn.aes(x="x", y="y", xend="xend", yend="yend"),
            linetype="dotted",
        )
        + pn.facet_wrap("~ estimand")
    )
    plot.save(filename="learning_curve.pdf", width=6, height=3)

learning_curve_plot()    
    
# def orbital_experiment(
#         program,
#         stepsize,
#         steps,
#         seed):
#     program_path = program + '.stan'
#     data_path = program + '.json'
#     model_bs = bs.StanModel(modellib = program_path, data = data_path, capture_stan_prints=False)
#     rng = np.random.default_rng(seed)
#     theta = rng.normal(
#     sampler = os.OrbitalSampler(model_bs, stepsize=0.3, rng=rng, 
    
# orbital_experiment('normal', 0.3, 8, 12349876)
        
