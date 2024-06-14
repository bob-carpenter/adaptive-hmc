import step_size_adapt_NUTS_coarse_fine as spn
import cmdstanpy as csp
import numpy as np
import bridgestan as bs
import plotnine as pn
import pandas as pd
import logging
import traceback
import warnings
import os
import time
from pathlib import Path

# seed = 20070707062819965829111
#seed = 20070707062819965829111
seed = 200707070628199

def traceplot(chain):
    """Return a traceplot for the specified single chain."""
    df = pd.DataFrame({"m": range(len(chain)), "theta": chain})
    plot = (
            pn.ggplot(df, pn.aes(x="m", y="theta"))
            + pn.geom_line()
            + pn.labs(x="Iteration", y="Parameter Value", title="Trace Plot of MCMC Chain")
            + pn.theme_minimal()
    )
    return plot


def histogram(xs):
    """Return a histogram plot of the specified values with a vertical blue line at the mean."""
    df = pd.DataFrame({"x": xs})
    plot = (
            pn.ggplot(df, pn.aes(x="x"))
            + pn.geom_histogram(bins=100)
            + pn.geom_vline(xintercept=np.mean(xs), color="blue", size=1)
    )
    return plot


def format_time(start, end):
    elapsed_time = end - start
    minutes, seconds = divmod(elapsed_time, 60)
    hours, minutes = divmod(minutes, 60)
    return seconds, minutes, hours


def traceplot(chain):
    """Return a traceplot for the specified single chain."""
    df = pd.DataFrame({"m": range(len(chain)), "theta": chain})
    plot = (
            pn.ggplot(df, pn.aes(x="m", y="theta"))
            + pn.geom_line()
            + pn.labs(x="Iteration", y="Parameter Value", title="Trace Plot of MCMC Chain")
        # + pn.theme_minimal()
    )
    return plot


if __name__ == '__main__':
    n_samples = 16000
    start_time = time.time()
    # Initial attempt to just get something running

    model_name = "normal"
    program_path = Path("../stan") / f"{model_name}.stan"
    data_path = Path("../stan") / f"{model_name}.json"
    print(f"Program path: {program_path.resolve()}")
    print(f"Data path: {data_path.resolve()}")
    # program_path = "./stan/normal.stan"
    # data_path = "./stan/normal.json"

    model_bs = bs.StanModel(model_lib=program_path, data=data_path, capture_stan_prints=False)
    # print("Sanity checking the methods\n")
    # print(f"Log density: {model_bs.log_density(np.ones(model_bs.param_unc_num()))} \n")
    # print(f"Log density gradient: {model_bs.log_density_gradient(np.ones(model_bs.param_unc_num()))}\n")
    rng = np.random.default_rng(seed)
    theta0 = rng.normal(size=model_bs.param_unc_num())
    sampler = spn.StepadaptNutsCoarseFineSampler(model_bs,
                                                 rng,
                                                 theta0,
                                                 0.99,
                                                 1,
                                                 8,
                                                 9)
    thetas = sampler.sample(n_samples)
    #print(f"Here are the actual samples: {thetas}")
    print(np.mean(thetas, axis=0))
    print(np.var(thetas, axis=0))
    end_time = time.time()
    seconds, minutes, hours = format_time(start_time, end_time)
    print(f"The evaluation took: {hours} h : {minutes} m: {seconds} s for {n_samples} samples")
    plot = traceplot(thetas[:, 0])
    plot.show()

