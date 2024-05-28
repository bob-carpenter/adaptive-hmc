import gist_sampler as gs
import multinomial_sampler as ms
import stepsize_adapt_sampler as sas
import gist_spectral_step_sampler as gws
import progressive_turnaround as pta
import cmdstanpy as csp
import numpy as np
import scipy as sp
import bridgestan as bs
import plotnine as pn
import pandas as pd
import logging
import traceback
import warnings


class NutsFit:
    """Object to store te result of a NUTS run"""
    def __init__(
        self,
        draws_dict,
        draws_array,
        theta,  # constrained
        theta_sd,
        theta_sq,
        theta_sq_sd,
        metric,
        stepsize,
    ):
        self.stepsize_ = stepsize
        self.draws_dict_ = draws_dict
        self.draws_array_ = draws_array
        self.theta_ = theta
        self.theta_sd_ = theta_sd
        self.theta_sq_ = theta_sq
        self.theta_sq_sd_ = theta_sq_sd
        self.metric_ = metric
        self.num_unc_params_ = np.shape(self.draws_array_[1, :])[0]


class GistFit:
    """Object to store te result of a GIST run"""
    def __init__(
        self,
        sampler,
        frac,
        stepsize,
        steps,
        prop_reject,
        prop_no_return,
        rmse,
        rmse_sq,
        msjd,
    ):
        self.sampler_ = sampler
        self.frac_ = frac
        self.stepsize_ = stepsize
        self.steps_ = steps
        self.prop_reject_ = prop_reject
        self.prop_no_return_ = prop_no_return
        self.rmse_ = rmse
        self.rmse_sq_ = rmse_sq
        self.msjd_ = msjd


def stop_griping():
    """Turn off warnings so that we only see real errors."""
    warnings.filterwarnings(
        "ignore", message="Loading a shared object .* that has already been loaded.*"
    )
    csp.utils.get_logger().setLevel(logging.ERROR)


def flatten_dict_values(data_dict):
    """Return an aray of entries in the dictionary concatenated together."""
    flattened_list = []
    for value in data_dict.values():
        if isinstance(value, (np.ndarray)):
            flattened_list.extend(value.flatten())
        elif isinstance(value, (list, tuple)):
            flattened_list.extend(value)
        else:
            flattened_list.append(value)
    print(flattened_list)
    return np.array(flattened_list)


def sq_jumps(draws):
    """Return an array of squared distances between consecutive draws."""
    M = np.shape(draws)[0]
    jumps = draws[range(1, M), :] - draws[range(0, M - 1), :]
    return [np.dot(jump, jump) for jump in jumps]


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


def histogram_vs_normal_density(xs):
    """Return a histogram plot of the specified value with a blue overlay of a standard normal density."""
    df = pd.DataFrame({"x": xs})
    plot = (
        pn.ggplot(df, pn.aes(x="x"))
        + pn.geom_histogram(
            pn.aes(y="..density.."), bins=50, color="black", fill="white"
        )
        + pn.stat_function(
            fun=sp.stats.norm.pdf, args={"loc": 0, "scale": 1}, color="blue", size=1
        )
    )
    return plot


def num_rejects(draws):
    """Return the number of times values are exactly repeated consecutively in the chain of draws."""
    num_draws = np.shape(draws)[0]
    rejects = 0
    for m in range(num_draws - 1):
        if (draws[m, :] == draws[m + 1, :]).all():
            rejects += 1
    return rejects, rejects / num_draws


def constrain(model, draws):
    """Return the constrained draws corresponding the specified draws using the specified model for the transform."""
    num_draws = np.shape(draws)[0]
    D = model.param_unc_num()
    draws_constr = np.empty((num_draws, D))
    for m in range(num_draws):
        draws_constr[m, :] = model.param_constrain(draws[m, :])
    return draws_constr


def metadata_columns(fit):
    """Return the number of metadata columns in CmdStanPy's output."""
    return len(fit.metadata.method_vars.keys())



def dict_draw(thetas_dict, n):
    """Extract the n-th draw from the specified dictionary and return as a dictionary to use for initialization."""
    return {name: draws[n] for name, draws in thetas_dict.items()}


def nuts_adapt(program_path, data_path, seed):
    """Return a NUTS fit for the specified program and data with the specified seed, running long enough to get a relatively accurate answer."""
    model = csp.CmdStanModel(stan_file=program_path)
    fit = model.sample(
        data=data_path,
        seed=seed,
        metric="unit_e",
        show_console=False,
        # adapt_delta=0.95,
        chains=1,
        iter_warmup=20_000,
        iter_sampling=100_000,
        show_progress=False,
    )
    thetas_dict = fit.stan_variables()
    N = metadata_columns(fit)
    theta_draws = fit.draws(concat_chains=True)[:, N:]
    theta_mean = theta_draws.mean(axis=0)
    theta_sd = theta_draws.std(axis=0)
    theta_sq_mean = (theta_draws ** 2).mean(axis=0)
    theta_sq_sd = (theta_draws ** 2).std(axis=0)
    metric = fit.metric
    stepsize = fit.step_size[0]
    nuts_fit = NutsFit(
        thetas_dict,
        theta_draws,
        theta_mean,
        theta_sd,
        theta_sq_mean,
        theta_sq_sd,
        metric,
        stepsize,
    )
    return nuts_fit


def nuts(program_path, data_path, inits, stepsize, draws, seed):
    """Run NUTS for comparison with the specified program, data, inits, stepsize, number of draws, and seed."""
    model = csp.CmdStanModel(stan_file=program_path)
    fit = model.sample(
        data=data_path,
        step_size=stepsize,
        chains=1,
        inits=inits,
        adapt_engaged=False,
        show_console=False,
        show_progress=False,
        metric="unit_e",
        iter_warmup=0,
        iter_sampling=draws,
        seed=seed,
    )
    draws = fit.draws(concat_chains=True)
    cols = np.shape(draws)[1]
    # Magic numbers because CmdStanPy does not expose these
    #    0,            1,         2,          3,           4,          5,       6
    # lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,energy__
    LEAPFROG_COLUMN = 4
    FIRST_PARAM_COLUMN = 7
    leapfrog_steps = np.sum(draws[:, LEAPFROG_COLUMN])
    parameter_draws = draws[:, FIRST_PARAM_COLUMN:]
    return parameter_draws, leapfrog_steps


def root_mean_square_error(theta, theta_sd, theta_hat):
    """Compute the standardized (z-score) RMSE for theta_hat given the reference mean and standard deviation."""
    return np.sqrt(np.sum(((theta_hat - theta) / theta_sd)** 2) / len(theta))


def nuts_experiment(
    program_path,
    data,
    inits,
    seed,
    theta,
    theta_sd,
    theta_sq,
    theta_sq_sd,
    draws,
    stepsize,
):
    """Return specified number of draws for NUTS for the specified program, data, inits, seed, and stepsize, with specified reference values for theta, theta squared and their standard deviation."""
    parameter_draws, leapfrog_steps = nuts(
        program_path, data, inits, stepsize, draws, seed
    )
    theta_nuts = parameter_draws.mean(axis=0)
    theta_sq_nuts = (parameter_draws ** 2).mean(axis=0)
    rmse = root_mean_square_error(theta, theta_sd, theta_nuts)
    rmse_sq = root_mean_square_error(theta_sq, theta_sq_sd, theta_sq_nuts)
    msjd = np.mean(sq_jumps(parameter_draws))
    print(
        f"NUTS: MSJD={msjd:8.3f};  leapfrog_steps={leapfrog_steps};  RMSE(theta)={rmse:7.4f};  RMSE(theta**2)={rmse_sq:8.4f}"
    )
    return msjd, leapfrog_steps, rmse, rmse_sq


def gist_experiment(
    program_path,
    data,
    theta_cons,
    stepsize,
    num_draws,
    frac,
    theta_hat,
    sd_theta_hat,
    theta_sq_hat,
    sd_theta_sq_hat,
    seed,
):
    """Return specified number of draws for GIST for the specified program, data, inits, seed, and stepsize, with specified reference values for theta, theta squared and their standard deviation."""
    model_bs = bs.StanModel(
        model_lib=program_path, data=data, capture_stan_prints=False
    )
    rng = np.random.default_rng(seed)
    theta_unc = model_bs.param_unconstrain(theta_cons)
    sampler = gs.GistSampler(
        model=model_bs, stepsize=stepsize, theta=theta_unc, rng=rng, frac=frac
    )
    constrained_draws = sampler.sample_constrained(num_draws)
    rejects, prop_rejects = num_rejects(constrained_draws)
    prop_no_return = sampler._cannot_get_back_rejects / num_draws
    prop_diverge = sampler._divergences / num_draws
    msjd = np.mean(sq_jumps(constrained_draws))
    theta_hat_gist = constrained_draws.mean(axis=0)
    theta_sq_hat_gist = (constrained_draws ** 2).mean(axis=0)
    rmse = root_mean_square_error(theta_hat, sd_theta_hat, theta_hat_gist)
    rmse_sq = root_mean_square_error(theta_sq_hat, sd_theta_sq_hat, theta_sq_hat_gist)
    steps = sampler._gradient_evals
    print(
        f"Gist({frac}): MSJD={msjd:8.3f};  leapfrog_steps={steps}  reject={prop_rejects:4.2f};  no return={prop_no_return:4.2f};  diverge={prop_diverge:4.2f};  RMSE(theta)={rmse:8.4f};  RMSE(theta**2)={rmse_sq:8.4f}"
    )
    gist_fit = GistFit(
        "ST-HMC",
        frac,
        stepsize,
        steps,
        prop_rejects,
        prop_no_return,
        rmse,
        rmse_sq,
        msjd,
    )
    return gist_fit

    # print(f"Mean(param): {np.mean(constrained_draws, axis=0)}")
    # print(f"Mean(param^2): {np.mean(constrained_draws**2, axis=0)}")
    # scalar_draws_for_traceplot = constrained_draws[: , 0]
    # print(traceplot(scalar_draws_for_traceplot))
    # print(histogram(sq_jumps(draws)))
    # print("(Forward steps to U-turn from initial, Backward steps to U-turn from proposal)")
    # for n in range(10):
    #   print(f"  ({sampler._fwds[n]:3d},  {sampler._bks[n]:3d})")


def model_names():
    """Return the file names of the models to evaluate."""
    return [
        'normal',
        'ill-normal',
        'corr-normal',
        'rosenbrock',
        'glmm-poisson',
        'hmm',
        'garch',
        'lotka-volterra'
        # 'irt-2pl',
        # 'eight-schools',
        # 'normal-mixture',
        # 'arma',
        # 'arK',
        # 'prophet', 
        # 'covid19-impperial-v2',
        # 'pkpd',
    ]


def progressive_experiment(
    program_path, data, theta_unc, stepsize, num_draws, theta_hat, theta_sq_hat, seed
):
    model_bs = bs.StanM
    odel(model_lib=program_path, data=data, capture_stan_prints=False)
    rng = np.random.default_rng(seed)
    theta = model_bs.param_unconstrain(theta_unc)
    sampler = pta.ProgressiveTurnaroundSampler(
        model=model_bs, stepsize=stepsize, theta=theta, rng=rng
    )
    constrained_draws = sampler.sample_constrained(num_draws)
    rejects, prop_rejects = num_rejects(constrained_draws)
    prop_no_return = sampler._cannot_get_back_rejects / num_draws
    prop_diverge = sampler._divergences / num_draws
    msjd = np.mean(sq_jumps(constrained_draws))
    theta_hat_gist = constrained_draws.mean(axis=0)
    theta_sq_hat_gist = (constrained_draws ** 2).mean(axis=0)
    rmse = root_mean_square_error(theta_hat, theta_hat_gist)
    rmse_sq = root_mean_square_error(theta_sq_hat, theta_sq_hat_gist)
    print(
        f"PrAHMC: MSJD={msjd:8.3f};  leapfrog_steps={sampler._gradient_evals}  reject={prop_rejects:4.2f};  no return={prop_no_return:4.2f};  diverge={prop_diverge:4.2f};  RMSE(theta)={rmse:8.4f};  RMSE(theta**2)={rmse_sq:8.4f}"
    )
    # print(f"Mean(param): {np.mean(constrained_draws, axis=0)}")
    v = np.mean(constrained_draws ** 2, axis=0)
    # print(f"Mean(param^2): {v}")
    print(f"Mean(var > 1): {np.mean(v > 1)}")
    # scalar_draws_for_traceplot = constrained_draws[: , 0]
    # print(traceplot(scalar_draws_for_traceplot))
    # print(histogram(sq_jumps(draws)))
    # print("(Forward steps to U-turn from initial, Backward steps to U-turn from proposal)")
    # for n in range(10):
    #   print(f"  ({sampler._fwds[n]:3d},  {sampler._bks[n]:3d})")


def gist_spectral_step_eval(program_name, seed):
    program_path = "../stan/" + program_name + ".stan"
    data_path = "../stan/" + program_name + ".json"
    model = bs.StanModel(model_lib=program_path, data=data_path)
    rng = np.random.default_rng(seed)
    D = model.param_unc_num()
    theta0 = rng.normal(loc=0, scale=1, size=D)  # random init
    sampler = gws.GistSpectralStepSampler(model=model, theta=theta0, rng=rng, lb_frac=0.5)
    num_draws=10_000
    sample = sampler.sample_constrained(num_draws)
    burned_in_sample = sample[range(num_draws // 2, num_draws), :]
    theta_hat = burned_in_sample.mean(axis=0)
    sd_theta_hat = burned_in_sample.std(axis=0)
    quant05_theta_hat = np.percentile(burned_in_sample, 5, axis=0)
    quant95_theta_hat = np.percentile(burned_in_sample, 95, axis=0)
    rejects, prop_rejects = num_rejects(sample)
    print(f"mean = {theta_hat}")
    print(f"sd = {sd_theta_hat}")
    print(f"05% quantile: {quant05_theta_hat}")
    print(f"95% quantile: {quant95_theta_hat}")
    print(f"% rejection: {prop_rejects}")
    
def all_vs_nuts(num_seeds, num_draws, meta_seed):
    stop_griping()
    seed_rng = np.random.default_rng(meta_seed)
    seeds = seed_rng.integers(low=0, high=2 ** 32, size=num_seeds)
    print(f"NUM DRAWS: {num_draws}  NUM SEEDS: {num_seeds}")
    columns = ["model", "sampler", "stepsize", "binom_prob", "val_type", "val"]
    df = pd.DataFrame(columns=columns)
    for program_name in model_names():
        program_path = "../stan/" + program_name + ".stan"
        data_path = "../stan/" + program_name + ".json"
        print(f"\nMODEL: {program_path}")
        print("============================================================")
        nuts_fit = nuts_adapt(
            program_path=program_path, data_path=data_path, seed=seeds[0]
        )
        print(f"# unconstrained parameters = {nuts_fit.num_unc_params_}")
        for stepsize, step_scale in zip(
            [nuts_fit.stepsize_, nuts_fit.stepsize_ / 2], ["step 1", "step 1/2"]
        ):
            print(f"\nSTEP SIZE = {stepsize}")
            for m, seed in enumerate(seeds):
                DRAW_INDEX = 5  # chosen arbitrarily
                DRAW_MULTIPLIER = 10  # also arbitrary
                idx = DRAW_INDEX + DRAW_MULTIPLIER * m
                nuts_draw_dict = dict_draw(nuts_fit.draws_dict_, idx)
                nuts_draw_array = nuts_fit.draws_array_[idx, :]
                msjd, steps, rmse, rmse_sq = nuts_experiment(
                    program_path=program_path,
                    data=data_path,
                    inits=nuts_draw_dict,
                    stepsize=stepsize,
                    theta=nuts_fit.theta_,
                    theta_sd=nuts_fit.theta_sd_,
                    theta_sq=nuts_fit.theta_sq_,
                    theta_sq_sd=nuts_fit.theta_sq_sd_,
                    draws=num_draws,
                    seed=seed,
                )
                df.loc[len(df)] = (
                    program_name,
                    "NUTS",
                    step_scale,
                    "-",
                    "Leapfrog Steps",
                    steps,
                )
                df.loc[len(df)] = (
                    program_name,
                    "NUTS",
                    step_scale,
                    "-",
                    "RMSE (param)",
                    rmse,
                )
                df.loc[len(df)] = (
                    program_name,
                    "NUTS",
                    step_scale,
                    "-",
                    "RMSE (param sq)",
                    rmse_sq,
                )
                df.loc[len(df)] = program_name, "NUTS", step_scale, "-", "MSJD", msjd
                for path_frac, binom_prob_rank in zip(
                    [0.0, 0.3, 0.5, 0.7], [".0", ".3", ".5", ".7"]
                ):
                    gist_fit = gist_experiment(
                        program_path=program_path,
                        data=data_path,
                        theta_cons=np.array(nuts_draw_array),
                        stepsize=stepsize,
                        num_draws=num_draws,
                        frac=path_frac,
                        theta_hat=nuts_fit.theta_,
                        sd_theta_hat=nuts_fit.theta_sd_,
                        theta_sq_hat=nuts_fit.theta_sq_,
                        sd_theta_sq_hat=nuts_fit.theta_sq_sd_,
                        seed=seed,
                    )
                    df.loc[len(df)] = (
                        program_name,
                        "ST",
                        step_scale,
                        binom_prob_rank,
                        "Leapfrog Steps",
                        gist_fit.steps_,
                    )
                    df.loc[len(df)] = (
                        program_name,
                        "ST",
                        step_scale,
                        binom_prob_rank,
                        "RMSE (param)",
                        gist_fit.rmse_,
                    )
                    df.loc[len(df)] = (
                        program_name,
                        "ST",
                        step_scale,
                        binom_prob_rank,
                        "RMSE (param sq)",
                        gist_fit.rmse_sq_,
                    )
                    df.loc[len(df)] = (
                        program_name,
                        "ST",
                        step_scale,
                        binom_prob_rank,
                        "MSJD",
                        gist_fit.msjd_,
                    )
    agg_df = (
        df.groupby(["stepsize", "val_type", "binom_prob", "sampler", "model"])
        .agg(
            mean_val=("val", "mean"),
            std_val=("val", "std"),
        )
        .reset_index()
    )
    df.to_csv("all-vs-nuts.csv", index=False)
    agg_df.to_csv("all-vs-nuts-agg.csv", index=False)


def vs_nuts_plot(val_type):
    df = pd.read_csv("all-vs-nuts.csv")
    rmse_df = df[df["val_type"] == val_type]
    rmse_df["label"] = rmse_df.apply(
        lambda x: "NUTS" if x["sampler"] == "NUTS" else f"STU-{x['binom_prob']}", axis=1
    )
    rmse_df["fill"] = rmse_df["sampler"].apply(
        lambda x: "lightgrey" if x == "NUTS" else "white"
    )
    plot = (
        pn.ggplot(
            rmse_df, pn.aes(x="label", y="val", color="sampler")
        )  # fill='stepsize'
        + pn.geom_boxplot()
        + pn.facet_wrap("~ stepsize + model", scales="free", ncol=len(model_names()))
        + pn.scale_y_continuous(expand=(0, 0, 0.05, 0))
        + pn.expand_limits(y=0)
        + pn.theme(
            axis_text_x=pn.element_text(rotation=90, hjust=1), legend_position="none"
        )
        + pn.labs(x="Sampler", y=val_type)
    )
    plot.save(filename="vs_nuts_" + val_type + ".pdf", width=24, height=6)

# [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
def uniform_interval_plot(num_seeds, num_draws):
    stop_griping()
    meta_seed = 57484894
    seed_rng = np.random.default_rng(meta_seed)
    seeds = seed_rng.integers(low=0, high=2 ** 32, size=num_seeds)
    print(f"NUM DRAWS: {num_draws}  NUM SEEDS: {num_seeds}")
    program_path = "../stan/normal.stan"
    data_path = "../stan/normal.json"
    nuts_fit = nuts_adapt(program_path=program_path, data_path=data_path, seed=seeds[0])
    columns = ["stepsize", "path_frac", "val_type", "val"]
    df = pd.DataFrame(columns=columns)
    for stepsize in [0.36, 0.18]:
        print(f"STEP SIZE: {stepsize}")
        for m, seed in enumerate(seeds):
            print(f"\n{m=}  {seed=}")
            idx = 10 * m
            nuts_draw_dict = dict_draw(nuts_fit.draws_dict_, idx)
            nuts_draw_array = np.array(nuts_fit.draws_array_[idx, :])
            for path_frac in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
                gist_fit = gist_experiment(
                    program_path=program_path,
                    data=data_path,
                    theta_unc=nuts_draw_array,
                    stepsize=stepsize,
                    num_draws=num_draws,
                    frac=path_frac,
                    theta_hat=nuts_fit.theta_,
                    sd_theta_hat=nuts_fit.theta_sd_,
                    theta_sq_hat=nuts_fit.theta_sq_,
                    sd_theta_sq_hat=nuts_fit.theta_sq_sd_,
                    seed=seed,
                )
                df.loc[len(df)] = stepsize, path_frac, "Leapfrog Steps", gist_fit.steps_
                df.loc[len(df)] = stepsize, path_frac, "Reject", gist_fit.prop_reject_
                df.loc[len(df)] = (
                    stepsize,
                    path_frac,
                    "No Return",
                    gist_fit.prop_no_return_,
                )
                df.loc[len(df)] = stepsize, path_frac, "RMSE (param)", gist_fit.rmse_
                df.loc[len(df)] = (
                    stepsize,
                    path_frac,
                    "RMSE (param sq)",
                    gist_fit.rmse_sq_,
                )
                df.loc[len(df)] = stepsize, path_frac, "MSJD", gist_fit.msjd_
    agg_df = (
        df.groupby(["stepsize", "val_type", "path_frac"])
        .agg(
            mean_val=("val", "mean"),
            lower_quantile=("val", lambda x: x.quantile(0.1)),
            upper_quantile=("val", lambda x: x.quantile(0.9)),
        )
        .reset_index()
    )
    plot = (
        pn.ggplot(
            agg_df,
            pn.aes(
                x="path_frac",
                y="mean_val",
                ymin="lower_quantile",
                ymax="upper_quantile",
                group="stepsize",
                color="factor(stepsize)",
            ),
        )
        + pn.geom_line(size=0.5)
        + pn.scale_x_continuous(
            limits=(0, 1),
            breaks=[0, 0.25, 0.5, 0.75, 1],
            labels=["0", "1/4", "1/2", "3/4", "1"],
        )
        + pn.coord_fixed(ratio=1)
        + pn.expand_limits(y=0)
        + pn.labs(y="", x="Lower Bound Fraction", color="Step Size")
        + pn.facet_wrap("~ val_type", scales="free_y", ncol=3)
    )
    plot.save(filename="uniform_prob_steps_plot.pdf", width=8.5, height=5)

def create_model(program_name):
    program_path = "../stan/" + program_name + ".stan"
    data_path = "../stan/" + program_name + ".json"
    model_bs = bs.StanModel(model_lib=program_path, data=data_path)
    return model_bs

def funnel_test():
    # accept 0.6, N = 10_000, time=10; 0.6, 10_000, 20
    seed = 3946456389
    model = create_model('funnel')
    D = model.param_unc_num()
    rng = np.random.default_rng(seed)
    theta0 = np.zeros(D)
    theta0[0] = rng.normal(scale=3)
    theta0[1:D] = rng.normal(scale=np.exp(theta0[0] / 2), size=(D - 1))
    sampler = sas.StepAdaptSampler(
        model=model, rng=rng, integration_time=25, theta=theta0,
        min_accept_prob=0.6
    )
    N = 100_000
    draws = sampler.sample_constrained(N)
    x = draws[:, 0]
    x_df = pd.DataFrame({'log sigma': x})
    plot = (
        pn.ggplot(mapping=pn.aes(x='log sigma'), data=x_df)
        + pn.geom_histogram(pn.aes(y='..density..', x='log sigma'), bins=31, color='black', fill='white')
        + pn.stat_function(fun=lambda x: sp.stats.norm.pdf(x, 0, 3), color='red')
        + pn.scale_x_continuous(limits=(-9, 9), breaks=(-9, -6, -3, 0, 3, 6, 9))
    )
    plot.show()
    
def learning_curve_plot():
    seed = 189236576
    stepsize = 0.5
    model_bs = create_model('normal')
    D = model.param_unc_num()
    rng = np.random.default_rng(seed)
    theta0 = rng.normal(size=D)
    sampler = gs.GistSampler(
        model=model_bs, stepsize=stepsize, theta=theta0, frac=0.0, rng=rng
    )
    sampler = ms.MultinomialSampler(
        model=model_bs, stepsize=stepsize, theta=theta0, steps=10, rng=rng
    )
    sampler = sas.StepAdaptSampler(
        model=model_bs, rng=rng, integration_time=5, theta=theta0,
        min_accept_prob=0.8
    )
    N = 100_000

    # choose one of next two to use sampler or take uniform draws
    draws = sampler.sample_constrained(N)
    # draws = rng.normal(0, 1, size=(N, D))

    cumsum_draws = np.cumsum(draws, axis=0)
    divisors = np.arange(1, draws.shape[0] + 1).reshape(-1, 1)
    sq_err = (cumsum_draws / divisors)**2
    mean_sq_err = np.mean(sq_err, axis=1)
    root_mean_sq_err = np.sqrt(mean_sq_err)
    
    draws_sq = draws ** 2
    cumsum_draws_sq = np.cumsum(draws_sq, axis=0)
    sq_err_sq = (cumsum_draws_sq / divisors - 1)**2  # E[ChiSquare(1)] = 1
    mean_sq_err_sq = np.mean(sq_err_sq, axis=1)
    root_mean_sq_err_sq = np.sqrt(mean_sq_err_sq)

    errs = np.concatenate([root_mean_sq_err, root_mean_sq_err_sq])
    estimands = np.concatenate([np.array(["theta"] * N), np.array(["theta**2"] * N)])
    iteration = np.arange(1, len(mean_sq_err) + 1)
    iterations = np.concatenate([iteration, iteration])
    df = pd.DataFrame(
        {"iteration": iterations, "RMSE": errs, "estimand": estimands}
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
        pn.ggplot(df, pn.aes(x="iteration", y="RMSE"))
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


### MAIN ###

# Generate plots for paper

funnel_test()
# learning_curve_plot()
# uniform_interval_plot(num_seeds = 200, num_draws=100)
# all_vs_nuts(num_seeds = 200, num_draws = 100, meta_seed = 57484894)
# for val_type in ['RMSE (param)', 'RMSE (param sq)', 'MSJD', 'Leapfrog Steps']:
#     vs_nuts_plot(val_type)


# Ongoing experimentation

# gist_spectral_step_eval('funnel', 25384736)
# gist_spectral_step_eval('corr-normal', 25384736)
# gist_spectral_step_eval('normal', 25384736)


