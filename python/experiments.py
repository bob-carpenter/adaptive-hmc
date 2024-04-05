import turnaround_binomial as ta
import cmdstanpy as csp
import numpy as np
import bridgestan as bs
import plotnine as pn
import pandas as pd
import logging
import traceback
import warnings

def stop_griping():
    warnings.filterwarnings("ignore", message="Loading a shared object .* that has already been loaded.*")
    csp.utils.get_logger().setLevel(logging.ERROR)

def flatten_dict_values(data_dict):
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
    M = np.shape(draws)[0]
    jumps = draws[range(1, M), :] - draws[range(0, M-1), :]
    return [np.dot(jump, jump) for jump in jumps]

def traceplot(chain):
    df = pd.DataFrame({'m': range(len(chain)), 'theta': chain})
    plot = (pn.ggplot(df, pn.aes(x='m', y='theta')) +
                pn.geom_line() +
                pn.labs(x='Iteration', y='Parameter Value', title='Trace Plot of MCMC Chain') +
                pn.theme_minimal())
    return plot

def histogram(xs):
    df = pd.DataFrame({'x': xs})
    plot = (pn.ggplot(df, pn.aes(x = 'x'))
                + pn.geom_histogram(bins=100)
                + pn.geom_vline(xintercept=np.mean(xs), color="blue", size=1))
    return plot

def histogram_vs_normal_density(xs):
    df = pd.DataFrame({"x": xs})
    plot = (
        pn.ggplot(df, pn.aes(x="x"))
        + pn.geom_histogram(
            pn.aes(y="..density.."), bins=50, color="black", fill="white"
        )
        + pn.stat_function(
            fun=sp.stats.norm.pdf, args={"loc": 0, "scale": 1}, color="red", size=1
        )
    )
    return plot

def num_rejects(draws):
    num_draws = np.shape(draws)[0]
    rejects = 0
    for m in range(num_draws - 1):
        if (draws[m, :] == draws[m + 1, :]).all():
            rejects += 1
    return rejects, rejects / num_draws
        
def constrain(model, draws):
    num_draws = np.shape(draws)[0]
    D = model.param_unc_num()
    draws_constr = np.empty((num_draws, D))
    for m in range(num_draws):
        draws_constr[m, :] = model.param_constrain(draws[m, :])
    return draws_constr

def metadata_columns(fit):
    return len(fit.metadata.method_vars.keys())

def nuts_adapt(program_path, data_path, seed):
    model = csp.CmdStanModel(stan_file = program_path)
    fit = model.sample(data = data_path, seed=seed,
                           metric="unit_e", show_console=False,
                           # adapt_delta=0.95,
                           chains=1, iter_warmup=2_000, iter_sampling=10_000,
                           show_progress=False)
    thetas_dict = fit.stan_variables()
    theta_draw_dict = {name:draws[0] for name, draws in thetas_dict.items()}
    N = metadata_columns(fit)
    theta_draws = fit.draws(concat_chains=True)[:, N:]
    theta_draw_array = theta_draws[5, :]  # number 5 is arbitrary
    theta_hat = theta_draws.mean(axis=0)
    theta_sq_hat = (theta_draws**2).mean(axis=0)
    metric = fit.metric
    step_size = fit.step_size
    return theta_draw_dict, theta_draw_array, theta_hat, theta_sq_hat, metric, step_size

    
def nuts(program_path, data_path, inits, step_size, draws, seed):
    model = csp.CmdStanModel(stan_file = program_path)
    fit = model.sample(data = data_path, step_size=step_size, chains=1,
                           inits = inits, adapt_engaged=False,
                           metric="unit_e", iter_warmup=0, iter_sampling=draws,
                           seed = seed, show_progress=False)
    draws = fit.draws(concat_chains = True)
    cols = np.shape(draws)[1]
    # Magic numbers because CmdStanPy does not expose these
    #    0,            1,         2,          3,           4,          5,       6
    # lp__,accept_stat__,stepsize__,treedepth__,n_leapfrog__,divergent__,energy__
    LEAPFROG_COLUMN = 4
    FIRST_PARAM_COLUMN = 7
    leapfrog_steps = np.sum(draws[:, LEAPFROG_COLUMN])
    parameter_draws = draws[:, FIRST_PARAM_COLUMN:]
    return parameter_draws, leapfrog_steps

def root_mean_square_error(theta1, theta2):
    return np.sqrt(np.sum((theta1 - theta2)**2) / len(theta1))

def nuts_experiment(program_path, data, inits, seed, theta_hat, theta_sq_hat, draws, step_size):
    parameter_draws, leapfrog_steps = nuts(program_path, data, inits, step_size, draws, seed)
    theta_hat_nuts = parameter_draws.mean(axis=0)
    theta_sq_hat_nuts = (parameter_draws**2).mean(axis=0)
    rmse = root_mean_square_error(theta_hat, theta_hat_nuts)
    rmse_sq = root_mean_square_error(theta_sq_hat, theta_sq_hat_nuts)
    print(f"NUTS: MSJD={np.mean(sq_jumps(parameter_draws)):8.3f};  leapfrog_steps={leapfrog_steps};  RMSE(theta)={rmse:7.4f};  RMSE(theta**2)={rmse_sq:8.4f}")
    # print(f"NUTS: Mean(param): {np.mean(parameter_draws, axis=0)}")
    # print(f"NUTS: Mean(param^2): {np.mean(parameter_draws**2, axis=0)}")
    

def turnaround_experiment(program_path, data, theta_unc, stepsize, num_draws,
                              uturn_condition, path_frac, theta_hat, theta_sq_hat,
                              seed):
    model_bs = bs.StanModel(model_lib=program_path, data=data,
                         capture_stan_prints=False)
    rng = np.random.default_rng(seed)
    theta = model_bs.param_unconstrain(theta_unc)
    sampler = ta.TurnaroundSampler(model=model_bs, stepsize=stepsize,
                                       theta=theta, rng=rng,
                                       uturn_condition=uturn_condition,
                                       path_frac=path_frac)
    constrained_draws = sampler.sample_constrained(num_draws)
    rejects, prop_rejects = num_rejects(constrained_draws)
    prop_no_return = sampler._cannot_get_back_rejects / num_draws
    prop_diverge = sampler._divergences / num_draws
    msjd = np.mean(sq_jumps(constrained_draws))
    theta_hat_turnaround = constrained_draws.mean(axis=0)
    theta_sq_hat_turnaround = (constrained_draws**2).mean(axis=0)
    rmse = root_mean_square_error(theta_hat, theta_hat_turnaround)
    rmse_sq = root_mean_square_error(theta_sq_hat, theta_sq_hat_turnaround)
    print(f"AHMC({uturn_condition}, {path_frac}): MSJD={msjd:8.3f};  leapfrog_steps={sampler._gradient_evals}  reject={prop_rejects:4.2f};  no return={prop_no_return:4.2f};  diverge={prop_diverge:4.2f};  RMSE(theta)={rmse:8.4f};  RMSE(theta**2)={rmse_sq:8.4f}")
    # print(f"Mean(param): {np.mean(constrained_draws, axis=0)}")
    # print(f"Mean(param^2): {np.mean(constrained_draws**2, axis=0)}")
    # scalar_draws_for_traceplot = constrained_draws[: , 0]
    # print(traceplot(scalar_draws_for_traceplot))
    # print(histogram(sq_jumps(draws)))
    # print("(Forward steps to U-turn from initial, Backward steps to U-turn from proposal)")
    # for n in range(10):
    #   print(f"  ({sampler._fwds[n]:3d},  {sampler._bks[n]:3d})")



normal = ('normal', [0.36, 0.18])
corr_normal = ('correlated-normal', [0.12, 0.06])
eight_schools = ('eight-schools', [0.5, 0.25])
irt = ('irt-2pl', [0.05, 0.025])
lotka_volterra = ('lotka-volterra', [0.018, 0.009])
arK = ('arK', [0.01, 0.005])
garch = ('garch', [0.16, 0.08])
normal_mix = ('normal-mixture', [0.01, 0.005])
hmm = ('hmm', [0.025, 0.0125])
pkpd = ('pkpd', [0.1, 0.05])
arma = ('arma', [0.016, 0.008])
poisson_glmm = ('glmm-poisson', [0.008, 0.004])
covid = ('covid19-imperial-v2', [0.01])
prophet = ('prophet', [0.0006, 0.0003])

model_steps = [normal, corr_normal, irt, poisson_glmm, eight_schools,
                   normal_mix, hmm, arma, garch, arK, pkpd, lotka_volterra, prophet] # [covid]

stop_griping()
meta_seed = 57484894
seed_rng = np.random.default_rng(meta_seed)
seeds = seed_rng.integers(low=0, high=2**32, size=2)
print(f"SEEDS: {seeds}")
num_draws = 400
for program_name, step_sizes in model_steps:
    program_path = '../stan/' + program_name + '.stan'
    data_path = '../stan/' + program_name + '.json'
    print(f"\nMODEL: {program_path}")
    print("============================================================")
    nuts_draw_dict, nuts_draw_array, theta_hat, theta_sq_hat, adapted_metric, adapted_step_size = nuts_adapt(program_path=program_path, data_path=data_path, seed=seeds[0])
    print(f"# unconstrained parameters = {np.shape(nuts_draw_array)[0]}")
    print(f"NUTS: adapted step size = {adapted_step_size}")
    for step_size in step_sizes:
        print(f"\nSTEP SIZE = {step_size}")
        for seed in seeds:
            nuts_experiment(program_path=program_path, data=data_path,
                                inits=nuts_draw_dict, step_size=step_size, theta_hat=theta_hat,
                                theta_sq_hat=theta_sq_hat, draws=num_draws, seed=seed)
            for uturn_condition in ['distance']:  # 'sym_distance'
                for path_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:  # ['full', 'half', 'quarter'] for uniform
                    turnaround_experiment(program_path=program_path,
                                            data=data_path,
                                            theta_unc=np.array(nuts_draw_array),
                                            stepsize=step_size,
                                            num_draws=num_draws,
                                            uturn_condition=uturn_condition,
                                            path_frac=path_frac,
                                            theta_hat=theta_hat,
                                            theta_sq_hat=theta_sq_hat,
                                            seed=seed)
    

