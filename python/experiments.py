import turnaround as ta
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
                           adapt_delta=0.95,
                           chains=1, iter_warmup=2_000, iter_sampling=40_000,
                           show_progress=False)
    thetas_dict = fit.stan_variables()
    theta_draw_dict = {name:draws[0] for name, draws in thetas_dict.items()}
    N = metadata_columns(fit)
    theta_draws = fit.draws(concat_chains=True)[:, N:]
    theta_draw_array = theta_draws[1, :]
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

def nuts_experiment(program_path, data, inits, seed, theta_hat, draws, step_size):
    parameter_draws, leapfrog_steps = nuts(program_path, data, inits, step_size, draws, seed)
    theta_hat_nuts = parameter_draws.mean(axis=0)
    rmse = root_mean_square_error(theta_hat, theta_hat_nuts)
    print(f"NUTS: MSJD={np.mean(sq_jumps(parameter_draws)):7.2f};  steps={leapfrog_steps=};  RMSE={rmse:6.3f}")
    # print(f"NUTS: Mean(param): {np.mean(parameter_draws, axis=0)}")
    # print(f"NUTS: Mean(param^2): {np.mean(parameter_draws**2, axis=0)}")
    

def turnaround_experiment(program_path, data, init, stepsize, num_draws,
                              uturn_condition, path_fraction, theta_hat, seed):
    model_bs = bs.StanModel(model_lib=program_path, data=data,
                         capture_stan_prints=False)
    rng = np.random.default_rng(seed)
    sampler = ta.TurnaroundSampler(model=model_bs, stepsize=stepsize,
                              rng=rng,
                              uturn_condition=uturn_condition,
                              path_fraction=path_fraction)
    constrained_draws = sampler.sample_constrained(num_draws)
    rejects, prop_rejects = num_rejects(constrained_draws)
    prop_no_return = sampler._cannot_get_back_rejects / num_draws
    prop_diverge = sampler._divergences / num_draws
    msjd = np.mean(sq_jumps(constrained_draws))
    theta_hat_turnaround = constrained_draws.mean(axis=0)
    rmse = root_mean_square_error(theta_hat, theta_hat_turnaround)
    print(f"AHMC({uturn_condition}, {path_fraction}): MSJD={msjd:7.2f};  reject={prop_rejects:4.2f};  no return={prop_no_return:4.2f};  diverge={prop_diverge:4.2f};  rmse={rmse:6.3f}")
    # print(f"Mean(param): {np.mean(constrained_draws, axis=0)}")
    # print(f"Mean(param^2): {np.mean(constrained_draws**2, axis=0)}")
    # scalar_draws_for_traceplot = constrained_draws[: , 0]
    # print(traceplot(scalar_draws_for_traceplot))
    # print(histogram(sq_jumps(draws)))
    # print("(Forward steps to U-turn from initial, Backward steps to U-turn from proposal)")
    # for n in range(10):
    #   print(f"  ({sampler._fwds[n]:3d},  {sampler._bks[n]:3d})")



normal = ('../stan/normal.stan', '../stan/normal.json', [0.5, 0.25])
multi_normal = ('../stan/multi_normal.stan', '../stan/multi_normal.json', [0.2, 0.1])
eight_schools = ('../stan/eight-schools.stan', '../stan/eight-schools.json', [0.5, 0.25])
irt = ('../stan/irt_2pl.stan', '../stan/irt_2pl.json', [0.05, 0.025])
lotka_volterra = ('../stan/lotka_volterra.stan', '../stan/hudson_lynx_hare.json', [0.018, 0.009, 0.004])
arK = ('../stan/arK.stan', '../stan/arK.json', [0.01, 0.005])
garch = ('../stan/garch11.stan', '../stan/garch.json', [0.16, 0.08])
gauss_mix = ('../stan/low_dim_gauss_mix.stan', '../stan/low_dim_gauss_mix.json', [0.01, 0.005])
hmm = ('../stan/hmm_example.stan', '../stan/hmm_example.json', [0.025, 0.125])
pkpd = ('../stan/one_comp_mm_elim_abs.stan', '../stan/one_comp_mm_elim_abs.json', [0.1, 0.05])

covid = ('../stan/covid19imperial_v2.stan', '../stan/ecdc0401.json', [0.01])
arma = ('../stan/arma11.stan', '../stan/arma.json', [0.016, 0.008])
prophet = ('../stan/prophet.stan', '../stan/rstan_downloads.json', [0.1])

model_data_steps = [normal, eight_schools, garch, arK, lotka_volterra, gauss_mix]  # [irt, multi_normal, covid, arma, prophet, pkpd]

stop_griping()
seed1 = 49876354
seed2 = 94281984
seed3 = 73727475
seeds = [seed1, seed2, seed3]
print(f"SEEDS: {seeds}")
num_draws = 200
for program_path, data_path, step_sizes in model_data_steps:
    print(f"\nMODEL: {program_path}")
    print("============================================================")
    nuts_draw_dict, nuts_draw_array, theta_hat, theta_sq_hat, adapted_metric, adapted_step_size = nuts_adapt(program_path=program_path, data_path=data_path, seed=seed1)
    print(f"NUTS: adapted step size = {adapted_step_size}")
    for step_size in step_sizes:
        print(f"\nSTEP SIZE = {step_size}")
        for seed in seeds:
            nuts_experiment(program_path=program_path, data=data_path,
                                inits=nuts_draw_dict, step_size=step_size, theta_hat=theta_hat, draws=num_draws, seed=seed)
        for uturn_condition in ['distance', 'sym_distance']:  # 'angle'
            for path_fraction in ['full', 'half', 'quarter']:
                for seed in seeds:
                    turnaround_experiment(program_path=program_path,
                                            data=data_path,
                                            init=nuts_draw_array,
                                            stepsize=step_size,
                                            num_draws=num_draws,
                                            uturn_condition=uturn_condition,
                                            path_fraction=path_fraction,
                                            theta_hat=theta_hat,
                                            seed=seed)
    

