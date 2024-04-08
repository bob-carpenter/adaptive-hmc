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

def dict_draw(thetas_dict, n):
    return {name:draws[n] for name, draws in thetas_dict.items()}

def nuts_adapt(program_path, data_path, seed):
    model = csp.CmdStanModel(stan_file = program_path)
    fit = model.sample(data = data_path, seed=seed,
                           metric="unit_e", show_console=False,
                           # adapt_delta=0.95,
                           chains=1, iter_warmup=5_000, iter_sampling=40_000,
                           show_progress=False)
    thetas_dict = fit.stan_variables()
    N = metadata_columns(fit)
    theta_draws = fit.draws(concat_chains=True)[:, N:]
    theta_hat = theta_draws.mean(axis=0)
    theta_sq_hat = (theta_draws**2).mean(axis=0)
    metric = fit.metric
    step_size = fit.step_size
    return thetas_dict, theta_draws, theta_hat, theta_sq_hat, metric, step_size

    
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
        path_frac, theta_hat, theta_sq_hat, seed):
    model_bs = bs.StanModel(model_lib=program_path, data=data,
                         capture_stan_prints=False)
    rng = np.random.default_rng(seed)
    theta = model_bs.param_unconstrain(theta_unc)
    sampler = ta.TurnaroundSampler(model=model_bs, stepsize=stepsize,
                                       theta=theta, rng=rng,
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
    print(f"AHMC({path_frac}): MSJD={msjd:8.3f};  leapfrog_steps={sampler._gradient_evals}  reject={prop_rejects:4.2f};  no return={prop_no_return:4.2f};  diverge={prop_diverge:4.2f};  RMSE(theta)={rmse:8.4f};  RMSE(theta**2)={rmse_sq:8.4f}")
    return "AHMC", path_frac, stepsize, sampler._gradient_evals, prop_rejects, prop_no_return, rmse, rmse_sq, msjd
    # print(f"Mean(param): {np.mean(constrained_draws, axis=0)}")
    # print(f"Mean(param^2): {np.mean(constrained_draws**2, axis=0)}")
    # scalar_draws_for_traceplot = constrained_draws[: , 0]
    # print(traceplot(scalar_draws_for_traceplot))
    # print(histogram(sq_jumps(draws)))
    # print("(Forward steps to U-turn from initial, Backward steps to U-turn from proposal)")
    # for n in range(10):
    #   print(f"  ({sampler._fwds[n]:3d},  {sampler._bks[n]:3d})")


def model_steps():
    normal = ('normal', [0.5, 0.25])# for 500: [0.36, 0.18])
    corr_normal = ('correlated-normal', [0.12, 0.06])
    ill_normal = ('ill-condition-normal', [0.1, 0.05])
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
    return [normal , ill_normal, corr_normal, irt, poisson_glmm, eight_schools, normal_mix, hmm, arma, garch, arK, pkpd, lotka_volterra, prophet] # [covid]

def all_vs_nuts():
    stop_griping()
    meta_seed = 57484894
    seed_rng = np.random.default_rng(meta_seed)
    seeds = seed_rng.integers(low=0, high=2**32, size=2)
    num_draws = 100
    print(f"NUM DRAWS: {num_draws}  SEEDS: {seeds}")
    for program_name, step_sizes in model_steps():
        program_path = '../stan/' + program_name + '.stan'
        data_path = '../stan/' + program_name + '.json'
        print(f"\nMODEL: {program_path}")
        print("============================================================")
        nuts_draws_dict, nuts_draws_array, theta_hat, theta_sq_hat, adapted_metric, adapted_step_size = nuts_adapt(program_path=program_path, data_path=data_path, seed=seeds[0])
        num_unc_params = np.shape(nuts_draws_array[1, :])[0]
        print(f"# unconstrained parameters = {num_unc_params}")
        print(f"NUTS: adapted step size = {adapted_step_size}")
        for step_size in step_sizes:
            print(f"\nSTEP SIZE = {step_size}")
            for m, seed in enumerate(seeds):
                DRAW_INDEX = 5  # chosen arbitrarily
                nuts_draw_dict =  dict_draw(nuts_draws_dict, DRAW_INDEX + 10 * m)
                nuts_draw_array = nuts_draws_array[DRAW_INDEX + 10 * m, :]
                nuts_experiment(program_path=program_path, data=data_path,
                                    inits=nuts_draw_dict, step_size=step_size, theta_hat=theta_hat,
                                    theta_sq_hat=theta_sq_hat, draws=num_draws, seed=seed)
                for uturn_condition in ['distance']:  # 'sym_distance'
                    for path_frac in [0.5, 0.6, 0.7, 0.8]:  # ['full', 'half', 'quarter'] for uniform
                        turnaround_experiment(program_path=program_path,
                                                data=data_path,
                                                theta_unc=np.array(nuts_draw_array),
                                                stepsize=step_size,
                                                num_draws=num_draws,
                                                path_frac=path_frac,
                                                theta_hat=theta_hat,
                                                theta_sq_hat=theta_sq_hat,
                                                seed=seed)



def binomial_prob_plot():
    stop_griping()
    num_seeds = 100
    num_draws = 100
    meta_seed = 57484894
    seed_rng = np.random.default_rng(meta_seed)
    seeds = seed_rng.integers(low=0, high=2**32, size=num_seeds)
    print(f"NUM DRAWS: {num_draws}  SEEDS: {seeds}")
    program_name, step_sizes = 'normal', [0.5, 0.25]  # for 500 dims: [0.36, 0.18])
    program_path = '../stan/' + program_name + '.stan'
    data_path = '../stan/' + program_name + '.json'
    nuts_draws_dict, nuts_draws_array, theta_hat, theta_sq_hat, adapted_metric, adapted_step_size = nuts_adapt(program_path=program_path, data_path=data_path, seed=seeds[0])
    columns = ['stepsize', 'binom_prob', 'val_type', 'val']   # val_type in ['steps', 'reject', 'no_return', 'rmse', 'rmse_sq']
    df = pd.DataFrame(columns=columns)
    for step_size in step_sizes:
        print(f"STEP SIZE: {step_size}")
        for m, seed in enumerate(seeds):
            print(f"{m=}  {seed=}")
            idx = 100 * m
            nuts_draw_dict =  dict_draw(nuts_draws_dict, idx)
            nuts_draw_array = nuts_draws_array[idx, :]
            for binom_prob in [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
                sampler, binom_prob, stepsize, steps, reject, no_return, rmse, rmse_sq, msjd = turnaround_experiment(program_path=program_path,
                                                                                                                         data=data_path,
                                                                                                                         theta_unc=np.array(nuts_draw_array),
                                                                                                                         stepsize=step_size,
                                                                                                                         num_draws=num_draws,
                                                                                                                         path_frac=binom_prob,
                                                                                                                         theta_hat=theta_hat,
                                                                                                                         theta_sq_hat=theta_sq_hat,
                                                                                                                         seed=seed)

                df.loc[len(df)] = stepsize, binom_prob, 'Leapfrog Steps', steps
                df.loc[len(df)] = stepsize, binom_prob, 'Reject', reject
                df.loc[len(df)] = stepsize, binom_prob, 'No Return', no_return
                df.loc[len(df)] = stepsize, binom_prob, 'RMSE (param)', rmse
                df.loc[len(df)] = stepsize, binom_prob, 'RMSE (param sq)', rmse_sq
                df.loc[len(df)] = stepsize, binom_prob, 'MSJD', msjd



                # WORKS FOR SINGLE SEED            
                # plot = (pn.ggplot(df, pn.aes(x='binom_prob', y='val', group='stepsize', color='factor(stepsize)'))
                #            + pn.geom_line()
                #            + pn.scale_x_continuous(limits=(0, 1), breaks = [0, 0.25, 0.5, 0.75, 1], labels=["0", "1/4", "1/2", "3/4", "1"])
                #            + pn.coord_fixed(ratio=1)
                #            + pn.labs(y = '', x='binomial success probability', color="step size")
                #            + pn.facet_wrap('~ val_type', scales='free_y', ncol=5))
                # plot.show()

                agg_df = df.groupby(['stepsize', 'val_type', 'binom_prob']).agg(
                    mean_val=('val', 'mean'),
                    #    std_val=('val', 'std'),
                    lower_quantile=('val', lambda x: x.quantile(0.1)),  # For 80% CI, lower bound
                    upper_quantile=('val', lambda x: x.quantile(0.9))   # For 80% CI, upper bound
                    ).reset_index()

                plot = (pn.ggplot(agg_df, pn.aes(x='binom_prob', y='mean_val', ymin='lower_quantile', ymax='upper_quantile', group='stepsize', color='factor(stepsize)'))
                            + pn.geom_line(size=0.5)
                            + pn.scale_x_continuous(limits=(0, 1), breaks = [0, 0.25, 0.5, 0.75, 1], labels=["0", "1/4", "1/2", "3/4", "1"])
                            + pn.coord_fixed(ratio=1)
                            + pn.labs(y = '', x='Binomial Success Probability', color="Step Size")
                            + pn.facet_wrap('~ val_type', scales='free_y', ncol=3))
                plot.save(filename='binomial_prob_steps_plot.pdf', width=8.5, height=5)
                # plot.show()

all_vs_nuts()

seed = 987599123
stepsize = 0.25
D = 100
program_name = 'normal'
program_path = '../stan/' + program_name + '.stan'
data_path = '../stan/' + program_name + '.json'
model_bs = bs.StanModel(model_lib=program_path, data=data_path,
                        capture_stan_prints=False)
rng = np.random.default_rng(seed)
theta0 = rng.normal(loc=0, scale=1, size=D)
sampler = ta.TurnaroundSampler(model=model_bs, stepsize=stepsize,
                               theta=theta0,
                               path_frac=0.6, rng=rng)
N = 100_000

draws = sampler.sample_constrained(N)
# draws = rng.normal(size=(N, D))  # use for validating plotting code

cumsum_draws = np.cumsum(draws, axis=0)
divisors = np.arange(1, draws.shape[0] + 1).reshape(-1, 1)
abs_err = np.abs(cumsum_draws) / divisors
avg_abs_err = np.mean(abs_err, axis=1)

# for square draws, expected value is 1
draws_sq = draws**2
cumsum_draws_sq = np.cumsum(draws_sq, axis=0)
# catastrophic cancel when result close to 1?
abs_err_sq = np.abs(cumsum_draws_sq / divisors - 1)  # expected value is E[ChiSquare(1)] = 1
avg_abs_err_sq = np.mean(abs_err_sq, axis=1)

errs = np.concatenate([avg_abs_err, avg_abs_err_sq])
estimands = np.concatenate([np.array(['theta'] * N),
                                   np.array(['theta**2'] * N)])

iteration = np.arange(1, len(avg_abs_err) + 1)
iterations = np.concatenate([iteration, iteration])
df = pd.DataFrame({
    'iteration': iterations,
    'E[|err|]': errs,
    'estimand': estimands
})

lines_df = pd.DataFrame({
    'estimand': np.array(['theta', 'theta**2']),
    'x': np.array([10, 10]),
    'y': np.array([1 / np.sqrt(10), np.sqrt(2) / np.sqrt(10)]),
    'xend': np.array([N, N]),
    'yend': np.array([1 / np.sqrt(N), np.sqrt(2) / np.sqrt(N)])
})

plot = (pn.ggplot(df, pn.aes(x='iteration', y='E[|err|]'))
            + pn.geom_line()
            + pn.scale_x_log10(limits=(10,N)) # breaks=[10**0, 10**1, 10**2, 10**3, 10**4], limits=(10**0, 10**4))
            + pn.scale_y_log10() # breaks=[10**-6, 10**-5, 10**-4, 10**-3, 10**-2, 10**-1, 10**0], limits=(10**-6, 10**0))
            + pn.geom_segment(data=lines_df,
                     mapping=pn.aes(x='x', y='y', xend='xend', yend='yend'),
                     linetype='dotted')
            + pn.facet_wrap('~ estimand')
)
plot.save(filename='learning_curve.pdf', width=8, height=4)
# plot.show()
