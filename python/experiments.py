import turnaround as ta
import progressive_turnaround as pta
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
                           chains=1, iter_warmup=5_000, iter_sampling=20_000,
                           show_progress=False)
    thetas_dict = fit.stan_variables()
    N = metadata_columns(fit)
    theta_draws = fit.draws(concat_chains=True)[:, N:]
    theta_hat = theta_draws.mean(axis=0)
    theta_sq_hat = (theta_draws**2).mean(axis=0)
    metric = fit.metric
    stepsize = fit.step_size[0]
    return thetas_dict, theta_draws, theta_hat, theta_sq_hat, metric, stepsize


def nuts(program_path, data_path, inits, stepsize, draws, seed):
    model = csp.CmdStanModel(stan_file = program_path)
    fit = model.sample(data = data_path, step_size=stepsize, chains=1,
                           inits = inits, adapt_engaged=False,
                           show_console=False, show_progress=False,
                           metric="unit_e", iter_warmup=0, iter_sampling=draws,
                           seed = seed)
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

def nuts_experiment(program_path, data, inits, seed, theta_hat, theta_sq_hat, draws, stepsize):
    parameter_draws, leapfrog_steps = nuts(program_path, data, inits, stepsize, draws, seed)
    theta_hat_nuts = parameter_draws.mean(axis=0)
    theta_sq_hat_nuts = (parameter_draws**2).mean(axis=0)
    rmse = root_mean_square_error(theta_hat, theta_hat_nuts)
    rmse_sq = root_mean_square_error(theta_sq_hat, theta_sq_hat_nuts)
    msjd = np.mean(sq_jumps(parameter_draws))
    print(f"NUTS: MSJD={msjd:8.3f};  leapfrog_steps={leapfrog_steps};  RMSE(theta)={rmse:7.4f};  RMSE(theta**2)={rmse_sq:8.4f}")
    return msjd, leapfrog_steps, rmse, rmse_sq
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
    steps = sampler._gradient_evals
    print(f"AHMC({path_frac}): MSJD={msjd:8.3f};  leapfrog_steps={steps}  reject={prop_rejects:4.2f};  no return={prop_no_return:4.2f};  diverge={prop_diverge:4.2f};  RMSE(theta)={rmse:8.4f};  RMSE(theta**2)={rmse_sq:8.4f}")
    return "ST-HMC", path_frac, stepsize, steps, prop_rejects, prop_no_return, rmse, rmse_sq, msjd
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
    ill_normal = ('ill-normal', [0.1, 0.05])
    corr_normal = ('corr-normal', [0.12, 0.06])
    irt = ('irt-2pl', [0.05, 0.025])
    poisson_glmm = ('glmm-poisson', [0.008, 0.004])
    eight_schools = ('eight-schools', [0.5, 0.25])
    normal_mix = ('normal-mixture', [0.01, 0.005])
    hmm = ('hmm', [0.025, 0.0125])
    arma = ('arma', [0.016, 0.008])
    garch = ('garch', [0.16, 0.08])
    arK = ('arK', [0.01, 0.005])
    pkpd = ('pkpd', [0.1, 0.05])
    lotka_volterra = ('lotka-volterra', [0.018, 0.009])
    prophet = ('prophet', [0.0006, 0.0003])
    covid = ('covid19-imperial-v2', [0.01])
    return [normal, ill_normal, corr_normal, irt, poisson_glmm, eight_schools, normal_mix, hmm, arma, garch, arK, pkpd, lotka_volterra, prophet] # covid

def progressive_experiment(program_path, data, theta_unc, stepsize, num_draws,
                           theta_hat, theta_sq_hat, seed):
    model_bs = bs.StanM
    odel(model_lib=program_path, data=data,
                         capture_stan_prints=False)
    rng = np.random.default_rng(seed)
    theta = model_bs.param_unconstrain(theta_unc)
    sampler = pta.ProgressiveTurnaroundSampler(model=model_bs, stepsize=stepsize,
                                               theta=theta, rng=rng)
    constrained_draws = sampler.sample_constrained(num_draws)
    rejects, prop_rejects = num_rejects(constrained_draws)
    prop_no_return = sampler._cannot_get_back_rejects / num_draws
    prop_diverge = sampler._divergences / num_draws
    msjd = np.mean(sq_jumps(constrained_draws))
    theta_hat_turnaround = constrained_draws.mean(axis=0)
    theta_sq_hat_turnaround = (constrained_draws**2).mean(axis=0)
    rmse = root_mean_square_error(theta_hat, theta_hat_turnaround)
    rmse_sq = root_mean_square_error(theta_sq_hat, theta_sq_hat_turnaround)
    print(f"PrAHMC: MSJD={msjd:8.3f};  leapfrog_steps={sampler._gradient_evals}  reject={prop_rejects:4.2f};  no return={prop_no_return:4.2f};  diverge={prop_diverge:4.2f};  RMSE(theta)={rmse:8.4f};  RMSE(theta**2)={rmse_sq:8.4f}")
    # print(f"Mean(param): {np.mean(constrained_draws, axis=0)}")
    v = np.mean(constrained_draws**2, axis=0)
    # print(f"Mean(param^2): {v}")
    print(f"Mean(var > 1): {np.mean(v > 1)}")
    # scalar_draws_for_traceplot = constrained_draws[: , 0]
    # print(traceplot(scalar_draws_for_traceplot))
    # print(histogram(sq_jumps(draws)))
    # print("(Forward steps to U-turn from initial, Backward steps to U-turn from proposal)")
    # for n in range(10):
    #   print(f"  ({sampler._fwds[n]:3d},  {sampler._bks[n]:3d})")

def all_vs_nuts():
    stop_griping()
    num_seeds = 20
    num_draws = 100
    meta_seed = 57484894
    seed_rng = np.random.default_rng(meta_seed)
    seeds = seed_rng.integers(low=0, high=2**32, size=num_seeds)
    print(f"NUM DRAWS: {num_draws}  SEEDS: {seeds}")
    columns = ['model', 'sampler', 'stepsize', 'binom_prob', 'val_type', 'val'] 
    df = pd.DataFrame(columns=columns)
    for program_name, stepsizes in model_steps():
        program_path = '../stan/' + program_name + '.stan'
        data_path = '../stan/' + program_name + '.json'
        print(f"\nMODEL: {program_path}")
        print("============================================================")
        nuts_draws_dict, nuts_draws_array, theta_hat, theta_sq_hat, adapted_metric, adapted_stepsize = nuts_adapt(program_path=program_path, data_path=data_path, seed=seeds[0])
        num_unc_params = np.shape(nuts_draws_array[1, :])[0]
        print(f"# unconstrained parameters = {num_unc_params}")
        print(f"NUTS: adapted step size = {adapted_stepsize}")
        for stepsize in [adapted_stepsize, adapted_stepsize / 2]:
            step_scale = "step 1" if stepsize==adapted_stepsize else "step 1/2"
            print(f"\nSTEP SIZE = {stepsize}")
            for m, seed in enumerate(seeds):
                DRAW_INDEX = 5  # chosen arbitrarily
                nuts_draw_dict =  dict_draw(nuts_draws_dict, DRAW_INDEX + 10 * m)
                nuts_draw_array = nuts_draws_array[DRAW_INDEX + 10 * m, :]
                msjd, steps, rmse, rmse_sq =  nuts_experiment(program_path=program_path, data=data_path,
                                                                  inits=nuts_draw_dict, stepsize=stepsize, theta_hat=theta_hat,
                                                                  theta_sq_hat=theta_sq_hat, draws=num_draws, seed=seed)
                df.loc[len(df)] = program_name, "NUTS", step_scale, "-", 'Leapfrog Steps', steps
                df.loc[len(df)] = program_name, "NUTS", step_scale, "-", 'RMSE (param)', rmse
                df.loc[len(df)] = program_name, "NUTS", step_scale, "-", 'RMSE (param sq)', rmse_sq
                df.loc[len(df)] = program_name, "NUTS", step_scale, "-", 'MSJD', msjd
                # progressive_experiment(program_path=program_path,
                #                    data=data_path,
                #                    theta_unc=np.array(nuts_draw_array),
                #                    stepsize=stepsize,
                #                    num_draws=num_draws,
                #                    theta_hat=theta_hat,
                #                    theta_sq_hat=theta_sq_hat,
                #                    seed=seed)   
                for uturn_condition in ['distance']:  # 'sym_distance'
                    for path_frac in [0.5, 0.625, 0.75]:  # ['full', 'half', 'quarter'] for uniform
                        if path_frac == 0.5:
                            binom_prob_rank = 'S'
                        elif path_frac == 0.625:
                            binom_prob_rank = 'M'
                        elif path_frac == 0.75:
                            binom_prob_rank = 'L'
                        else:
                            path_rank = "?"
                        sampler_name, binom_prob, stepsize, steps, _, _, rmse, rmse_sq, msjd = turnaround_experiment(program_path=program_path,
                                                                                                                             data=data_path,
                                                                                                                             theta_unc=np.array(nuts_draw_array),
                                                                                                                             stepsize=stepsize,
                                                                                                                             num_draws=num_draws,
                                                                                                                             path_frac=path_frac,
                                                                                                                             theta_hat=theta_hat,
                                                                                                                             theta_sq_hat=theta_sq_hat,
                                                                                                                             seed=seed)
                        df.loc[len(df)] = program_name, 'ST', step_scale, binom_prob_rank, 'Leapfrog Steps', steps
                        df.loc[len(df)] = program_name, 'ST', step_scale, binom_prob_rank, 'RMSE (param)', rmse
                        df.loc[len(df)] = program_name, 'ST', step_scale, binom_prob_rank, 'RMSE (param sq)', rmse_sq
                        df.loc[len(df)] = program_name, 'ST', step_scale, binom_prob_rank, 'MSJD', msjd
    agg_df = df.groupby(['stepsize', 'val_type', 'binom_prob', 'sampler', 'model']).agg(
        mean_val=('val', 'mean'),
        std_val=('val', 'std'),
        ).reset_index()
    pd.set_option('display.max_rows', None)
    print(agg_df)
    df.to_csv('all-vs-nuts.csv', index=False)
    agg_df.to_csv('all-vs-nuts-agg.csv', index=False)
    return agg_df

def plot_vs_nuts(val_type):
    df = pd.read_csv('all-vs-nuts.csv')
    rmse_df = df[df['val_type'] == val_type]
    rmse_df['label'] = rmse_df.apply(lambda x: 'NUTS' if x['sampler'] == 'NUTS' else f"STU-{x['binom_prob']}", axis=1)
    rmse_df['fill'] = rmse_df['sampler'].apply(lambda x: 'lightgrey' if x == 'NUTS' else 'white')
    plot = (
        pn.ggplot(rmse_df, pn.aes(x='label', y='val', color='sampler')) # fill='stepsize'
        + pn.geom_boxplot()
        + pn.facet_wrap('~ stepsize + model', scales='free', ncol=len(model_steps()))
        + pn.scale_y_continuous(expand=(0, 0, 0.05, 0))
        + pn.expand_limits(y = 0)
        + pn.theme(axis_text_x=pn.element_text(rotation=90, hjust=1),
                       legend_position='none')
        + pn.labs(x='Sampler', y='RMSE (param sq)', title=(val_type + ' by model and stepsize fraction'))
    )
    plot.save(filename='vs_nuts_' + val_type + '.pdf', width=24, height=6)
    # print(plot)

def binomial_prob_plot():
    stop_griping()
    num_seeds = 400
    num_draws = 100
    meta_seed = 57484894
    seed_rng = np.random.default_rng(meta_seed)
    seeds = seed_rng.integers(low=0, high=2**32, size=num_seeds)
    print(f"NUM DRAWS: {num_draws}  SEEDS: {seeds}")
    program_name, stepsizes = 'normal', [0.5, 0.25]  # for 500 dims: [0.36, 0.18])
    program_path = '../stan/' + program_name + '.stan'
    data_path = '../stan/' + program_name + '.json'
    nuts_draws_dict, nuts_draws_array, theta_hat, theta_sq_hat, adapted_metric, adapted_stepsize = nuts_adapt(program_path=program_path, data_path=data_path, seed=seeds[0])
    columns = ['stepsize', 'binom_prob', 'val_type', 'val']   # val_type in ['steps', 'reject', 'no_return', 'rmse', 'rmse_sq']
    df = pd.DataFrame(columns=columns)
    for stepsize in stepsizes:
        print(f"STEP SIZE: {stepsize}")
        for m, seed in enumerate(seeds):
            print(f"{m=}  {seed=}")
            idx = 100 * m
            nuts_draw_dict =  dict_draw(nuts_draws_dict, idx)
            nuts_draw_array = nuts_draws_array[idx, :]
            for binom_prob in [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9]:
                sampler, binom_prob, stepsize, steps, reject, no_return, rmse, rmse_sq, msjd = turnaround_experiment(program_path=program_path,
                                                                                                                         data=data_path,
                                                                                                                         theta_unc=np.array(nuts_draw_array),
                                                                                                                         stepsize=stepsize,
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

    
def learning_curve_plot():                
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
    N = 1_000_000
    draws = sampler.sample_constrained(N)
    cumsum_draws = np.cumsum(draws, axis=0)
    divisors = np.arange(1, draws.shape[0] + 1).reshape(-1, 1)
    abs_err = np.abs(cumsum_draws) / divisors
    avg_abs_err = np.mean(abs_err, axis=1)

    draws_sq = draws**2
    cumsum_draws_sq = np.cumsum(draws_sq, axis=0)
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
    plot.save(filename='learning_curve.pdf', width=6, height=3)


### MAIN ###
# all_vs_nuts()
for val_type in ['RMSE (param)', 'RMSE (param sq)', 'MSJD', 'Leapfrog Steps']:
    plot_vs_nuts(val_type)
# binomial_prob_plot()
# learning_curve_plot()
