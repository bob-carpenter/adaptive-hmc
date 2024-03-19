import u_turn_sampler as uts
import models
import util
import cmdstanpy as csp
import numpy as np
import pandas as pd
import plotnine as pn
import matplotlib.pyplot as plt
import json

eight_schools = {
    'model': "eight-schools.stan",
    'data': "eight-schools.json",
    'params': 10
}

std_normal = {
    'model': "normal.stan",
    'data': "normal.json",
    'params': 100
}

def sq_jump_dist_df(draws, sampler):
    diffs = draws[1:] - draws[:-1]
    sjds = np.sum(diffs**2, axis=1)
    size = np.size(sjds)
    df = pd.DataFrame({'SJD': sjds, 'sampler': np.full(size, sampler)})
    return df, np.mean(sjds)
    
    
def test_model(config, M, seed = None):
    print(f"TESTING: {config=}  {M=} {seed=}")
    if seed == None:
        np.random.randint(1, 100_000)
    prefix = "../stan/"
    model_path = prefix + config['model']
    data_path = prefix + config['data']
    model = csp.CmdStanModel(stan_file = model_path)
    chains = 4
    metric_diag = {'inv_metric': np.ones(config['params'])}
    print(f"{metric_diag = }")
    fit = model.sample(data = data_path, chains = chains,
                           show_console = True,
                           adapt_engaged = False,
                           metric=metric_diag,
                           step_size = 0.5,
                           iter_warmup=0,
                           iter_sampling= M // 4, seed = seed)
    draws = fit.draws(concat_chains = True)[:, 7:(7 + config['params'])]
    means = np.mean(draws, axis=0)
    means_sq = np.mean(draws**2, axis=0)
    print(fit.summary())
    print(f"mean sq jumps = {util.mean_sq_jump_distance(draws)}")
    print("\n\n")
    print(f"{means = }")
    print(f"{means_sq = }")
    print(f"metric: {np.mean(fit.metric, axis = 0)}")
    model2 = models.StanModel(model_path, data = data_path, seed = seed)
    stepsize = 0.5
    seed = 12345
    with open(data_path, 'r') as f:
        data_dict = json.load(f)
    sampler = uts.UTurnSampler(model2, stepsize = stepsize, seed = seed)
    draws2 = sampler.sample(M)
    D_constr = model2.dims_constrained()
    draws2_constr = np.empty((M, D_constr))

    for m in range(M):
        draws2_constr[m, :] = model2.param_constrain(draws2[m, :])

    print(f"{np.shape(draws2_constr) = }")
    print(f"means: {np.mean(draws2_constr, axis=0) = }")
    print(f"means of squares: {np.mean(draws2_constr**2, axis=0) = }")
    print(f"mean sq jumps = {util.mean_sq_jump_distance(draws2_constr)}")

    df_nuts, msjd_nuts = sq_jump_dist_df(draws, "NUTS")
    df_ahmc, msjd_ahmc = sq_jump_dist_df(draws2_constr, "AHMC")
    df_lines = pd.DataFrame({ 'xintercept': [ msjd_nuts, msjd_ahmc ],
                                  'sampler' : [ "NUTS", "AHMC" ]})
    df = pd.concat([df_nuts, df_ahmc], ignore_index=True)
    plot = (
        pn.ggplot(df, pn.aes(x = 'SJD'))
        + pn.geom_histogram(color='black', fill='white', bins=100, boundary=0)
        + pn.geom_vline(pn.aes(xintercept='xintercept'), data=df_lines,
                            color="blue", size=1)
        + pn.facet_grid('sampler ~ .')
    )
    print(plot)

s = 9834598
M = 500 * 500
# test_model(std_normal, M = M, seed = s) 
test_model(eight_schools, M = M, seed = s)
