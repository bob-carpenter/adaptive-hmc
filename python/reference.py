import u_turn_sampler as uts
import models
import util
import cmdstanpy as csp
import numpy as np
import json

eight_schools = {
    'model': "eight-schools.stan",
    'data': "eight-schools.json",
    'params': 10
}


def test_model(config):
    prefix = "../stan/"
    model_path = prefix + config['model']
    data_path = prefix + config['data']
    model = csp.CmdStanModel(stan_file = model_path)
    fit = model.sample(data = data_path,
                           iter_warmup = 10_000, iter_sampling=10_000, thin = 10)
    draws = fit.draws(concat_chains = True)[:, 7:(7 + config['params'])]
    means = np.mean(draws, axis=0)
    means_sq = np.mean(draws**2, axis=0)
    print(f"{means = }")
    print(f"{means_sq = }")
    print(f"mean sq jumps = {util.mean_sq_jump_distance(draws)}")
    print("\n\n")
    print(fit.summary())
    print(f"metric: {np.mean(fit.metric, axis = 0)}")
    model2 = models.StanModel(model_path, data = data_path)
    M = 100 * 100
    stepsize = 0.1
    seed = 12345
    with open(data_path, 'r') as f:
        data_dict = json.load(f)
    sampler = uts.UTurnSampler(model2, stepsize, seed = seed)
    draws2 = sampler.sample(M)
    print(f"means: {np.mean(draws2, axis=0) = }")
    print(f"means of squares: {np.mean(draws2**2, axis=0) = }")
    print(f"mean sq jumps = {util.mean_sq_jump_distance(draws2)}")

test_model(eight_schools)
