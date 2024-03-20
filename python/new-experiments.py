import turnaround as ta
import numpy as np
import bridgestan as bs

def constrain(model, draws):
    num_draws = np.shape(draws)[0]
    D = model.param_unc_num()
    draws_constr = np.empty((num_draws, D))
    for m in range(num_draws):
        draws_constr[m, :] = model.param_constrain(draws[m, :])
    return draws_constr

def turnaround_experiment(model_path, data, stepsize, num_draws, seed):
    rng = np.random.default_rng(seed)
    model = bs.StanModel(model_lib=model_path, data=data,
                             capture_stan_prints=False)
    sampler = ta.TurnaroundSampler(model=model, stepsize=stepsize, rng=rng)
    draws = sampler.sample(num_draws)
    constrained_draws = constrain(model, draws)
    print(f"MEAN(param): {np.mean(constrained_draws, axis=0)}")
    print(f"MEAN(param^2): {np.mean(constrained_draws**2, axis=0)}")

turnaround_experiment('../stan/normal.stan', data='{"D": 2}',
                   stepsize=0.5, num_draws = 2000,
                   seed=997459)    
