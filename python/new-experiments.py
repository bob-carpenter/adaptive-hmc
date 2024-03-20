import turnaround_two as ta
import numpy as np
import bridgestan as bs
import plotnine as pn
import pandas as pd

def sq_jumps(draws):
    M = len(draws)
    M = np.shape(draws)[0]
    result = np.empty(M - 1)
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
    draws1 = constrained_draws[: , 1]
    # print(traceplot(draws1))
    # print(histogram(sq_jumps(draws)))
    print(f"too short rejects: {sampler._too_short_rejects} / {len(draws1)} = {sampler._too_short_rejects / len(draws1)}")
    for n in range(50):
        print(f"  ({sampler._fwds[n]:3d},  {sampler._bks[n]:3d})")

turnaround_experiment('../stan/normal.stan', data='{"D": 100}',
                   stepsize=0.5, num_draws = 10000,
                  seed=997459)    

# turnaround_experiment('../stan/eight-schools.stan', data='../stan/eight-schools.json',
#                   stepsize=0.5, num_draws = 10000,
#                   seed=997459)    

