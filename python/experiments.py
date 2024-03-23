import turnaround as ta
import cmdstanpy as csp
import numpy as np
import bridgestan as bs
import plotnine as pn
import pandas as pd
import warnings
warnings.filterwarnings("ignore", message="Loading a shared object .* that has already been loaded.*")


def sq_jumps(draws):
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

def nuts(model_path, data_path, stepsize):
    import cmdstanpy as csp
    model = csp.CmdStanModel(stan_file = model_path)
    fit = model.sample(data = data_path, step_size = stepsize,
                           adapt_engaged = False)
    draws = fit.draws(concat_chains = True)
    cols = np.shape(draws)[1]
    parameter_draws = draws[:, 7:cols]
    return parameter_draws
        

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

def turnaround_experiment(program_path, data, stepsize, num_draws,
                              uturn_condition, path_fraction, seed):
    print(f"CONFIG: {program_path=}; {stepsize=};  {num_draws=}; {uturn_condition=}; {path_fraction=}; {seed=}")
    model_bs = bs.StanModel(model_lib=program_path, data=data,
                         capture_stan_prints=False)
    # model_csp = csp.CmdStanModel(stan_file=program_path)
    rng = np.random.default_rng(seed)
    sampler = ta.TurnaroundSampler(model=model_bs, stepsize=stepsize,
                              rng=rng,
                              uturn_condition=uturn_condition,
                              path_fraction=path_fraction)
    constrained_draws = sampler.sample_constrained(num_draws)
    # print(f"Mean(param): {np.mean(constrained_draws, axis=0)}")
    # print(f"Mean(param^2): {np.mean(constrained_draws**2, axis=0)}")
    # scalar_draws_for_traceplot = constrained_draws[: , 0]
    # print(traceplot(scalar_draws_for_traceplot))
    # print(histogram(sq_jumps(draws)))
    print(f"Mean square jump distance: {np.mean(sq_jumps(constrained_draws)):5.1f}")
    rejects, prop_rejects = num_rejects(constrained_draws)
    print(f"Proportion rejects: {prop_rejects:5.3f}")
    print(f"Cannot return rejects: {sampler._cannot_get_back_rejects / num_draws}")
    # print("(Forward steps to U-turn from initial, Backward steps to U-turn from proposal)")
    # for n in range(10):
    #   print(f"  ({sampler._fwds[n]:3d},  {sampler._bks[n]:3d})")


normal = ('../stan/normal.stan', '{"D": 100}')
eight_schools = ('../stan/eight-schools.stan', '../stan/eight-schools.json')
num_draws = 1_000



for program_path, data in [eight_schools, normal]:
       for step in [0.8, 0.4, 0.2, 0.1, 0.05]:
           print("")
           turnaround_experiment(program_path = program_path,
                              data = data,
                              stepsize=step,
                              num_draws = num_draws,
                              uturn_condition = 'distance',
                              path_fraction='full',
                              seed=997459)    
    

