import u_turn_sampler as uts
import models
import util
import numpy as np
import scipy as sp
import plotnine as pn
import pandas as pd
import logging
import warnings
import json
import cmdstanpy as csp

# FILTER JUNK FROM CmdStanPy and plotnine
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings( "ignore", module = "plotnine\..*" )
csp.utils.get_logger().setLevel(logging.ERROR)

def stan_model_experiment():
    model = models.StanModel('../stan/normal.stan', data = '{ "D": 5 }')
    print(f"{model.dims() = }")
    theta = np.array([0.1, -1.3, 4.8, -13.2, 2])
    print(f"{model.log_density(theta) = }")
    print(f"{model.log_density_gradient(theta) = }")


def stan_model_experiment_b():
    model = models.StanModel('../stan/eight-schools.stan', data = '../stan/eight-schools.json')
    print(f"{model.dims() = }")
    theta = np.array([0.1, -1.3, 4.8, -3.2, 2, 1.5, -1.1, -0.3, 1.2, 0.3 ])
    print(f"{model.log_density(theta) = }")
    print(f"{model.log_density_gradient(theta) = }")


def uturn_eight_schools(seed = 1234):
    model = models.StanModel(file = "../stan/eight-schools.stan", data = "../stan/eight-schools.json")
    D = model.dims()
    M = 100 * 100
    stepsize = 0.9
    N = 2
    print(f"STEP SIZE: {stepsize:4.2f}  {D = }  {N = }")
    msq_jumps = np.empty(N)
    accept_probs = np.empty(N)
    sq_err_X = np.empty((N, D))
    sq_err_Xsq = np.empty((N, D))
    grad_calls = np.empty(N)
    for n in range(N):
        print(f"***** {n = }")
        sampler = uts.UTurnSampler(model, stepsize, seed = seed + n)
        theta0 = sampler._rng.normal(size = D)
        sample = sampler.sample(M)
        msq_jumps[n] = util.mean_sq_jump_distance(sample)
        accept_probs[n] = sampler._accepted / sampler._proposed
        sq_err_X[n, :] = np.mean(sample, axis=0) ** 2
        sq_err_Xsq[n, :] = (np.mean(sample**2, axis=0) - 1) ** 2
        grad_calls[n] = sampler._gradient_calls
    print(f"Y[d] standard error: {np.sqrt(sq_err_X.reshape(N * D).sum() / (N * D))}")
    print(f"Y[d]**2 standard error: {np.sqrt(sq_err_Xsq.reshape(N * D).sum() / (N * D))}")
    print(f"average mean squared jump distance: {np.mean(msq_jumps):5.1f}  ({np.std(msq_jumps):4.2f})")
    print(f"accept probability: {np.mean(accept_probs):4.2f} ({np.std(accept_probs):4.2f})")
    print(f"average gradient calls: {np.mean(grad_calls):8.1f}") 
    
    
def uturn_normal(seed = 1234):
    M = 100 * 100
    stepsize = 0.9
    D = 5
    N = 2
    print(f"STEP SIZE: {stepsize:4.2f}  {D = }  {N = }")
    msq_jumps = np.empty(N)
    accept_probs = np.empty(N)
    sq_err_X = np.empty((N, D))
    sq_err_Xsq = np.empty((N, D))
    grad_calls = np.empty(N)
    model = models.StanModel(file = "../stan/normal.stan", data = '{"D": 5}')
    for n in range(N):
        print(f"***** {n = }")
        sampler = uts.UTurnSampler(model, stepsize, seed = seed + n)
        theta0 = sampler._rng.normal(size=5)
        sample = sampler.sample(M)
        msq_jumps[n] = util.mean_sq_jump_distance(sample)
        accept_probs[n] = sampler._accepted / sampler._proposed
        sq_err_X[n, :] = np.mean(sample, axis=0) ** 2
        sq_err_Xsq[n, :] = (np.mean(sample**2, axis=0) - 1) ** 2
        grad_calls[n] = sampler._gradient_calls
    print(f"Y[d] standard error: {np.sqrt(sq_err_X.reshape(N * D).sum() / (N * D))}")
    print(f"Y[d]**2 standard error: {np.sqrt(sq_err_Xsq.reshape(N * D).sum() / (N * D))}")
    print(f"average mean squared jump distance: {np.mean(msq_jumps):5.1f}  ({np.std(msq_jumps):4.2f})")
    print(f"accept probability: {np.mean(accept_probs):4.2f} ({np.std(accept_probs):4.2f})")
    print(f"average gradient calls: {np.mean(grad_calls):8.1f}") 

def plot_normal(seed = 1234):
    D = 100
    model = models.StdNormal(D)
    stepsize = 0.5
    M = 100_000
    sampler = uts.UTurnSampler(model, stepsize = stepsize, seed = seed)
    sample = sampler.sample(M)
    print(f"E[X]: {np.mean(sample, axis=0)}")
    print(f"E[X^2]: {np.mean(sample**2, axis=0)}")    
    df = pd.DataFrame({"x": sample[1:M, 1]})
    # cf. fully random:
    # df = pd.DataFrame({"x": np.random.randn(M)})
    plot = (
        pn.ggplot(df, pn.aes(x="x"))
        + pn.geom_histogram(
            pn.aes(y="..density.."), bins=50, color="black", fill="white"
        )
        + pn.stat_function(
            fun=sp.stats.norm.pdf, args={"loc": 0, "scale": 1}, color="red", size=1
        )
    )
    print(plot)


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
    
def test_model(config, M, seed):
    print(f"TESTING: {config=}  {M=} {seed=}")
    prefix = "../stan/"
    model_path = prefix + config['model']
    data_path = prefix + config['data']
    model = csp.CmdStanModel(stan_file = model_path)
    fit = model.sample(data = data_path,
                           step_size = 0.5, adapt_engaged = False, chains = 4,
                           iter_sampling=2500, seed = seed)
    draws = fit.draws(concat_chains = True)[:, 7:(7 + config['params'])]
    means = np.mean(draws, axis=0)
    means_sq = np.mean(draws**2, axis=0)
    print(f"{means = }")
    print(f"{means_sq = }")
    print(f"mean sq jumps = {util.mean_sq_jump_distance(draws)}")
    print("\n\n")
    print(fit.summary())
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

def std_normal_errors(seed, D, M, K, stepsize):
    print(f"STD NORMAL: {M=}  {D=} {K=} {stepsize=} {seed=}")

    config = std_normal
    
    # AHMC
    model = models.StdNormal(D)
    sampler = uts.UTurnSampler(model, stepsize = stepsize, seed =  seed)
    errs_X = np.empty((K, D))
    errs_Xsq = np.empty((K, D))

    # NUTS
    prefix = "../stan/"
    model_path = prefix + config['model']
    data_path = prefix + config['data']  # HACK HACK HACK (not reading D!!!)
    model = csp.CmdStanModel(stan_file = model_path)
    errs_X_NUTS = np.empty((K, D))
    errs_Xsq_NUTS = np.empty((K, D))

    for k in range(K):
        # NUTS
        fit = model.sample(data = data_path,
                            step_size = stepsize, adapt_engaged = False, chains = 1,
                            show_progress = False, 
                            iter_sampling=M, seed = (seed * k) // 3)
        draws = fit.draws(concat_chains = True)[:, 7:(7 + config['params'])]
        errs_X_NUTS[k] = np.mean(draws, axis=0)
        errs_Xsq_NUTS[k] = np.mean(draws**2, axis=0)

        # AHMC
        print(f"    {k = } / {range(K)}")
        sample = sampler.sample(M)
        errs_X[k] = np.mean(sample, axis=0)
        errs_Xsq[k] = np.mean(sample**2, axis=0) - 1

    # NUTS
    se_X_hat_NUTS = errs_X_NUTS.std(ddof=1)
    se_Xsq_hat_NUTS = errs_Xsq_NUTS.std(ddof=1)
    ESS_X_hat_NUTS = 1 / se_X_hat_NUTS**2
    ESS_Xsq_hat_NUTS = 2 / se_Xsq_hat_NUTS**2
    print(f"(NUTS)    se[X] = {se_X_hat_NUTS:6.4f}     ESS[X] = {ESS_X_hat_NUTS:6.0f}")
    print(f"(NUTS) se[X**2] = {se_Xsq_hat_NUTS:6.4f}  ESS[X**2] = {ESS_Xsq_hat_NUTS:6.0f}")

    
    # AHMC
    se_X_hat = errs_X.std(ddof=1)
    se_Xsq_hat = errs_Xsq.std(ddof=1)
    ESS_X_hat = 1 / se_X_hat**2
    ESS_Xsq_hat = 2 / se_Xsq_hat**2
    print(f"(AHMC)    se[X] = {se_X_hat:6.4f}     ESS[X] = {ESS_X_hat:6.0f}")
    print(f"(AHMC) se[X**2] = {se_Xsq_hat:6.4f}  ESS[X**2] = {ESS_Xsq_hat:6.0f}")


    

M = 200 * 200
stepsize = 0.5
D = 20
seed = 5245323
K = 50

test_model(std_normal, M = M, seed = seed) 
# test_model(eight_schools, M = M, seed = seed)
    
# std_normal_errors(seed = seed, D = D, M = M, K = K, stepsize = stepsize)

# stan_model_experiment()    
# stan_model_experiment_b()    
# plot_normal(647483)
# uturn_normal(seed = 67375765)    
# uturn_eight_schools(seed = 67375765)    
