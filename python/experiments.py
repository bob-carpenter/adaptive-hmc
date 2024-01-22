import u_turn_sampler as uts
import models
import util
import numpy as np
import scipy as sp
import plotnine as pn
import pandas as pd

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
    D = 5
    model = models.StdNormal(D)
    stepsize = 0.9
    M = 100_000
    sampler = uts.UTurnSampler(model, stepsize, seed = seed)
    sample = sampler.sample(M)
    df = pd.DataFrame({"x": sample[1:M, 1]})
    plot = (
        pn.ggplot(df, pn.aes(x="x"))
        + pn.geom_histogram(
            pn.aes(y="..density.."), bins=30, color="black", fill="white"
        )
        + pn.stat_function(
            fun=sp.stats.norm.pdf, args={"loc": 0, "scale": 1}, color="red", size=1
        )
    )
    print(plot)

# stan_model_experiment()    
# stan_model_experiment_b()    
# plot_normal()
# uturn_normal(seed = 67375765)    
uturn_eight_schools(seed = 67375765)    
