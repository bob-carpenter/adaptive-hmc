import numpy as np
import os, sys, time
import matplotlib.pyplot as plt

from dr_hmc import DRHMC_AdaptiveStepsize, DRHMC_AdaptiveStepsize_autotune, HMC_uturn
import util

import logging
import cmdstanpy as csp
csp.utils.get_logger().setLevel(logging.ERROR)

import chirag_models

# Setup MPI environment
from mpi4py import MPI
comm = MPI.COMM_WORLD
wrank = comm.Get_rank()
wsize = comm.Get_size()
print('My rank is ',wrank)
savepath = '/mnt/ceph/users/cmodi/adaptive_hmc/'

#######
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp', type=str, help='which experiment')
parser.add_argument('-n', type=int, help='dimensionality or model number')
#arguments for GSM
parser.add_argument('--nleap', type=int, default=40, help='number of leapfrog steps')
parser.add_argument('--nsamples', type=int, default=1001, help='number of samples')
parser.add_argument('--burnin', type=int, default=10, help='number of iterations for burn-in')
parser.add_argument('--stepadapt', type=int, default=100, help='step size adaptation')
parser.add_argument('--targetaccept', type=float, default=0.80, help='target acceptance')
parser.add_argument('--stepsize', type=float, default=0.1, help='initial step size')
parser.add_argument('--debug', type=int, default=0, help='constant trajectory for delayed')
#arguments for path name
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')


args = parser.parse_args()
experiment = args.exp
n = args.n

if experiment == 'pdb':
    model, D, lp, lp_g, ref_samples, files = chirag_models.setup_pdb(n)

if experiment == 'normal':
    D = n
    savepath = f'{savepath}/normal-{D}/'
    model, D, lp, lp_g, ref_samples, files = chirag_models.normal(D)

if experiment == 'funnel':
    D = n
    savepath = f'{savepath}/funnel-{D}/'
    model, D, lp, lp_g, ref_samples, files = chirag_models.funnel(D)

if experiment == 'rosenbrock':
    D = n
    savepath = f'{savepath}/rosenbrock-{D}/'
    model, D, lp, lp_g, ref_samples, files = chirag_models.rosenbrock(D)

if experiment == 'stochastic_volatility':
    savepath = f'{savepath}/stochastic_volatility/'
    model, D, lp, lp_g, ref_samples, files = chirag_models.stochastic_volatility()
    
if experiment == 'eight_schools_centered':
    savepath = f'{savepath}/eight_schools_centered/'
    model, D, lp, lp_g, ref_samples, files = chirag_models.eight_schools_centered()
    
if ~args.debug:
    os.makedirs(f'{savepath}', exist_ok=True)
    np.save(f'{savepath}/samples', ref_samples)

###################################
##### Setup the algorithm parameters

nleap = args.nleap
step_size = args.stepsize
nsamples = args.nsamples
burnin = args.burnin
n_stepsize_adapt = args.stepadapt
nchains = wsize
target_accept = args.targetaccept
print(f"Saving runs in folder : {savepath}")

debug = args.debug
if debug:
    nsamples = 1000
    burnin = 100
    n_stepsize_adapt = 100


###################################
# NUTS
if wrank == 0:
    print("\nNow run NUTS on rank 0")
    stanfile, datafile = files
    model = csp.CmdStanModel(stan_file = stanfile)
    sample = model.sample(data=datafile, chains=wsize, iter_sampling=nsamples-1, 
                          metric="unit",
                          adapt_delta=target_accept,
                          adapt_metric_window=0,
                          adapt_init_phase=n_stepsize_adapt//2,
                          adapt_step_size=n_stepsize_adapt//2,
                          show_console=False, show_progress=False, save_warmup=False)
    draws_pd = sample.draws_pd()
    if not debug:
        samples_nuts, leapfrogs_nuts = util.cmdstanpy_wrapper(draws_pd, savepath=f'{savepath}/nuts/')
        np.save(f'{savepath}/nuts/stepsize', sample.step_size)
    else:
        samples_nuts, leapfrogs_nuts = util.cmdstanpy_wrapper(draws_pd, savepath=None)

    step_size = sample.step_size
else:
    step_size = 0.
    samples_nuts = np.zeros([nchains, D])
comm.Barrier()

# Scatter step size and initial point to different ranks
comm.Barrier()
step_size = comm.scatter(step_size, root=0)
q0 = comm.scatter(samples_nuts[:, -1], root=0)
print(f"Step size in rank {wrank}: ", step_size)
comm.Barrier()


# UTurn
np.random.seed(0)
#q0 = np.random.normal(np.zeros(D*nchains).reshape(nchains, D))[wrank]
kernel = HMC_uturn(D, lp, lp_g, mass_matrix=np.eye(D), min_nleap=10)
sampler = kernel.sample(q0, nleap=None, step_size=step_size, nsamples=nsamples, burnin=burnin,
                        epsadapt=0., target_accept=target_accept, 
                        verbose=False)

print(f"Acceptance for Uturn HMC in chain {wrank} : ", np.unique(sampler.accepts, return_counts=True))
if not debug: sampler.save(path=f"{savepath}/uturn/", suffix=f"-{wrank}")
         
comm.Barrier()
samples_uturn = comm.gather(sampler.samples, root=0)
accepts_uturn = comm.gather(sampler.accepts, root=0)
counts_uturn = comm.gather(sampler.counts, root=0)
stepsizes = comm.gather(kernel.step_size, root=0)
if wrank == 0 :
    print(stepsizes)
    np.save(f'{savepath}/uturn/stepsize', stepsizes)
comm.Barrier()


if wrank == 0:
    # plot
    plt.figure()
    #plt.hist(np.random.normal(0, 3, 100000), density=True, alpha=1, bins='auto', lw=2, histtype='step', color='k', label='Reference')
    plt.hist(ref_samples[..., 0].flatten(), density=True, alpha=1, bins='auto', lw=2, histtype='step', color='k', label='Reference')
    plt.hist(samples_nuts[..., 0].flatten(), density=True, alpha=0.5, bins='auto', label='NUTS');
    samples_uturn = np.stack(samples_uturn, axis=0)
    plt.hist(samples_uturn[..., 0].flatten(), density=True, alpha=0.5, bins='auto', label='1step ADR-HMC');
    
    print("\nPlotting")
    plt.title(f"{D-1} dimension funnel")
    plt.legend()
    plt.savefig('tmp.png')
    plt.savefig(f"{savepath}/uturn/hist")
    plt.close()

    print()
    #print("Total accpetances for HMC : ", np.unique(np.stack(accepts_hmc), return_counts=True))
    print("Total accpetances for adaptive HMC : ", np.unique(np.stack(accepts_uturn), return_counts=True))

comm.Barrier()

sys.exit()
