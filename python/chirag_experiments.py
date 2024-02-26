import numpy as np
import sys, time
import matplotlib.pyplot as plt

from dr_hmc import DRHMC_AdaptiveStepsize
from mpi4py import MPI
import util

import logging
import cmdstanpy as csp
csp.utils.get_logger().setLevel(logging.ERROR)

import chirag_models

# Setup MPI environment
comm = MPI.COMM_WORLD
wrank = comm.Get_rank()
wsize = comm.Get_size()
print('My rank is ',wrank)
savepath = '/mnt/ceph/users/cmodi/adaptive_hmc/'

#######
experiment = sys.argv[1] #'funnel'
n = sys.argv[2]

if experiment == 'pdb':
    model, D, lp, lp_g, ref_samples, files = chirag_models.setup_pdb(n)

if experiment == 'funnel':
    D = n
    savepath = f'{savepath}/funnel-{D}/'
    model, D, lp, lp_g, ref_samples, files = chirag_models.funnel(D)

if experiment == 'rosenbrock':
    D = n
    savepath = f'{savepath}/rosenbrock-{D}/'
    model, D, lp, lp_g, ref_samples, files = chirag_models.rosenbrock(D)


###################################
##### Setup the algorithm parameters

nleap = 100
step_size = 0.1
nsamples = 10000
burnin = 1000
n_stepsize_adapt = 1000
nleap_adapt = 10
nchains = wsize
target_accept = 0.68
savepath = f"{savepath}/nleap{nleap}/"

debug = False
if debug:
    nsamples = 1000
    burnin = 100
    n_stepsize_adapt = 100
    

#############
# Vanilla HMC
np.random.seed(0)
kernel = DRHMC_AdaptiveStepsize(D, lp, lp_g, mass_matrix=np.eye(D))
q0 = np.random.normal(np.zeros(D*nchains).reshape(nchains, D))
sampler = kernel.sample(q0[wrank], nleap=nleap, step_size=step_size, nsamples=nsamples, burnin=burnin,
                        epsadapt=n_stepsize_adapt, target_accept=target_accept, #nleap_adapt=nleap_adapt,
                        delayed_proposals=False, verbose=False)
print(f"Acceptance for HMC in chain {wrank} : ", np.unique(sampler.accepts, return_counts=True))
if not debug: sampler.save(path=f"{savepath}/hmc/", suffix=f"-{wrank}")
         
comm.Barrier()
samples_hmc = comm.gather(sampler.samples, root=0)
accepts_hmc = comm.gather(sampler.accepts, root=0)
counts_hmc = comm.gather(sampler.counts, root=0)
comm.Barrier()


# Adaptive DRHMC
np.random.seed(0)
kernel = DRHMC_AdaptiveStepsize(D, lp, lp_g, mass_matrix=np.eye(D))
q0 = np.random.normal(np.zeros(D*nchains).reshape(nchains, D))
sampler_adapt = kernel.sample(q0[wrank], nleap=nleap, step_size=step_size, nsamples=nsamples, burnin=burnin,
                         epsadapt=n_stepsize_adapt, target_accept=target_accept, #nleap_adapt=nleap_adapt,
                         delayed_proposals=True, constant_trajectory=False,
                         verbose=False)
print(f"Acceptance for adaptive HMC in chain {wrank} : ", np.unique(sampler_adapt.accepts, return_counts=True))
if not debug : sampler_adapt.save(path=f"{savepath}/adrhmc/", suffix=f"-{wrank}")
comm.Barrier()

samples_adapt = comm.gather(sampler_adapt.samples, root=0)
accepts_adapt = comm.gather(sampler_adapt.accepts, root=0)
steps_adapt = comm.gather(sampler_adapt.steplist, root=0)
comm.Barrier()



if wrank == 0:

    samples_hmc = np.stack(samples_hmc, axis=0)
    samples_adapt = np.stack(samples_adapt, axis=0)
    
    # NUTS
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
        np.save(f'{savepath}/nuts/', sample.step_size)
        
    # plot
    print("\nPlotting")
    plt.figure()
    plt.hist(np.random.normal(0, 3, 100000), density=True, alpha=1, bins='auto', lw=2, histtype='step', color='k', label='Reference')
    plt.hist(samples_hmc[..., 0].flatten(), density=True, alpha=0.5, bins='auto', label='HMC')
    plt.hist(samples_nuts[..., 0].flatten(), density=True, alpha=0.5, bins='auto', label='NUTS');
    plt.hist(samples_adapt[..., 0].flatten(), density=True, alpha=0.5, bins='auto', label='1step ADR-HMC');
    plt.title(f"{D-1} dimension funnel")
    plt.legend()
    plt.savefig('tmp.png')
    plt.close()
    print()
    print("Total accpetances for HMC : ", np.unique(np.stack(accepts_hmc), return_counts=True))

    print("Total accpetances for adaptive HMC : ", np.unique(np.stack(accepts_adapt), return_counts=True))

