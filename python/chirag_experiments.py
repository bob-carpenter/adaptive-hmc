import numpy as np
import sys, time
import matplotlib.pyplot as plt

from dr_hmc import DRHMC_AdaptiveStepsize, DRHMC_AdaptiveStepsize_autotune
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
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp', type=str, help='which experiment')
parser.add_argument('-n', type=int, help='dimensionality or model number')
#arguments for GSM
parser.add_argument('--nleap', type=int, default=40, help='number of leapfrog steps')
parser.add_argument('--nsamples', type=int, default=1001, help='number of samples')
parser.add_argument('--burnin', type=int, default=10, help='number of iterations for burn-in')
parser.add_argument('--stepadapt', type=int, default=100, help='step size adaptation')
parser.add_argument('--nleapadapt', type=int, default=100, help='step size adaptation')
parser.add_argument('--targetaccept', type=float, default=0.80, help='target acceptance')
parser.add_argument('--stepsize', type=float, default=0.1, help='initial step size')
parser.add_argument('--adapt_traj', type=int, default=1, help='adapt trajectory')
parser.add_argument('--constant_traj', type=int, default=0, help='constant trajectory for delayed')
#arguments for path name
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')


args = parser.parse_args()
experiment = args.exp
n = args.n

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

nleap = args.nleap
step_size = args.stepsize
nsamples = args.nsamples
burnin = args.burnin
n_stepsize_adapt = args.stepadapt
nleap_adapt = args.nleapadapt
nchains = wsize
target_accept = args.targetaccept
adapt_traj = bool(args.adapt_traj)
constant_traj = bool(args.constant_traj)
if adapt_traj:
    print("Adapting trajectory length")
    savepath = f"{savepath}/autotune/"
else:
    savepath = f"{savepath}/nleap{nleap}/"
print(f"Saving runs in folder : {savepath}")

debug = True
if debug:
    nsamples = 1000
    burnin = 100
    n_stepsize_adapt = 100
    

#############
# Vanilla HMC
np.random.seed(0)
q0 = np.random.normal(np.zeros(D*nchains).reshape(nchains, D))
if adapt_traj:
    kernel = DRHMC_AdaptiveStepsize_autotune(D, lp, lp_g, mass_matrix=np.eye(D), min_nleap=10)
    sampler = kernel.sample(q0[wrank], nleap=nleap, step_size=step_size, nsamples=nsamples, burnin=burnin,
                        epsadapt=n_stepsize_adapt, target_accept=target_accept, nleap_adapt=nleap_adapt,
                        delayed_proposals=False, verbose=False)
else:
    kernel = DRHMC_AdaptiveStepsize(D, lp, lp_g, mass_matrix=np.eye(D))
    sampler = kernel.sample(q0[wrank], nleap=nleap, step_size=step_size, nsamples=nsamples, burnin=burnin,
                        epsadapt=n_stepsize_adapt, target_accept=target_accept, 
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
q0 = np.random.normal(np.zeros(D*nchains).reshape(nchains, D))
if adapt_traj:
    kernel = DRHMC_AdaptiveStepsize_autotune(D, lp, lp_g, mass_matrix=np.eye(D), min_nleap=10)
    sampler_adapt = kernel.sample(q0[wrank], nleap=nleap, step_size=step_size, nsamples=nsamples, burnin=burnin,
                         epsadapt=n_stepsize_adapt, target_accept=target_accept, nleap_adapt=nleap_adapt,
                         delayed_proposals=True, constant_trajectory=constant_traj,
                         verbose=False)
else:
    kernel = DRHMC_AdaptiveStepsize(D, lp, lp_g, mass_matrix=np.eye(D))
    sampler_adapt = kernel.sample(q0[wrank], nleap=nleap, step_size=step_size, nsamples=nsamples, burnin=burnin,
                         epsadapt=n_stepsize_adapt, target_accept=target_accept, 
                         delayed_proposals=True, constant_trajectory=constant_traj,
                         verbose=False)

    
print(f"Acceptance for adaptive HMC in chain {wrank} : ", np.unique(sampler_adapt.accepts, return_counts=True))
if not debug :
    if constant_traj:
        sampler_adapt.save(path=f"{savepath}/adrhmc_ctraj/", suffix=f"-{wrank}")
    else:
        sampler_adapt.save(path=f"{savepath}/adrhmc/", suffix=f"-{wrank}")
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
        np.save(f'{savepath}/nuts/stepsize', sample.step_size)
    else:
        samples_nuts, leapfrogs_nuts = util.cmdstanpy_wrapper(draws_pd, savepath=None)
        
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

comm.Barrier()

sys.exit()
