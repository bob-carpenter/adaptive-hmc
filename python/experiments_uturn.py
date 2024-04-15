import numpy as np
import os, sys, time
import matplotlib.pyplot as plt

from algorithms import DRHMC_AdaptiveStepsize, DRHMC_AdaptiveStepsize_autotune, HMC_uturn, HMC
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
parser.add_argument('-n', type=int, default=0, help='dimensionality or model number')
#arguments for GSM
parser.add_argument('--nleap', type=int, default=40, help='number of leapfrog steps')
parser.add_argument('--nsamples', type=int, default=1001, help='number of samples')
parser.add_argument('--burnin', type=int, default=10, help='number of iterations for burn-in')
parser.add_argument('--stepadapt', type=int, default=100, help='step size adaptation')
parser.add_argument('--targetaccept', type=float, default=0.80, help='target acceptance')
parser.add_argument('--stepsize', type=float, default=0.1, help='initial step size')
parser.add_argument('--dist', type=str, default='uniform', help='distribution for u-turn sampler')
parser.add_argument('--offset', type=float, default=0.5, help='offset for uturn sampler')
parser.add_argument('--pbinom', type=float, default=0.6, help='binomial log-prob')
parser.add_argument('--hmc', type=int, default=0, help='run hmc')
parser.add_argument('--nuts', type=int, default=1, help='run nuts')
parser.add_argument('--symmetric', type=int, default=1, help='u turn condition on both sides')
parser.add_argument('--debug', type=int, default=0, help='constant trajectory for delayed')
#arguments for path name
parser.add_argument('--suffix', type=str, default="", help='suffix, default=""')


args = parser.parse_args()
experiment = args.exp
n = args.n

if experiment == 'pdb':
    model, D, lp, lp_g, ref_samples, files = chirag_models.setup_pdb(n)
else :
    print("Model name : ", experiment)
    if n == 0:
        model, D, lp, lp_g, ref_samples, files = chirag_models.setup_model(experiment)
        savepath = f'/mnt/ceph/users/cmodi/adaptive_hmc/{experiment}'
    else:
        D = n
        model, D, lp, lp_g, ref_samples, files = chirag_models.setup_model(experiment, D) 
        savepath = f'/mnt/ceph/users/cmodi/adaptive_hmc/{experiment}-{D}/'

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
    savefolder = f"{savepath}/nuts/"
    if args.nuts == 0 :
        print("\nTrying to load NUTS results on rank 0")
        try:
            samples_nuts = np.load(f"{savefolder}/samples.npy")
            step_size = np.load(f"{savefolder}/stepsize.npy")
            assert samples_nuts.shape[0] == wsize
            print(f"Loaded nuts samples and stepsize from {savefolder}")
        except Exception as e:
            print("Exception in loading NUTS results : ", e)
            args.nuts = 1
    
    if args.nuts == 1:
        print("\nNow run NUTS on rank 0")
        stanfile, datafile = files
        cmd_model = csp.CmdStanModel(stan_file = stanfile)
        sample = cmd_model.sample(data=datafile, chains=wsize, iter_sampling=nsamples-1, 
                              metric="unit_e",
                              adapt_delta=target_accept,
                              adapt_metric_window=0,
                              adapt_init_phase=n_stepsize_adapt//2,
                              adapt_step_size=n_stepsize_adapt//2,
                              show_console=False, show_progress=True, save_warmup=False)
        draws_pd = sample.draws_pd()
        if not debug:
            samples_nuts, leapfrogs_nuts = util.cmdstanpy_wrapper(draws_pd, savepath=f'{savefolder}/')
            np.save(f'{savefolder}/stepsize', sample.step_size)
        else:
            samples_nuts, leapfrogs_nuts = util.cmdstanpy_wrapper(draws_pd, savepath=None)

        difference = np.diff(samples_nuts[..., 0])
        print("accept/reject for NUTS: ", difference.size - (difference == 0 ).sum(),  (difference == 0 ).sum())
        step_size = sample.step_size
        
else:
    step_size = 0.
    samples_nuts = np.zeros([nchains, D])
comm.Barrier()

# Scatter step size and initial point to different ranks
comm.Barrier()
if wrank == 0 : print()
step_size = comm.scatter(step_size, root=0)
q0 = comm.scatter(samples_nuts[:, 0], root=0)
q0 = model.param_unconstrain(q0)
print(f"Step size in rank {wrank}: ", step_size)
comm.Barrier()


#####################
# UTurn
print("Run U-turn sampler")
np.random.seed(0)
#q0 = np.random.normal(np.zeros(D*nchains).reshape(nchains, D))[wrank]
kernel = HMC_uturn(D, lp, lp_g, mass_matrix=np.eye(D), min_nleap=None,
                   distribution=args.dist, offset=args.offset, p_binom=args.pbinom, symmetric=bool(args.symmetric))
sampler = kernel.sample(q0, nleap=args.nleap, step_size=step_size, nsamples=nsamples, burnin=burnin,
                        epsadapt=0., #n_stepsize_adapt,
                        target_accept=target_accept, 
                        verbose=False)

print(f"Acceptance for Uturn HMC in chain {wrank} : ", np.unique(sampler.accepts, return_counts=True))

if args.dist=='uniform':
    savefolder = f"{savepath}/uturn/offset{args.offset:0.2f}/"
elif args.dist=='binomial':
    savefolder = f"{savepath}/uturn/pbinom{args.pbinom:0.2f}/"
if not bool(args.symmetric):
    savefolder = f"{savefolder}"[:-1] + "-asymm/"
    
os.makedirs(f'{savefolder}', exist_ok=True)
print(f"Saving runs in folder : {savefolder}")
if not debug: sampler.save(path=f"{savefolder}", suffix=f"-{wrank}")
samples_constrained = []
for s in sampler.samples:
    samples_constrained.append(model.param_constrain(s))
np.save(f"{savefolder}/samples_constrained-{wrank}", samples_constrained)
comm.Barrier()

samples_uturn = comm.gather(sampler.samples, root=0)
accepts_uturn = comm.gather(sampler.accepts, root=0)
stepsizes = comm.gather(kernel.step_size, root=0)
if wrank == 0 :
    print(stepsizes)
    np.save(f'{savefolder}/stepsize', stepsizes)
comm.Barrier()

#####################
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
    plt.savefig(f"{savefolder}/hist")
    plt.close()

    print()
    #print("Total accpetances for HMC : ", np.unique(np.stack(accepts_hmc), return_counts=True))
    print("Total accpetances for adaptive HMC : ", np.unique(np.stack(accepts_uturn), return_counts=True))

comm.Barrier()


#####################
# HMC
if args.hmc:
    print("Run HMC sampler")
    np.random.seed(0)
    #q0 = np.random.normal(np.zeros(D*nchains).reshape(nchains, D))[wrank]
    kernel = HMC(D, lp, lp_g, mass_matrix=np.eye(D))
    sampler = kernel.sample(q0, nleap=args.nleap, step_size=step_size, nsamples=nsamples, burnin=burnin,
                            epsadapt=n_stepsize_adapt,
                            target_accept=target_accept, 
                            verbose=False)

    print(f"Acceptance for Uturn HMC in chain {wrank} : ", np.unique(sampler.accepts, return_counts=True))
    savefolder = f"{savepath}/hmc/nleap{args.nleap}/"
    os.makedirs(f'{savefolder}', exist_ok=True)
    if not debug: sampler.save(path=f"{savefolder}", suffix=f"-{wrank}")
    samples_constrained = []
    for s in sampler.samples:
        samples_constrained.append(model.param_constrain(s))
    np.save(f"{savefolder}/samples_constrained-{wrank}", samples_constrained)

    comm.Barrier()
    samples_hmc = comm.gather(sampler.samples, root=0)
    accepts_hmc = comm.gather(sampler.accepts, root=0)
    stepsizes = comm.gather(kernel.step_size, root=0)
    if wrank == 0 :
        print(stepsizes)
        np.save(f'{savefolder}/stepsize', stepsizes)
    comm.Barrier()

    

sys.exit()
