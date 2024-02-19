import numpy as np
import sys, time
import matplotlib.pyplot as plt

sys.path.append('/mnt/home/cmodi/Research/Projects/posterior_database/')
from posteriordb import BSDB
from jax_utils import jaxify_bs
from dr_hmc import DRHMC_AdaptiveStepsize
from mpi4py import MPI

import bridgestan as bs
BRIDGESTAN = "/mnt/home/cmodi/Research/Projects/bridgestan/"
bs.set_bridgestan_path(BRIDGESTAN)

comm = MPI.COMM_WORLD
wrank = comm.Get_rank()
wsize = comm.Get_size()
print('My rank is ',wrank)
savepath = '/mnt/ceph/users/cmodi/adaptive_hmc/'

#######
import chirag_models
experiment = 'rosenbrock'
n = 20

if experiment == 'pdb':
    model, D, lp, lp_g, ref_samples = chirag_models.setup_pdb(n)

if experiment == 'funnel':
    D = n
    savepath = f'{savepath}/funnel-{D}/'
    model, D, lp, lp_g, ref_samples = chirag_models.funnel(D)

if experiment == 'rosenbrock':
    D = n
    savepath = f'{savepath}/rosenbrock-{D}/'
    model, D, lp, lp_g, ref_samples = chirag_models.rosenbrock(D)


###################################
##### Setup the algorithm parameters

nleap = 40
step_size = 0.1
nsamples = 10000
epsadapt = 1000
nleap_adapt = 10
nchains = wsize
target_accept = 0.68

# Vanilla HMC
np.random.seed(0)
kernel = DRHMC_AdaptiveStepsize(D, lp, lp_g, mass_matrix=np.eye(D))
q0 = np.random.normal(np.zeros(D*nchains).reshape(nchains, D))
sampler = kernel.sample(q0[wrank], nleap=nleap, step_size=step_size, nsamples=nsamples, burnin=0,
                        epsadapt=epsadapt, target_accept=target_accept, #nleap_adapt=nleap_adapt,
                        delayed_proposals=False, verbose=False)
print(np.unique(sampler.accepts, return_counts=True))
sampler.save(path=f"{savepath}/hmc/", suffix=f"-{wrank}")
         
comm.Barrier()
all_samples = comm.gather(sampler.samples, root=0)
all_accepts = comm.gather(sampler.accepts, root=0)
comm.Barrier()


# Adaptive DRHMC
np.random.seed(0)
kernel = DRHMC_AdaptiveStepsize(D, lp, lp_g, mass_matrix=np.eye(D))
q0 = np.random.normal(np.zeros(D*nchains).reshape(nchains, D))
sampler2 = kernel.sample(q0[wrank], nleap=nleap, step_size=step_size, nsamples=nsamples, burnin=0,
                         epsadapt=epsadapt, target_accept=target_accept, #nleap_adapt=nleap_adapt,
                         delayed_proposals=True, constant_trajectory=False,
                         verbose=False)
print(np.unique(sampler2.accepts, return_counts=True))
sampler2.save(path=f"{savepath}/adrhmc/", suffix=f"-{wrank}")
comm.Barrier()

all_samples2 = comm.gather(sampler2.samples, root=0)
all_accepts2 = comm.gather(sampler2.accepts, root=0)
all_steps2 = comm.gather(sampler2.steplist, root=0)
comm.Barrier()

if wrank == 0:
    # plot
    plt.figure()
    plt.hist(np.random.normal(0, 3, 100000), density=True, alpha=1, bins='auto', lw=2, histtype='step', color='k', label='Reference')
    plt.hist(np.concatenate(all_samples, axis=0)[:, 0], density=True, alpha=0.5, bins='auto', label='HMC')
    plt.hist(np.concatenate(all_samples2, axis=0)[:, 0], density=True, alpha=0.5, bins='auto', label='1step ADR-HMC');
    plt.title(f"{D-1} dimension funnel")
    plt.legend()
    plt.savefig('tmp.png')
    plt.close()
    #np.save('samples', np.stack(all_samples, axis=0))
    #np.save('samples2', np.stack(all_samples2, axis=0))
    print()
    print(np.unique(np.stack(all_accepts), return_counts=True))
    print(np.unique(np.stack(all_accepts2), return_counts=True))
    #np.save('stepsizes', np.stack(all_steps2, axis=0))

