import numpy as np
import os, sys, time
import matplotlib.pyplot as plt

from algorithms import  HMC_Uturn, HMC
import util

import logging
import cmdstanpy as csp
csp.utils.get_logger().setLevel(logging.ERROR)

import chirag_models

savepath = '/mnt/ceph/users/cmodi/adaptive_hmc/'

#######
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--exp', type=str, help='which experiment')
parser.add_argument('-n', type=int, default=0, help='dimensionality or model number')
#arguments for GSM
parser.add_argument('--seed', type=int, default=999, help='seed')
parser.add_argument('--nchains', type=int, default=16, help='number of chains')
parser.add_argument('--nleap', type=int, default=40, help='number of leapfrog steps')
parser.add_argument('--nsamples', type=int, default=1001, help='number of samples')
parser.add_argument('--stepadapt', type=int, default=1000, help='step size adaptation')
parser.add_argument('--targetaccept', type=float, default=0.80, help='target acceptance')
parser.add_argument('--stepsize', type=float, default=0.1, help='initial step size')
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



###################################
##### Setup the algorithm parameters
step_size = args.stepsize
nsamples = args.nsamples
n_stepsize_adapt = args.stepadapt
target_accept = args.targetaccept
print(f"Saving runs in parent folder : {savepath}")


###################################
# NUTS
np.random.seed(999)
savefolder = f"{savepath}/nuts/target{args.targetaccept:0.2f}/"
os.makedirs(savefolder, exist_ok=True)
print("\nNow run NUTS on rank 0")
stanfile, datafile = files
cmd_model = csp.CmdStanModel(stan_file = stanfile)
sample = cmd_model.sample(data=datafile, chains=args.nchains, iter_sampling=nsamples-1,
                          seed = args.seed,
                          metric="unit_e",
                          #step_size=args.stepsize,
                          #adapt_engaged=False,
                          adapt_delta=target_accept,
                          adapt_metric_window=0,
                          adapt_init_phase=n_stepsize_adapt,
                          adapt_step_size=n_stepsize_adapt,
                          show_console=False, show_progress=True, save_warmup=False)
draws_pd = sample.draws_pd()
samples_nuts, leapfrogs_nuts = util.cmdstanpy_wrapper(draws_pd, savepath=f'{savefolder}/')
np.save(f'{savefolder}/stepsize', sample.step_size)

difference = np.diff(samples_nuts[..., 0])
print("accept/reject for NUTS: ", difference.size - (difference == 0 ).sum(),  (difference == 0 ).sum())
step_size = sample.step_size
        

#####################
# plot
plt.figure()
#plt.hist(np.random.normal(0, 3, 100000), density=True, alpha=1, bins='auto', lw=2, histtype='step', color='k', label='Reference')
plt.hist(ref_samples[..., 0].flatten(), density=True, alpha=1, bins='auto', lw=2, histtype='step', color='k', label='Reference')
plt.hist(samples_nuts[..., 0].flatten(), density=True, alpha=0.5, bins='auto', label='NUTS');

print("\nPlotting")
plt.legend()
plt.title(savefolder.split(f'nuts/')[1][:-1])
plt.savefig('tmp.png')
plt.savefig(f"{savefolder}/hist")
plt.close()


