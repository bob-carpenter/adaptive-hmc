import numpy as np
import sys, time
import matplotlib.pyplot as plt

sys.path.append('/mnt/home/cmodi/Research/Projects/posterior_database/')
from posteriordb import BSDB
from jax_utils import jaxify_bs
from dr_hmc import DRHMC_AdaptiveStepsize

import bridgestan as bs
BRIDGESTAN = "/mnt/home/cmodi/Research/Projects/bridgestan/"
bs.set_bridgestan_path(BRIDGESTAN)


def setup_pdb(model_n):

    model = BSDB(model_n)
    D = model.dims
    lp = model.lp
    lp_g = lambda x: model.lp_g(x)[1]
    try:
        ref_samples = model.samples_unc.copy()
    except Exception as e:
        print(e)
        ref_samples = None

    return model, D, lp, lp_g, ref_samples


#model, D, lp, lp_g, ref_samples = setup_pdb(31)
bsmodel = bs.StanModel.from_stan_file("/mnt/home/cmodi/Research/Projects/posteriordb/compiled_models/funnel_10/funnel_10.stan", 
                                      "/mnt/home/cmodi/Research/Projects/posteriordb/compiled_models/funnel_10/funnel_10.data.json")
D = bsmodel.param_num()
print("Dimensions : ", D)
lp = lambda x: bsmodel.log_density(x)
lp_g = lambda x: bsmodel.log_density_gradient(x)[1]
kernel = DRHMC_AdaptiveStepsize(D, lp, lp_g, mass_matrix=np.eye(D))


nleap = 20
step_size = 0.01
nsamples = 100000
epsadapt = 1000


np.random.seed(0)
q0 = np.random.normal(np.zeros(D))
sampler = kernel.sample(q0, nleap=nleap, step_size=step_size, nsamples=nsamples, burnin=0, epsadapt=epsadapt, target_accept=0.8,
                        delayed_proposals=False, verbose=False)
print(np.unique(sampler.accepts, return_counts=True))

start = time.time()
np.random.seed(0)
q0 = np.random.normal(np.zeros(D))
sampler2 = kernel.sample(q0, nleap=nleap, step_size=step_size, nsamples=nsamples, burnin=0, epsadapt=epsadapt, target_accept=0.8,
                         delayed_proposals=True, verbose=False)
print(np.unique(sampler2.accepts, return_counts=True))
end = time.time()


# plot
plt.figure()
plt.hist(np.random.normal(0, 3, 10000), density=True, alpha=1, bins='auto', lw=2, histtype='step', color='k')
plt.hist(sampler.samples[:, 0], density=True, alpha=0.5, bins='auto')
plt.hist(sampler2.samples[:, 0], density=True, alpha=0.5, bins='auto');
plt.savefig('tmp.png')
plt.close()
