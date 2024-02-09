import numpy as np
import sys
sys.path.append('/mnt/home/cmodi/Research/Projects/posterior_database/')
from posteriordb import BSDB
from jax_utils import jaxify_bs
from dr_hmc import DRHMC_AdaptiveStepsize


def setup_pdb(model_n):

    model = BSDB(model_n)
    D = model.dims
    lp = model.lp
    lp_g = lambda x: model.lp_g(x)[1]
    #lpjax, lp_itemjax = jaxify_bs(model)
    #lpjaxsum = jit(lambda x: jnp.sum(lpjax(x)))
    try:
        ref_samples = model.samples_unc.copy()
    except Exception as e:
        print(e)
        ref_samples = None

    return model, D, lp, lp_g, ref_samples


model, D, lp, lp_g, ref_samples = setup_pdb(31)
kernel = DRHMC_AdaptiveStepsize(D, lp, lp_g, mass_matrix=np.eye(D))


nleap = 20
step_size = 0.1
nsamples = 1000
epsadapt = 0


q0 = np.random.normal(np.zeros(D))
np.random.seed(0)
sampler = kernel.sample(q0, nleap=nleap, step_size=step_size, nsamples=nsamples, burnin=0, epsadapt=epsadapt, delayed_proposals=True, verbose=True)
print(np.unique(sampler.accepts, return_counts=True))

# q0 = np.random.normal(np.zeros(D))
# np.random.seed(0)
# sampler = kernel.sample(q0, nleap=nleap, step_size=step_size, nsamples=nsamples, burnin=0, epsadapt=epsadapt, delayed_proposals=True, verbose=True)
# print(np.unique(sampler.accepts, return_counts=True))
