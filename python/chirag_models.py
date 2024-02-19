import sys
import jax_models
import bridgestan as bs
BRIDGESTAN = "/mnt/home/cmodi/Research/Projects/bridgestan/"
bs.set_bridgestan_path(BRIDGESTAN)


##### Setup the model
def setup_pdb(model_n):
    sys.path.append('/mnt/home/cmodi/Research/Projects/posterior_database/')
    from posteriordb import BSDB

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



### setup funnel
def funnel(D):
    bsmodel = bs.StanModel.from_stan_file(f"../stan/funnel_{D}.stan", 
                                          f"../stan/funnel.data.json")

    D = bsmodel.param_num()
    lp = lambda x: bsmodel.log_density(x)
    lp_g = lambda x: bsmodel.log_density_gradient(x)[1]
    return bsmodel, D, lp, lp_g, None

### setup funnel
def rosenbrock(D):
    bsmodel = bs.StanModel.from_stan_file(f"../stan/rosenbrock_{D}.stan", 
                                          f"../stan/rosenbrock.data.json")

    D = bsmodel.param_num()
    lp = lambda x: bsmodel.log_density(x)
    lp_g = lambda x: bsmodel.log_density_gradient(x)[1]
    return bsmodel, D, lp, lp_g, None


### setup rosenbrock
def rosenbrock_full(D):
    
    model = jax_models.Rosenbrock(D)
    lp =  model.log_density
    lp_g = model.log_density_gradient
    return model, D, lp, lp_g, None


