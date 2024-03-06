import numpy as np
import os, sys
import json
import jax_models
import bridgestan as bs
BRIDGESTAN = "/mnt/home/cmodi/Research/Projects/bridgestan/"
bs.set_bridgestan_path(BRIDGESTAN)

def write_tmpdata(D, datapath):
    # Data to be written
    dictionary = {
        "D": D,
    }
    # Serializing json
    json_object = json.dumps(dictionary, indent=4)
    print(dictionary)
    # Writing to sample.json
    with open(f"{datapath}", "w") as outfile:
        outfile.write(json_object)

    return f"{datapath}"


##### Setup the models
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

    return model, D, lp, lp_g, ref_samples, None


def normal(D):
    stanfile = f"../stan/normal.stan"
    datafile = write_tmpdata(D, f"../stan/normal.data.json")
    bsmodel = bs.StanModel.from_stan_file(stanfile, datafile)
    os.remove(datafile)
    
    D = bsmodel.param_num()
    lp = lambda x: bsmodel.log_density(x)
    lp_g = lambda x: bsmodel.log_density_gradient(x)[1]
    return bsmodel, D, lp, lp_g, None, [stanfile, datafile]


def funnel(D):
    stanfile = f"../stan/funnel.stan"
    datafile = write_tmpdata(D, f"../stan/funnel.data.json")
    bsmodel = bs.StanModel.from_stan_file(stanfile, datafile)
    #os.remove(datafile)    

    D = bsmodel.param_num()
    lp = lambda x: bsmodel.log_density(x)
    lp_g = lambda x: bsmodel.log_density_gradient(x)[1]
    ref_samples = np.load('/mnt/ceph/users/cmodi/PosteriorDB/funnel/samples.npy')
    return bsmodel, D, lp, lp_g, ref_samples, [stanfile, datafile]


def rosenbrock(D):
    stanfile = f"../stan/rosenbrock.stan" 
    datafile = write_tmpdata(D, f"../stan/rosenbrock.data.json")
    bsmodel = bs.StanModel.from_stan_file(stanfile, datafile)
    #os.remove(datafile)    

    D = bsmodel.param_num()
    lp = lambda x: bsmodel.log_density(x)
    lp_g = lambda x: bsmodel.log_density_gradient(x)[1]
    ref_samples = np.load('/mnt/ceph/users/cmodi/PosteriorDB/rosenbrock/samples.npy')
    return bsmodel, D, lp, lp_g, ref_samples, [stanfile, datafile]

def stochastic_volatility():
    name = 'stochastic_volatility'
    stanfile = f"../stan/{name}.stan" 
    datafile = f"../stan/{name}.data.json"
    bsmodel = bs.StanModel.from_stan_file(stanfile, datafile)

    D = bsmodel.param_num()
    lp = lambda x: bsmodel.log_density(x)
    lp_g = lambda x: bsmodel.log_density_gradient(x)[1]
    ref_samples = np.load(f'/mnt/ceph/users/cmodi/PosteriorDB/{name}/samples.npy')
    return bsmodel, D, lp, lp_g, ref_samples, [stanfile, datafile]

def eight_schools():
    name = 'eight_schools'
    stanfile = f"../stan/{name}.stan" 
    datafile = f"../stan/{name}.data.json"
    bsmodel = bs.StanModel.from_stan_file(stanfile, datafile)

    D = bsmodel.param_num()
    lp = lambda x: bsmodel.log_density(x)
    lp_g = lambda x: bsmodel.log_density_gradient(x)[1]
    ref_samples = np.load(f'/mnt/ceph/users/cmodi/PosteriorDB/{name}/samples.npy')
    return bsmodel, D, lp, lp_g, ref_samples, [stanfile, datafile]

def eight_schools_centered():
    name = 'eight_schools_centered'
    stanfile = f"../stan/{name}.stan" 
    datafile = f"../stan/{name}.data.json"
    bsmodel = bs.StanModel.from_stan_file(stanfile, datafile)

    D = bsmodel.param_num()
    lp = lambda x: bsmodel.log_density(x)
    lp_g = lambda x: bsmodel.log_density_gradient(x)[1]
    ref_samples = np.load(f'/mnt/ceph/users/cmodi/PosteriorDB/{name}/samples.npy')
    return bsmodel, D, lp, lp_g, ref_samples, [stanfile, datafile]


def irt():
    name = 'irt'
    stanfile = f"../stan/{name}.stan" 
    datafile = f"../stan/{name}.data.json"
    bsmodel = bs.StanModel.from_stan_file(stanfile, datafile)

    D = bsmodel.param_num()
    lp = lambda x: bsmodel.log_density(x)
    lp_g = lambda x: bsmodel.log_density_gradient(x)[1]
    ref_samples = np.load(f'/mnt/ceph/users/cmodi/PosteriorDB/{name}/samples.npy')
    return bsmodel, D, lp, lp_g, ref_samples, [stanfile, datafile]

# ### setup funnel
# def funnel(D):
#     stanfile = f"../stan/funnel_{D}.stan"
#     datafile = f"../stan/funnel.data.json"
#     bsmodel = bs.StanModel.from_stan_file(stanfile, datafile)

#     D = bsmodel.param_num()
#     lp = lambda x: bsmodel.log_density(x)
#     lp_g = lambda x: bsmodel.log_density_gradient(x)[1]
#     return bsmodel, D, lp, lp_g, None, [stanfile, datafile]


# ### setup rosenbrock
# def rosenbrock_full(D):
    
#     model = jax_models.Rosenbrock(D)
#     lp =  model.log_density
#     lp_g = model.log_density_gradient
#     return model, D, lp, lp_g, None


