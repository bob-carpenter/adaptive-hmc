import orbital_sampler as os
import cmdstanpy as csp
import numpy as np
import bridgestan as bs
import plotnine as pn
import pandas as pd
import logging
import traceback
import warnings

def orbital_experiment(
        program,
        stepsize,
        steps,
        seed):
    program_path = program + '.stan'
    data_path = program + '.json'
    model_bs = bs.StanModel(modellib = program_path, data = data_path, capture_stan_prints=False)
    rng = np.random.default_rng(seed)
    theta = rng.normal(
    sampler = os.OrbitalSampler(model_bs, stepsize=0.3, rng=rng, 
    

orbital_experiment('normal', 0.3, 8, 12349876)
        
