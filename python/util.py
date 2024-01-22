import cmdstanpy as csp
import numpy as np
import scipy as sp
import plotnine as pn
import pandas as pd

def mean_sq_jump_distance(sample):
    sq_jump = []
    M = np.shape(sample)[0]
    for m in range(M - 1):
        jump = sample[m + 1, :] - sample[m, :]
        sq_jump.append(jump.dot(jump))
    return np.mean(sq_jump)
