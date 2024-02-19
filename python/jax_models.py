import jax
import jax.numpy as jnp
from jax import	grad, jit, vmap
from functools import partial

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
#############
class Rosenbrock():
    #https://arxiv.org/pdf/1903.09556.pdf
    def __init__(self, D):
        self.D = D
    
    @partial(jit, static_argnums=(0))
    def log_density(self, x):        
        lpx = 0
        for i in range(self.D - 1):
            lpx += 100*(x[i+1] - x[i]**2)**2 + (1-x[i])**2
        lpx *= -1/20.
        return lpx
    
    @partial(jit, static_argnums=(0))
    def log_density_batch(self, x):
        lp = jnp.sum(vmap(self.log_density)(x))
        return lp
    

    @partial(jit, static_argnums=(0))
    def log_density_gradient(self, x):
        lpg = grad(self.log_density)(x)
        return lpg
    
    @partial(jit, static_argnums=(0))
    def log_density_gradient_batch(self, x):
        lpg = grad(self.log_density_batch)(x)
        return lpg

    
