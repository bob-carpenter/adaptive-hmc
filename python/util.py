import cmdstanpy as csp
import numpy as np
import scipy as sp
import plotnine as pn
import pandas as pd
import sys
import os
import linecache

def mean_sq_jump_distance(sample):
    sq_jump = []
    M = np.shape(sample)[0]
    for m in range(M - 1):
        jump = sample[m + 1, :] - sample[m, :]
        sq_jump.append(jump.dot(jump))
    return np.mean(sq_jump)


class Sampler():
    '''A simple sampler class to store samples and other use quantities
    over iterations in MCMC chain
    '''
    def __init__(self):
        self.samples = []
        self.accepts = []
        self.Hs = []
        self.counts = []
        self.i = 0

    def to_array(self):
        for key in self.__dict__:
            if type(self.__dict__[key]) == list:
                self.__dict__[key] = np.array(self.__dict__[key])            

    def to_list(self):
        for key in self.__dict__:
            if type(self.__dict__[key]) == np.ndarray:
                self.__dict__[key] = list(self.__dict__[key])

    def appends(self, q, acc, Hs, count):
        self.i += 1
        self.accepts.append(acc)
        self.samples.append(q)
        self.Hs.append(Hs)
        self.counts.append(count)
        
    def save(self, path, suffix="", thin=1):
        os.makedirs(path, exist_ok=True)
        for key in self.__dict__:
            if (type(self.__dict__[key]) == list) or (type(self.__dict__[key]) == np.ndarray):
                np.save(f"{path}/{key}{suffix}", self.__dict__[key][::1])
                



class DualAveragingStepSize():
    """Dual averaging adaptation for tuning step-size to meet a acceptance probability.
    """
    def __init__(self, initial_step_size, target_accept=0.65, gamma=0.05, t0=10.0, kappa=0.75, nadapt=0):
        self.initial_step_size = initial_step_size 
        self.mu = np.log(10 * initial_step_size)  # proposals are biased upwards to stay away from 0.
        self.target_accept = target_accept
        self.gamma = gamma
        self.t = t0
        self.kappa = kappa
        self.error_sum = 0
        self.log_averaged_step = 0
        self.nadapt = nadapt
        
    def update(self, p_accept):

        if np.isnan(p_accept) : p_accept = 0.
        if p_accept > 1: p_accept = 1. 
        # Running tally of absolute error. Can be positive or negative. Want to be 0.
        self.error_sum += self.target_accept - p_accept
        # This is the next proposed (log) step size. Note it is biased towards mu.
        log_step = self.mu - self.error_sum / (np.sqrt(self.t) * self.gamma)
        # Forgetting rate. As `t` gets bigger, `eta` gets smaller.
        eta = self.t ** -self.kappa
        # Smoothed average step size
        self.log_averaged_step = eta * log_step + (1 - eta) * self.log_averaged_step
        # This is a stateful update, so t keeps updating
        self.t += 1

        # Return both the noisy step size, and the smoothed step size
        return np.exp(log_step), np.exp(self.log_averaged_step)

    
    def __call__(self, i, p_accept):
        if i == 0:
            return self.initial_step_size 
        elif i < self.nadapt:
            step_size, avgstepsize = self.update(p_accept)
        elif i == self.nadapt:
            _, step_size = self.update(p_accept)
            print("\nStep size fixed to : %0.3e\n"%step_size)
        else:
            step_size = np.exp(self.log_averaged_step)
        return step_size



def inverse_Hessian_approx(positions, gradients, H=None): # needs to be cleaned based on Hessian_approx code
    '''
    Inverse Hessian approximation with BFGS Quasi-Newton Method
    '''
    d = positions.shape[1]
    npos = positions.shape[0]
    if H is None:
        H = np.eye(d) # initial hessian
    
    nabla = gradients[0]
    x = positions[0]
    
    it = 0 
    not_pos = 0 
    for i in range(npos-1):
        it += 1
        x_new = positions[i+1]
        nabla_new = gradients[i+1]
        s = x_new - x
        y = nabla_new - nabla 
        y = np.array([y])
        s = np.array([s])
        y = np.reshape(y,(d,1)) 
        s = np.reshape(s,(d,1))
        r = 1/(y.T@s)
        if r < 0:               # Curvature condition to ensure positive definite
            not_pos +=1 
            continue
        not_pos = 0 
        li = (np.eye(d)-(r*((s@(y.T)))))
        ri = (np.eye(d)-(r*((y@(s.T)))))
        hess_inter = li@H@ri
        H = hess_inter + (r*((s@(s.T)))) # BFGS Update

        nabla = nabla_new[:].flatten()
        x = x_new[:].flatten()
    
    return H, not_pos



def Hessian_approx(positions, gradients, H=None):
    '''
    Hessian approximation with BFGS Quasi-Newton Method (B matrix)
    '''
    d = positions.shape[1]
    npos = positions.shape[0]
    #if H is None: 
    #    H = np.eye(d) # initial hessian. Moved to initilization below which is better
   
    nabla = gradients[0]
    x = positions[0]
    
    it = 0 
    not_pos = 0 
    for i in range(npos-1):
        it += 1
        x_new = positions[i+1]
        nabla_new = gradients[i+1]
        s = x_new - x
        y = nabla_new - nabla
        r = 1/(np.dot(y,s))
        if r < 0:               #Curvature condition to ensure positive definite
            not_pos +=1 
            continue
        if (H is None) : #initialize based on, but before first update. Taken from Nocedal
            #H = np.eye(x.size) * np.dot(y,s)/np.dot(y, y) #gets confusing if we multiply or divide
            H = np.eye(x.size) / np.dot(y,s) *np.dot(y, y) 
        z = np.dot(H, s)
        update = np.outer(y, y) / np.dot(s, y) - np.outer(z, z) / np.dot(s, z)
        H += update
        nabla = nabla_new[:].flatten()
        x = x_new[:].flatten()
    return H, not_pos

    
def power_iteration(A, num_iters=100):

    # Starting vector
    b = np.random.rand(A.shape[0])
    # Power iteration
    for ii in range(num_iters):
        # Project
        bnew = A @ b
        # Normalize
        b = bnew / np.linalg.norm(bnew, ord=2)
        
    eigval = (A @ b)@b/(b@b)
    return eigval, b


def PrintException():
    '''
    Useful piece of code for debugging with try-except clause.
    Gives the line where exception occured and other details.
    '''
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    print ('EXCEPTION IN ({}, LINE {} "{}"): {}'.format(filename, lineno, line.strip(), exc_obj))
