import sys
import numpy as np
from scipy.stats import multivariate_normal
from util import Sampler, DualAveragingStepSize, PrintException


class HMC():

    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None):

        self.D = D
        self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
        self.V = lambda x : self.log_prob(x)*-1.

        if mass_matrix is None: self.mass_matrix = np.eye(D)
        else: self.mass_matrix = mass_matrix
        self.inv_mass_matrix = np.linalg.inv(self.mass_matrix)
        
        self.leapcount = 0
        self.Vgcount = 0
        self.Hcount = 0


    def adapt_stepsize(self, q, epsadapt, target_accept=0.65):
        print("Adapting step size for %d iterations"%epsadapt)
        step_size = self.step_size
        epsadapt_kernel = DualAveragingStepSize(step_size, target_accept=target_accept)

        for i in range(epsadapt+1):
            qprev = q.copy()
            q, p, acc, Hs, count = self.hmc_step(q, self.nleap, step_size)
            if (qprev == q).all():
                prob = 0 
            else:
                prob = np.exp(Hs[0] - Hs[1])

            if i < epsadapt:
                if np.isnan(prob) or np.isinf(prob): 
                    prob = 0.
                    continue
                if prob > 1: prob = 1.
                step_size, avgstepsize = epsadapt_kernel.update(prob)
            elif i == epsadapt:
                _, step_size = epsadapt_kernel.update(prob)
                print("Step size fixed to : ", step_size)
                self.step_size = step_size
        return q
        

    def V_g(self, x):
        self.Vgcount += 1
        v_g = self.grad_log_prob(x)
        return v_g *-1.

    
    def H(self, q, p, M=None):
        self.Hcount += 1
        Vq = self.V(q)
        KE, _ = self.setup_KE(M)
        Kq = KE(p)
        return Vq + Kq


    def setup_KE(self, M):      # This is unnecessary if we are not adapting mass matrix. 
        if M is None:
            M = self.mass_matrix 
        KE =  lambda p : 0.5*np.dot(p, np.dot(M, p))
        KE_g =  lambda p : np.dot(M, p)
        return KE, KE_g
        

    def leapfrog(self, q, p, N, step_size, M=None, g=None):
        self.leapcount += 1
        
        KE, KE_g = self.setup_KE(M)    
        qvec, gvec = [], []
        q0, p0 = q, p
        g0 = g
        try:
            if g0 is not None:
                g = g0
                g0 = None
            else:
                g =  self.V_g(q)
            p = p - 0.5*step_size * g
            qvec.append(q)
            gvec.append(g)
            for i in range(N-1):
                q = q + step_size * KE_g(p)
                g = self.V_g(q)
                p = p - step_size * g
                qvec.append(q)
                gvec.append(g)
            q = q + step_size * KE_g(p)
            g = self.V_g(q)
            p = p - 0.5*step_size * g
            qvec.append(q)
            gvec.append(g)            
            return q, p, qvec, gvec

        except Exception as e:  # Sometimes nans happen. 
            return q0, p0, qvec, gvec


    def accept_log_prob(self, qp0, qp1, return_H=False):
        q0, p0 = qp0
        q1, p1 = qp1
        H0 = self.H(q0, p0)
        H1 = self.H(q1, p1)
        log_prob = H0 - H1
        if np.isnan(log_prob)  or (q0-q1).sum()==0:
            log_prob = -np.inf
        log_prob = min(0., log_prob)
        if return_H is False: return log_prob
        else: return log_prob, H0, H1
    

    def metropolis(self, qp0, qp1, M=None):

        log_prob, H0, H1 = self.accept_log_prob(qp0, qp1, return_H=True)
        q0, p0 = qp0
        q1, p1 = qp1
        if np.isnan(log_prob) or (q0-q1).sum()==0:
            return q0, p0, -1, [H0, H1]
        else:
            u =  np.random.uniform(0., 1., size=1)
            if  np.log(u) > min(0., log_prob):
                return q0, p0, 0., [H0, H1]
            else:
                return q1, p1, 1., [H0, H1]
            

    def hmc_step(self, q, nleap=None, step_size=None): # Quality of life function to inherit for standard stepping

        if nleap is None: nleap = self.nleap
        if step_size is None: step_size = self.step_size
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0 # reset counts
        
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
        q1, p1, qvec, gvec = self.leapfrog(q, p, N=nleap, step_size=step_size)
        qf, pf, accepted, Hs = self.metropolis([q, p], [q1, p1])
        counts = [self.Hcount, self.Vgcount, self.leapcount]
        return qf, pf, accepted, Hs, counts

    
    def step(self, q, nleap=None, step_size=None):
        
        return self.hmc_step(q, nleap, step_size)

    
    def sample(self, q, p=None,
               nsamples=100, burnin=0, step_size=0.1, nleap=10,
               epsadapt=0, target_accept=0.65, jitter=True,
               callback=None, verbose=False):

        self.nsamples = nsamples
        self.burnin = burnin
        self.step_size = step_size
        self.nleap = nleap
        self.verbose = verbose

        state = Sampler()
        state.steplist = []
        
        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt, target_accept=target_accept) 

        for i in range(self.nsamples + self.burnin):
            if jitter:
                nleap = np.random.uniform(1, self.nleap)
            else:
                nleap = self.nleap
            q, p, acc, Hs, count = self.step(q, nleap=nleap, step_size=self.step_size) 
            state.i += 1
            if (i > self.burnin):
                state.accepts.append(acc)
                state.samples.append(q)
                state.Hs.append(Hs)
                state.counts.append(count)
                if callback is not None: callback(state)

        state.to_array()
        return state
    

