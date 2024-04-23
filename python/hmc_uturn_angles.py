import sys
import numpy as np
from scipy.stats import expon, beta, multivariate_normal, uniform, binom

from util import Sampler, DualAveragingStepSize, PrintException

from hmc import HMC

# Setup MPI environment
from mpi4py import MPI
comm = MPI.COMM_WORLD
wrank = comm.Get_rank()
wsize = comm.Get_size()



class HMC_Uturn_with_angles(HMC):

    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None, min_nleap=5, max_nleap=1024,
                 distribution='uniform', offset=0.5, p_binom=0.5,symmetric=True):
        super(HMC_Uturn, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob, mass_matrix=mass_matrix)        
        self.min_nleap = min_nleap
        self.max_nleap = max_nleap
        self.distribution = distribution
        self.offset = offset
        self.p_binom = p_binom
        self.symmetric = symmetric
        if not self.symmetric:
            print("Assymmetric U-turn condition")
    
    
    def nuts_criterion_with_angles(self, theta, rho, step_size, Noffset=0, theta_next=None, rho_next=None, check_goodness=True):
        if theta_next is None: theta_next = theta
        if rho_next is None: rho_next = rho
        N = Noffset
        qs, ps, gs = [], [], []
        H0 = self.H(theta, rho)
        g_next = None
        angles = []
        while True:
            theta_next, rho_next, qvec, gvec = self.leapfrog(theta_next, rho_next, 1, step_size, g=g_next)
            g_next = gvec[-1]
            if check_goodness:
                H1 = self.H(theta_next, rho_next)
                log_prob = H0 - H1
                if np.isnan(log_prob) : # or np.isinf(prob) or (prob < 0.001):
                    return N, qs, ps, gs, angles, False
            qs.append(theta_next)
            ps.append(rho_next)
            gs.append(g_next)
            dnorm, pnorm = np.linalg.norm(rho_next), np.linalg.norm(theta_next - theta)
            angles.append([np.dot((theta_next - theta), rho_next)/dnorm/pnorm , np.dot((theta - theta_next), -rho)/dnorm/pnom])
            # condition
            if (np.dot((theta_next - theta), rho_next) > 0)  and (N < self.max_nleap) :
                if self.symmetric:
                    if not (np.dot((theta - theta_next), -rho) > 0):
                        return N, qs, ps, gs, angles, True
                    else:
                        N+=1
                else: N+=1
            else:
                return N, qs, ps, gs, angles, True
            
                
    def nuts_criterion(self, theta, rho, step_size, Noffset=0, theta_next=None, rho_next=None, check_goodness=True):
        if theta_next is None: theta_next = theta
        if rho_next is None: rho_next = rho
        N = Noffset
        qs, ps, gs = [], [], []
        H0 = self.H(theta, rho)
        g_next = None
        # check if given theta/rho already break the condition
        # if (np.dot((theta_next - theta), rho_next) < 0) or (np.dot((theta - theta_next), -rho) < 0) :
        #     print('returning at the beginning')
        #     return 0
        while True:
            theta_next, rho_next, qvec, gvec = self.leapfrog(theta_next, rho_next, 1, step_size, g=g_next)
            g_next = gvec[-1]
            if check_goodness:
                H1 = self.H(theta_next, rho_next)
                log_prob = H0 - H1
                if np.isnan(log_prob) : # or np.isinf(prob) or (prob < 0.001):
                    return N, qs, ps, gs, False
            qs.append(theta_next)
            ps.append(rho_next)
            gs.append(g_next)
            # condition
            if (np.dot((theta_next - theta), rho_next) > 0)  and (N < self.max_nleap) :
                if self.symmetric:
                    if not (np.dot((theta - theta_next), -rho) > 0):
                        return N, qs, ps, gs, True
                    else:
                        N+=1
                else: N+=1
            else:
                return N, qs, ps, gs, True
            
                    
            
    def nleap_dist(self, N, nleap=None):
        
        if self.distribution == 'uniform':
            N0, N1 = int(self.offset*N), N
            if nleap is None:
                nleap = self.rng.integers(N0, N1)
            lp =  -np.log(N1-N0)
            if (N1 - N0) == 0:
                lp = 0.
            if (nleap < N0) or (nleap > N1) :
                lp = -np.inf
                
        if self.distribution == 'binomial':
            if nleap is None:
                nleap = self.rng.binomial(N-1, self.p_binom) # inlcusive N-1
            lp = binom.logpmf(nleap, N-1, self.p_binom)
            if nleap > N-1:
                lp = -np.inf
                
        return nleap, lp

    
    def step(self, q, step_size=None):

        if step_size is None: step_size = self.step_size
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)

        # Go forward
        Nuturn, qs, ps, gs, success = self.nuts_criterion(q, p, step_size)
        if Nuturn == 0:
            return q, p, -1, [0, 0], [self.Hcount, self.Vgcount, self.leapcount], [0, 0, 0], 0

        nleap, lp1 = self.nleap_dist(Nuturn)
        #q1, p1, qvec, gvec = self.leapfrog(q, p, N=nleap+1, step_size=step_size)
        q1, p1, qvec, gvec = qs[nleap], ps[nleap], qs, gs
        
        # Go backward
        Nuturn_rev, _, _, _, _ = self.nuts_criterion(q1, -p1, step_size)
        #Nuturn_rev, _, _, _ = self.nuts_criterion(q1, -p1, step_size, Noffset=nleap, theta_next=q, rho_next=-p)
        
        #N0_rev, N1_rev = int(offset*Nuturn_rev), Nuturn_rev
        nleap2, lp2 = self.nleap_dist(Nuturn_rev, nleap=nleap)
        assert nleap2 == nleap
        steplist = [Nuturn, Nuturn_rev, nleap]
        
        # accept/reject
        log_prob, H0, H1 = self.accept_log_prob([q, p], [q1, p1], return_H=True)
        log_prob_N = lp2 - lp1
        mhfac = np.exp(log_prob_N)
        log_prob = log_prob + log_prob_N 
        if np.isnan(log_prob) or (q-q1).sum()==0:
            return q, p, -1, [H0, H1], [self.Hcount, self.Vgcount, self.leapcount], steplist, 0
        else:
            u =  np.random.uniform(0., 1., size=1)
            if  np.log(u) > min(0., log_prob):
                qf, pf = q, p
                accepted = 0
                Hs = [H0, H1]
            else:
                qf, pf = q1, p1
                accepted = 1
                Hs = [H0, H1]
                    
        return qf, pf, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount], steplist, mhfac

    
    def sample(self, q, p=None,
               nsamples=100, burnin=0, step_size=0.1, nleap=None, 
               epsadapt=0, target_accept=0.65, offset=0.5,
               callback=None, verbose=False, seed=99):

        self.nsamples = nsamples
        self.burnin = burnin
        self.step_size = step_size
        self.nleap = None
        if nleap is not None:
            print("Nleap argument is ignored in U-turn sampler")
        self.verbose = verbose
        self.rng = np.random.default_rng(seed)
        
        state = Sampler()
        state.stepcount = []
        state.mhfac = []
        
        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt, target_accept=target_accept) 

        for i in range(self.nsamples + self.burnin):
            state.i += 1
            q, p, acc, Hs, count, stepcount, mhfac = self.step(q, self.step_size)
            if (i > self.burnin):
                state.accepts.append(acc)
                state.samples.append(q)
                state.Hs.append(Hs)
                state.counts.append(count)
                state.stepcount.append(stepcount)
                state.mhfac.append(mhfac)
                if callback is not None: callback(state)

        state.to_array()
        print(f"In rank {wrank}, exceeded max steps : {self.exceed_max_steps}, i.e {self.exceed_max_steps/len(state.accepts)} times")
        return state
    


