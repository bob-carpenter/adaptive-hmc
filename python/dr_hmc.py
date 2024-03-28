import sys
import numpy as np
from scipy.stats import expon, beta, multivariate_normal, uniform

from util import Sampler, inverse_Hessian_approx, Hessian_approx, DualAveragingStepSize, PrintException, power_iteration


def get_beta_dist(eps, epsmax, min_fac=500):
    """return a beta distribution given the mean step-size, max step-size and min step size(factor).
    Mean of the distribution=eps. Mode of the distribution=eps/2
    """
    epsmin = epsmax/min_fac
    scale = epsmax-epsmin
    eps_scaled = eps/epsmax
    b = 2 * (1-eps_scaled)**2/eps_scaled
    a = 2 * (1-eps_scaled)
    dist = beta(a=a, b=b, loc=epsmin, scale=scale)
    return dist



class DRHMC_AdaptiveStepsize():

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
            q, p, acc, Hs, count, steplist, mhfac = self.step(q, self.nleap, step_size, delayed=False)
            if (qprev == q).all():
                prob = 0 
            else:
                prob = np.exp(Hs[0] - Hs[1])
            if mhfac is not None:
                prob *= mhfac

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
            

    def get_stepsize_dist(self, q0, p0, qvec, gvec, step_size, n_lbfgs=10, attempts=10):

        est_hessian = True
        i = 0
        vb = False
        qs, gs = [], []
        while (est_hessian) & (i < attempts): # if we hit nans rightaway, reduce step size and try again
            if i:
                step_size /= 2.
                if self.verbose: print(f'{i}, halve stepsize', step_size)
                q1, p1, qvec, gvec = self.leapfrog(q0, p0, N=n_lbfgs + 1, step_size=step_size)
            qs, gs = [], []
            for ig, g in enumerate(gvec):  # sometimes nans might be at the end of the trajectory. Discard them
                if np.isnan(g).any():
                    pass
                else:
                    qs.append(qvec[ig])
                    gs.append(gvec[ig])
            if len(qs) < n_lbfgs:
                if self.verbose: print('nans in g')
                i+= 1
                continue
            h_est = None
            h_est, points_used = Hessian_approx(np.array(qs[::-1]), np.array(gs[::-1]), h_est)
            if (points_used < n_lbfgs) :
                if self.verbose: print('skipped too many')
                i += 1
                continue
            elif  np.isnan(h_est).any():
                if self.verbose: print("nans in H")
                i+=1
                continue
            else:
                est_hessian = False

        if est_hessian:
            print(f"step size reduced to {step_size} from {self.step_size}")
            print("Exceeded max attempts to estimate Hessian")
            raise
        eigv = power_iteration(h_est + np.eye(self.D)*1e-6)[0]
        if eigv < 0:
            print("negative eigenvalue : ", eigv)
            raise
        eps = min(0.5*step_size, 0.5*np.sqrt(1/ eigv))
        epsf = get_beta_dist(eps, step_size)
        # eps = 0.5*np.sqrt(1/ eigv)
        # if eps < step_size:
        #     epsf = get_beta_dist(eps, step_size)
        # else:
        #     #print("increase step size?")
        #     epsf = get_beta_dist(step_size, eps)
        return eps, epsf
        
        

    def delayed_step(self, q0, p0, qvec, gvec, nleap, step_size, log_prob_accept1):
        
        verbose = self.verbose
        if verbose: print(f"trying delayed step")
        H0, H1 = 0., 0.
        try:
            if q0[0] > 50: self.verbose = True
            else: self.verbose = False
            # Estimate the Hessian given the rejected trajectory, use it to estimate step-size
            eps1, epsf1 = self.get_stepsize_dist(q0, p0, qvec, gvec, step_size)
            step_size_new = epsf1.rvs(size=1)[0]
            
            # Make the second proposal
            if self.constant_trajectory:
                nleap_new = int(min(nleap*step_size/step_size_new, nleap*100))
            else:
                nleap_new = int(nleap)
            q1, p1, _, _ = self.leapfrog(q0, p0, nleap_new, step_size_new)
            vanilla_log_prob = self.accept_log_prob([q0, p0], [q1, p1])
            
            # Ghost trajectory for the second proposal
            q1_ghost, p1_ghost, qvec_ghost, gvec_ghost = self.leapfrog(q1, -p1, nleap, step_size)
            log_prob_accept2 = self.accept_log_prob([q1, -p1], [q1_ghost, p1_ghost])

            # Estimate Hessian and step-size distribution for ghost trajectory
            eps2, epsf2 = self.get_stepsize_dist(q1, -p1, qvec_ghost, gvec_ghost, step_size)
            steplist = [eps1, eps2, step_size_new]

            # Calcualte different Hastings corrections
            if log_prob_accept2 == 0:
                if verbose: print("first and ghost accept probs: ", log_prob_accept1, log_prob_accept2)
                return q0, p0, 0, [H0, H1], steplist
            else:            
                log_prob_delayed = np.log((1-np.exp(log_prob_accept2))) - np.log((1- np.exp(log_prob_accept1)))
            log_prob_eps = epsf2.logpdf(step_size_new) - epsf1.logpdf(step_size_new)
            log_prob = vanilla_log_prob + log_prob_eps + log_prob_delayed
            
            if verbose:
                print("first and ghost accept probs: ", log_prob_accept1, log_prob_accept2)
                print("original, new step size : ", step_size, step_size_new)
                print("max allowed step size: : ", eps1, eps2)
                print(f"vanilla_log_prob : {vanilla_log_prob}\nlog_prob_eps : {log_prob_eps}\nlog_prob_delayed : {log_prob_delayed}\nlog_prob : {log_prob}")
            
            u =  np.random.uniform(0., 1., size=1)
            if np.isnan(log_prob) or (q0-q1).sum()==0:
                if verbose: print("reject\n")
                return q0, p0, -1, [H0, H1], steplist
            elif  np.log(u) > min(0., log_prob):
                if verbose: print("reject\n")
                return q0, p0, 0, [H0, H1], steplist
            else: 
                if verbose: print("accept\n")
                return q1, p1, 2., [H0, H1], steplist
            
        except Exception as e:
            PrintException()
            print("exception : ", e)
            return q0, p0, -1, [0, 0], [0., 0., 0.]
            
        
    def step(self, q, nleap=None, step_size=None, delayed=None):

        if nleap is None: nleap = self.nleap
        if step_size is None: step_size = self.step_size
        if delayed is None: delayed = self.delayed_proposals
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
        KE = self.setup_KE(self.mass_matrix)
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
        q1, p1, qvec, gvec = self.leapfrog(q, p, N=nleap, step_size=step_size)
        qf, pf, accepted, Hs = self.metropolis([q, p], [q1, p1])
        log_prob_accept1 = (Hs[0] - Hs[1])
        if np.isnan(log_prob_accept1) or (q-q1).sum() == 0 :
            log_prob_accept1 = -np.inf
        log_prob_accept1 = min(0, log_prob_accept1)
                    
        if (accepted <= 0 ) & delayed:
            qf, pf, accepted, Hs, steplist = self.delayed_step(q, p, qvec, gvec, nleap=nleap, step_size=step_size, log_prob_accept1=log_prob_accept1)
        else:
            steplist = [0, 0, step_size]
        return qf, pf, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount], steplist, None


    def sample(self, q, p=None,
               nsamples=100, burnin=0, step_size=0.1, nleap=10,
               delayed_proposals=True, constant_trajectory=False,
               epsadapt=0, target_accept=0.65,
               callback=None, verbose=False):

        self.nsamples = nsamples
        self.burnin = burnin
        self.step_size = step_size
        self.nleap = nleap
        self.delayed_proposals = delayed_proposals
        self.constant_trajectory = constant_trajectory
        self.verbose = verbose

        state = Sampler()
        state.steplist = []
        
        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt, target_accept=target_accept) 

        for i in range(self.nsamples + self.burnin):
            q, p, acc, Hs, count, steplist, mhfac = self.step(q) 
            state.i += 1
            if (i > self.burnin):
                state.accepts.append(acc)
                state.samples.append(q)
                state.Hs.append(Hs)
                state.counts.append(count)
                state.steplist.append(steplist)
                if callback is not None: callback(state)

        state.to_array()
        return state
    

    

class DRHMC_AdaptiveStepsize_autotune(DRHMC_AdaptiveStepsize):

    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None, min_nleap=10, max_nleap=1024):
        super(DRHMC_AdaptiveStepsize_autotune, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob, mass_matrix=mass_matrix)        
        self.min_nleap = min_nleap
        self.max_nleap = max_nleap

    def uturn(self, theta, rho, step_size): # Can be deleted as no longer used
            theta_next = theta
            rho_next = rho
            last_dist = 0
            N = 0
            H0 = self.H(theta, rho)
            while True:
                theta_next, rho_next, _, _ = self.leapfrog(theta_next, rho_next, 1, step_size)
                H1 = self.H(theta_next, rho_next)
                prob = np.exp(H0 - H1)
                if np.isnan(prob) or np.isinf(prob) or prob < 0.01:
                    return N, theta # THIS NEEDS TO BE CHANGED TO RETURN 0
                else:
                    dist = np.sum((theta_next - theta)**2)
                    if (dist <= last_dist) or (N > 1000):
                        theta_new = self.metropolis([theta, rho], [theta_next, rho_next])[0]
                        return N, theta_new
                    last_dist = dist
                    N += 1
                    

    def nuts_criterion(self, theta, rho, step_size):
            theta_next = theta
            rho_next = rho
            last_dist = 0
            N = 0
            H0 = self.H(theta, rho)
            while True:
                theta_next, rho_next, _, _ = self.leapfrog(theta_next, rho_next, 1, step_size)
                H1 = self.H(theta_next, rho_next)
                prob = np.exp(H0 - H1)
                Hs = [H0, H1]
                if np.isnan(prob) or np.isinf(prob) or prob < 0.01:
                    return N, theta, rho, Hs # SHOULD THIS RETURN 0?
                else:
                    if (np.dot((theta_next - theta), rho_next) > 0) and (np.dot((theta - theta_next), -rho) > 0) and (N < self.max_nleap) :
                        N += 1
                    else:
                        theta_new = self.metropolis([theta, rho], [theta_next, rho_next])[0]
                        return N, theta_new, rho_next, Hs


    
    def adapt_trajectory_length(self, q, n_adapt, target_accept):
        
        print("Adapting trajectory length for %d iterations"%n_adapt)
        self.traj_array = np.empty(n_adapt)
        nleaps, traj = [], []
        step_size = self.step_size
        epsadapt_kernel = DualAveragingStepSize(step_size, target_accept=target_accept)
        
        for i in range(n_adapt):
            p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
            qprev = q.copy()
            N, q, _, Hs = self.nuts_criterion(q, p, step_size)
            #add leapfrogs
            N = max(N, self.min_nleap)
            #self.nleap_array[i] = N
            self.traj_array[i] = N * step_size

            #update step size
            if (qprev == q).all():
                prob = 0 
            else:
                prob = np.exp(Hs[0] - Hs[1])
                
            if np.isnan(prob) or np.isinf(prob): 
                prob = 0.
                continue
            if prob > 1: prob = 1.
            step_size, avgstepsize = epsadapt_kernel.update(prob)

        # construct a distribution of leap frog steps
        self.step_size = step_size
        self.traj_array *= 1.
        return q


    def nleap_jitter(self, lowp=10, midp=30, highp=50):

        self.trajectory = np.percentile(self.traj_array, midp)
        print(f'base trajectory length = {self.trajectory}' )

        self.nleap_array = (self.traj_array / self.step_size).astype(int)
        self.nleap = max(self.min_nleap, int(self.trajectory / self.step_size))
        low = int(np.percentile(self.nleap_array, lowp))
        high = int(np.percentile(self.nleap_array, highp))
        print(f"Min and max number of leapfrog steps identified to be {low} and {high}")
        if low < self.min_nleap:
            low = self.min_nleap
            print(f"Raise min leapfrog steps to default min_nleap = {self.min_nleap}")
        if (high < low) or (high < self.min_nleap * 2):
            high = self.min_nleap * 2
            print(f"Raise min leapfrog steps to default 2 x min_nleap = {2*self.min_nleap}")
        self.nleap_dist = lambda x : np.random.randint(low=low, high=high)
            

    def sample(self, q, p=None,
               nsamples=100, burnin=0, step_size=0.1, nleap=10, delayed_proposals=True, 
               epsadapt=0, nleap_adapt=0, target_accept=0.65, constant_trajectory=True,
               callback=None, verbose=False):

        self.nsamples = nsamples
        self.burnin = burnin
        self.step_size = step_size
        self.nleap = nleap
        self.delayed_proposals = delayed_proposals
        self.verbose = verbose
        self.constant_trajectory = constant_trajectory
        self.nleap_dist = lambda x:  self.nleap

        state = Sampler()
        state.steplist = []
        
        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt//2, target_accept=target_accept) 
            
        if nleap_adapt:
            q = self.adapt_trajectory_length(q, nleap_adapt, target_accept)
            
        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt//2)
            
        # setup function for jittering step size
        if nleap_adapt:
            self.nleap_jitter()
        
        for i in range(self.nsamples + self.burnin):
            nleap = self.nleap_dist(1)
            q, p, acc, Hs, count, steplist, mhfac = self.step(q, nleap, self.step_size)
            state.i += 1
            if (i > self.burnin):
                state.accepts.append(acc)
                state.samples.append(q)
                state.Hs.append(Hs)
                state.counts.append(count)
                state.steplist.append(steplist)
                if callback is not None: callback(state)

        state.to_array()
        
        return state
    



##################################
class HMC_uturn(DRHMC_AdaptiveStepsize):

    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None, min_nleap=10, max_nleap=512):
        super(HMC_uturn, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob, mass_matrix=mass_matrix)        
        self.min_nleap = min_nleap
        self.max_nleap = max_nleap
                    

    def nuts_criterion(self, theta, rho, step_size, Noffset=0, theta_next=None, rho_next=None):
        if theta_next is None: theta_next = theta
        if rho_next is None: rho_next = rho
        N = Noffset
        qs, ps, gs = [], [], []
        # # check if given theta/rho already break the condition
        # if (np.dot((theta_next - theta), rho_next) < 0) or (np.dot((theta - theta_next), -rho) < 0) :
        #     print('returning at the beginning')
        #     return 0

        g_next = None
        while True:
            theta_next, rho_next, qvec, gvec = self.leapfrog(theta_next, rho_next, 1, step_size, g=g_next)
            g_next = gvec[-1]
            assert (theta_next == qvec[-1]).all()
            qs.append(theta_next)
            ps.append(rho_next)
            gs.append(g_next)
            if (np.dot((theta_next - theta), rho_next) > 0) and (np.dot((theta - theta_next), -rho) > 0) and (N < self.max_nleap) :
                N += 1
            else:
                N += 1
                return N, qs, ps, gs

    def step(self, q, nleap=None, step_size=None, delayed=None, offset=0.50):

        if nleap is None: nleap = self.nleap
        if step_size is None: step_size = self.step_size
        if delayed is None: delayed = self.delayed_proposals
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
        KE = self.setup_KE(self.mass_matrix)
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)

        Nuturn, qs, ps, gs = self.nuts_criterion(q, p, step_size)
        if Nuturn == 0:
            return q, p, -1, [0, 0], [self.Hcount, self.Vgcount, self.leapcount], [0, 0, 0], 0

        N0, N1 = int(offset*Nuturn), Nuturn
        nleap = self.rng.integers(N0, N1)
        #q1, p1, qvec, gvec = self.leapfrog(q, p, N=nleap+1, step_size=step_size)
        q1, p1, qvec, gvec = qs[nleap], ps[nleap], qs, gs
        
        Nuturn_rev, _, _, _ = self.nuts_criterion(q1, -p1, step_size)
        #Nuturn_rev, _, _, _ = self.nuts_criterion(q1, -p1, step_size, Noffset=nleap, theta_next=q, rho_next=-p)
        N0_rev, N1_rev = int(offset*Nuturn_rev), Nuturn_rev
        steplist = [Nuturn, Nuturn_rev, nleap]
        
        log_prob, H0, H1 = self.accept_log_prob([q, p], [q1, p1], return_H=True)
        lp1, lp2 =   -np.log(N1-N0), -np.log(N1_rev-N0_rev)
        if (nleap < N0_rev) or (nleap >= N1_rev): lp2 = -np.inf
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
               nsamples=100, burnin=0, step_size=0.1, nleap=10, delayed_proposals=True, 
               epsadapt=0, nleap_adapt=0, target_accept=0.65, constant_trajectory=True,
               callback=None, verbose=False, seed=99):

        self.nsamples = nsamples
        self.burnin = burnin
        self.step_size = step_size
        self.nleap = nleap
        self.delayed_proposals = delayed_proposals
        self.verbose = verbose
        self.constant_trajectory = constant_trajectory
        self.nleap_dist = lambda x:  self.nleap
        self.rng = np.random.default_rng(seed)
        
        state = Sampler()
        state.steplist = []
        
        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt, target_accept=target_accept) 
            
        
        for i in range(self.nsamples + self.burnin):
            q, p, acc, Hs, count, steplist, mhfac = self.step(q, nleap, self.step_size)
            #q, p, acc, Hs, count, steplist = self.step(q) 
            state.i += 1
            if (i > self.burnin):
                state.accepts.append(acc)
                state.samples.append(q)
                state.Hs.append(Hs)
                state.counts.append(count)
                state.steplist.append(steplist)
                if callback is not None: callback(state)

        state.to_array()
        
        return state
    



###################################################
class DRHMC_Adaptive(DRHMC_AdaptiveStepsize):

    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None, min_nleap=10, max_nleap=512):
        super(DRHMC_Adaptive, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob, mass_matrix=mass_matrix)        
        self.min_nleap = min_nleap
        self.max_nleap = max_nleap


    def nuts_criterion(self, theta, rho, step_size, Noffset=0, theta_next=None, rho_next=None):
        if theta_next is None: theta_next = theta
        if rho_next is None: rho_next = rho
        N = Noffset
        qs, ps, gs = [], [], []
        # check if given theta/rho already break the condition
        if (np.dot((theta_next - theta), rho_next) < 0) or (np.dot((theta - theta_next), -rho) < 0) :
            print('returning at the beginning')
            return 0, qs, ps, gs

        g_next = None
        while True:
            theta_next, rho_next, qvec, gvec = self.leapfrog(theta_next, rho_next, 1, step_size, g=g_next)
            g_next = gvec[-1]
            qs.append(theta_next)
            ps.append(rho_next)
            gs.append(g_next)
            if (np.dot((theta_next - theta), rho_next) >= 0) and (np.dot((theta - theta_next), -rho) >= 0) and (N < self.max_nleap) :
                N += 1
            else:
                return N, qs, ps, gs

            
    def delayed_step(self, q0, p0, qvec, gvec, Ns, step_size, log_prob_accept1, offset=0.5):
        
        verbose = self.verbose
        if verbose: print(f"trying delayed step")
        H0, H1 = 0., 0.
        nleap = 49 ###THIS NEEDS TO BE RE-THOUGHT
        #nleap = Ns[0]
        
        try:
            # Estimate the Hessian given the rejected trajectory, use it to estimate step-size
            #Npdf = uniform(offset*nuturn, (1-offset)*nuturn)
            #nleap = int(Npdf.rvs())  
            eps1, epsf1 = self.get_stepsize_dist(q0, p0, qvec, gvec, step_size)
            step_size_new = epsf1.rvs(size=1)[0]

            # Make the second proposal
            if self.constant_trajectory:
                nleap_new = int(min(nleap*step_size/step_size_new, nleap*100))
            else:
                nleap_new = int(nleap)
            q1, p1, _, _ = self.leapfrog(q0, p0, nleap_new, step_size_new)
            vanilla_log_prob = self.accept_log_prob([q0, p0], [q1, p1])
            
            # Ghost trajectory for the second proposal
            q1_ghost, p1_ghost, qvec_ghost, gvec_ghost, nleap_ghost, log_prob_accept2, Hs_ghost, Ns_ghost = self.first_step(q1, -p1, step_size)
            
            # Estimate Hessian and step-size distribution for ghost trajectory
            eps2, epsf2 = self.get_stepsize_dist(q1, -p1, qvec_ghost, gvec_ghost, step_size)
            steplist = [eps1, eps2, step_size_new]
            #Npdf2 = uniform(offset*Ns_ghost[0], (1-offset)*Ns_ghost[0])

            # Calcualte different Hastings corrections
            if log_prob_accept2 == 0:
                if verbose: print("first and ghost accept probs: ", log_prob_accept1, log_prob_accept2)
                #print("first and ghost accept probs: ", log_prob_accept1, log_prob_accept2, Ns, Ns_ghost, q0[0])
                return q0, p0, 0, [H0, H1], steplist
            else:            
                log_prob_delayed = np.log((1-np.exp(log_prob_accept2))) - np.log((1- np.exp(log_prob_accept1)))
            log_prob_eps = epsf2.logpdf(step_size_new) - epsf1.logpdf(step_size_new)
            log_prob_N = 0. #Npdf2.logpdf(nleap) - Npdf.logpdf(nleap)
            log_prob = vanilla_log_prob + log_prob_eps + log_prob_delayed + log_prob_N
            
            if verbose:
                print("first and ghost accept probs: ", log_prob_accept1, log_prob_accept2)
                print("original, new step size : ", step_size, step_size_new)
                print("max allowed step size: : ", eps1, eps2)
                print(f"vanilla_log_prob : {vanilla_log_prob}\nlog_prob_eps : {log_prob_eps}\nlog_prob_delayed : {log_prob_delayed}\nlog_prob : {log_prob}")
            
            u =  np.random.uniform(0., 1., size=1)
            if np.isnan(log_prob) or (q0-q1).sum()==0:
                if verbose: print("reject\n")
                return q0, p0, -1, [H0, H1], steplist
            elif  np.log(u) > min(0., log_prob):
                if verbose: print("reject\n")
                return q0, p0, 0, [H0, H1], steplist
            else: 
                if verbose: print("accept\n")
                return q1, p1, 2., [H0, H1], steplist
            
        except Exception as e:
            PrintException()
            print("exception : ", e)
            return q0, p0, -1, [0, 0], [0., 0., 0.]


        
    def first_step(self, q, p, step_size, offset=0.5):

        Nuturn, qs, ps, gs = self.nuts_criterion(q, p, step_size)

        Npdf = uniform(offset*Nuturn, (1-offset)*Nuturn)
        nleap = int(Npdf.rvs())
        if nleap == 0:
            #print(f'nleap is 0')
            if (qs[-1] - q).sum() == 0: print('there was no movement')
            return q, p, [], [], 0, -np.inf, [0, 0], [0, 0]
            #q1, p1, qvec, gvec = self.leapfrog(q, p, N=self.min_nleap, step_size=step_size)
        
        q1, p1, qvec, gvec = qs[nleap], ps[nleap], qs, gs
        
        Nuturn_rev, _, _, _ = self.nuts_criterion(q1, -p1, step_size, Noffset=nleap, theta_next=q, rho_next=-p)
        #Nuturn_rev, _, _, _ = self.nuts_criterion(q1, -p1, step_size)
        Npdf_rev = uniform(offset*Nuturn_rev, (1-offset)*Nuturn_rev)
        
        log_prob, H0, H1 = self.accept_log_prob([q, p], [q1, p1], return_H=True)

        # Hastings correction for leapfrog steps
        if nleap == 0 :
            log_prob_N = 0
        else:
            lp1, lp2 =   Npdf.logpdf(nleap), Npdf_rev.logpdf(nleap)
            log_prob_N = lp2 - lp1
        log_prob = log_prob + log_prob_N
        if np.isnan(log_prob) or (q-q1).sum()==0:
            log_prob = -np.inf
        return q1, p1, qvec, gvec, nleap, log_prob, [H0, H1], [Nuturn, Nuturn_rev]
    

        
    def step(self, q, nleap=None, step_size=None, delayed=None):

        if nleap is None: nleap = self.nleap
        if step_size is None: step_size = self.step_size
        if delayed is None: delayed = self.delayed_proposals
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
        KE = self.setup_KE(self.mass_matrix)
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
        
        q1, p1, qvec, gvec, nleap, log_prob_accept1, Hs, Ns = self.first_step(q, p, step_size)
        mhfac = np.exp(log_prob_accept1 - (Hs[0] - Hs[1]))
        
        log_prob_accept1 = min(0, log_prob_accept1)
        u =  np.random.uniform(0., 1., size=1)
        if  np.log(u) > min(0., log_prob_accept1):
            qf, pf = q, p
            accepted = 0
        else:
            qf, pf = q1, p1
            accepted = 1
                    
        if (accepted <= 0 ) & delayed:
            qf, pf, accepted, Hs, steplist = self.delayed_step(q, p, qvec, gvec, Ns=Ns, step_size=step_size, log_prob_accept1=log_prob_accept1)
        else:
            steplist = [0, 0, step_size]
        return qf, pf, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount], steplist, mhfac


    def sample(self, q, p=None,
               nsamples=100, burnin=0, step_size=0.1, nleap=10,
               delayed_proposals=True, constant_trajectory=False,
               epsadapt=0, nleap_adapt=0,
               target_accept=0.65,
               callback=None, verbose=False, seed=99):

        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.nsamples = nsamples
        self.burnin = burnin
        self.step_size = 0.1 #step_size
        self.nleap = nleap
        self.delayed_proposals = delayed_proposals
        self.constant_trajectory = constant_trajectory
        self.verbose = verbose
        
        state = Sampler()
        state.steplist = []
        
        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt, target_accept=target_accept) 

        for i in range(self.nsamples + self.burnin):
            q, p, acc, Hs, count, steplist, mhfac_n = self.step(q) 
            state.i += 1
            if (i > self.burnin):
                state.accepts.append(acc)
                state.samples.append(q)
                state.Hs.append(Hs)
                state.counts.append(count)
                state.steplist.append(steplist)
                if callback is not None: callback(state)

        state.to_array()
        return state
    

    
###################################################
class HMC_Adaptive(DRHMC_AdaptiveStepsize):

    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None, min_nleap=10, max_nleap=512):
        super(HMC_Adaptive, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob, mass_matrix=mass_matrix)        
        self.min_nleap = min_nleap
        self.max_nleap = max_nleap


    def nuts_criterion(self, theta, rho, step_size, Noffset=0, theta_next=None, rho_next=None):
        if theta_next is None: theta_next = theta
        if rho_next is None: rho_next = rho
        N = Noffset
        qs, ps, gs = [], [], []
        # check if given theta/rho already break the condition
        if (np.dot((theta_next - theta), rho_next) < 0) or (np.dot((theta - theta_next), -rho) < 0) :
            print('returning at the beginning')
            return 0

        g_next = None
        while True:
            theta_next, rho_next, qvec, gvec = self.leapfrog(theta_next, rho_next, 1, step_size, g=g_next)
            g_next = gvec[-1]
            qs.append(theta_next)
            ps.append(rho_next)
            gs.append(g_next)
            if (np.dot((theta_next - theta), rho_next) > 0) and (np.dot((theta - theta_next), -rho) > 0) and (N < self.max_nleap) :
                N += 1
            else:
                return N, qs, ps, gs

            
    def delayed_step(self, q0, p0, qvec, gvec, nuturn, step_size, log_prob_accept1, offset=0.5, skip_first=False):
        
        verbose = self.verbose
        if verbose: print(f"trying delayed step")
        H0, H1 = 0., 0.
        #nleap = 49 ###THIS NEEDS TO BE RE-THOUGHT
        try:
            # Estimate the Hessian given the rejected trajectory, use it to estimate step-size
            Npdf = uniform(offset*nuturn, (1-offset)*nuturn)
            nleap = int(Npdf.rvs())  
            eps1, epsf1 = self.get_stepsize_dist(q0, p0, qvec, gvec, step_size, nleap)
            step_size_new = epsf1.rvs(size=1)[0]

            # Make the second proposal
            if self.constant_trajectory:
                nleap_new = int(min(nleap*step_size/step_size_new, nleap*100))
            else:
                nleap_new = int(nleap)
            q1, p1, _, _ = self.leapfrog(q0, p0, nleap_new, step_size_new)
            vanilla_log_prob = self.accept_log_prob([q0, p0], [q1, p1])
            
            # Ghost trajectory for the second proposal
            #q1_ghost, p1_ghost, qvec_ghost, gvec_ghost = self.leapfrog(q1, -p1, nleap, step_size)
            #log_prob_accept2 = self.accept_log_prob([q1, -p1], [q1_ghost, p1_ghost])
            q1_ghost, p1_ghost, qvec_ghost, gvec_ghost, nleap_ghost, log_prob_accept2, Hs_ghost, Ns_ghost = self.first_step(q1, -p1, step_size)

            # Estimate Hessian and step-size distribution for ghost trajectory
            eps2, epsf2 = self.get_stepsize_dist(q1, -p1, qvec_ghost, gvec_ghost, step_size, nleap)
            steplist = [eps1, eps2, step_size_new]
            Npdf2 = uniform(offset*Ns_ghost[0], (1-offset)*Ns_ghost[0])
            if skip_first:
                log_prob_accept2 = -np.inf

            # Calcualte different Hastings corrections
            if log_prob_accept2 == 0:
                if verbose: print("first and ghost accept probs: ", log_prob_accept1, log_prob_accept2)
                return q0, p0, 0, [H0, H1], steplist
            else:            
                log_prob_delayed = np.log((1-np.exp(log_prob_accept2))) - np.log((1- np.exp(log_prob_accept1)))
            log_prob_eps = epsf2.logpdf(step_size_new) - epsf1.logpdf(step_size_new)
            log_prob_N = Npdf2.logpdf(nleap) - Npdf.logpdf(nleap)
            log_prob = vanilla_log_prob + log_prob_eps + log_prob_delayed + log_prob_N
            
            if verbose:
                print("first and ghost accept probs: ", log_prob_accept1, log_prob_accept2)
                print("original, new step size : ", step_size, step_size_new)
                print("max allowed step size: : ", eps1, eps2)
                print(f"vanilla_log_prob : {vanilla_log_prob}\nlog_prob_eps : {log_prob_eps}\nlog_prob_delayed : {log_prob_delayed}\nlog_prob : {log_prob}")
            
            u =  np.random.uniform(0., 1., size=1)
            if np.isnan(log_prob) or (q0-q1).sum()==0:
                if verbose: print("reject\n")
                return q0, p0, -1, [H0, H1], steplist
            elif  np.log(u) > min(0., log_prob):
                if verbose: print("reject\n")
                return q0, p0, 0, [H0, H1], steplist
            else: 
                if verbose: print("accept\n")
                return q1, p1, 2., [H0, H1], steplist
            
        except Exception as e:
            PrintException()
            print("exception : ", e)
            return q0, p0, -1, [0, 0], [0., 0., 0.]


        
    def first_step(self, q, p, step_size, offset=0.5):

        Nuturn, qs, ps, gs = self.nuts_criterion(q, p, step_size)
        if Nuturn == 0:
            #print("Nuturn is 0")
            return q, p, [], [], 0, -np.inf, [0, 0], [0, 0]

        Npdf = uniform(offset*Nuturn, (1-offset)*Nuturn)
        nleap = int(Npdf.rvs())  
        q1, p1, qvec, gvec = qs[nleap], ps[nleap], qs, gs
        
        Nuturn_rev, _, _, _ = self.nuts_criterion(q1, -p1, step_size, Noffset=nleap, theta_next=q, rho_next=-p)
        Npdf_rev = uniform(offset*Nuturn_rev, (1-offset)*Nuturn_rev)
        
        log_prob, H0, H1 = self.accept_log_prob([q, p], [q1, p1], return_H=True)
        # Hastings correction for leapfrog steps
        lp1, lp2 =   Npdf.logpdf(nleap), Npdf_rev.logpdf(nleap)
        log_prob_N = lp2 - lp1
        log_prob = log_prob + log_prob_N
        if np.isnan(log_prob) or (q-q1).sum()==0:
            log_prob = -np.inf
        return q1, p1, qvec, gvec, nleap, log_prob, [H0, H1], [Nuturn, Nuturn_rev]
    

        
    def step(self, q, nleap=None, step_size=None, delayed=None, skip_first=False):

        if nleap is None: nleap = self.nleap
        if step_size is None: step_size = self.step_size
        if delayed is None: delayed = self.delayed_proposals
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
        KE = self.setup_KE(self.mass_matrix)
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
        
        q1, p1, qvec, gvec, nleap, log_prob_accept1, Hs, Ns = self.first_step(q, p, step_size)
        mhfac = np.exp(log_prob_accept1 - (Hs[0] - Hs[1]))

        if ~skip_first:
            log_prob_accept1 = min(0, log_prob_accept1)
            u =  np.random.uniform(0., 1., size=1)
            if  np.log(u) > min(0., log_prob_accept1):
                qf, pf = q, p
                accepted = 0
            else:
                qf, pf = q1, p1
                accepted = 1
        else:
            accepted = 0
            log_prob_accept1 = -np.inf
                    
        if (accepted <= 0 ) & delayed:
            qf, pf, accepted, Hs, steplist = self.delayed_step(q, p, qvec, gvec, nuturn=Ns[0], step_size=step_size, log_prob_accept1=log_prob_accept1, skip_first=skip_first)
        else:
            steplist = [0, 0, step_size]
        return qf, pf, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount], steplist, mhfac


    def sample(self, q, p=None,
               nsamples=100, burnin=0, step_size=0.1, nleap=10,
               delayed_proposals=True, constant_trajectory=False,
               epsadapt=0, nleap_adapt=0,
               target_accept=0.65,
               callback=None, verbose=False):

        self.nsamples = nsamples
        self.burnin = burnin
        self.step_size = 0.1 #step_size
        self.nleap = nleap
        self.delayed_proposals = delayed_proposals
        self.constant_trajectory = constant_trajectory
        self.verbose = verbose
        
        state = Sampler()
        state.steplist = []
        
        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt, target_accept=target_accept) 

        for i in range(self.nsamples + self.burnin):
            q, p, acc, Hs, count, steplist, mhfac_n = self.step(q,  skip_first=True) 
            state.i += 1
            if (i > self.burnin):
                state.accepts.append(acc)
                state.samples.append(q)
                state.Hs.append(Hs)
                state.counts.append(count)
                state.steplist.append(steplist)
                if callback is not None: callback(state)

        state.to_array()
        return state
