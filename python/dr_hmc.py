import numpy as np
from scipy.stats import expon, beta, multivariate_normal

from util import Sampler, inverse_Hessian_approx, DualAveragingStepSize


def get_stepsize_dist(eps, epsmax, min_fac=100):
    """return a beta distribution given the mean step-size, max step-size and min step size(factor).
    Mean of the distribution=eps. Mode of the distribution=eps/2
    """
    epsmin = epsmax/min_fac
    scale = epsmax*0.9
    eps_scaled = eps/epsmax
    b = (2-eps_scaled) * (1-eps_scaled)/eps_scaled
    a = 2-eps_scaled
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


    def adapt_stepsize(self, q, epsadapt):
        print("Adapting step size for %d iterations"%epsadapt)
        step_size = self.step_size
        epsadapt_kernel = DualAveragingStepSize(step_size)

        for i in range(epsadapt+1):
            q, p, acc, Hs, count = self.step(q, self.nleap, step_size)
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
        

    def leapfrog(self, q, p, N, step_size, M=None):
        self.leapcount += 1
        
        KE, KE_g = self.setup_KE(M)    
        qvec, gvec = [], []
        q0, p0 = q, p
        
        try:
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
            print("exception : ", e)
            return q0, p0, qvec, gvec


    def metropolis(self, qp0, qp1, M=None):

        q0, p0 = qp0
        q1, p1 = qp1
        H0 = self.H(q0, p0, M)
        H1 = self.H(q1, p1, M)
        log_prob = H0 - H1

        u =  np.random.uniform(0., 1., size=1)
        if np.isnan(log_prob) or np.isinf(log_prob) or (q0-q1).sum()==0:
            return q0, p0, -1, [H0, H1]
        elif  np.log(u) > min(0., log_prob):
            return q0, p0, 0., [H0, H1]
        else: return q1, p1, 1., [H0, H1]


    def delayed_step(self, q0, p0, qvec, gvec, nleap, step_size, prob_accept1):
        
        verbose = self.verbose
        if verbose: print(f"trying delayed step")

        try:
            # Estimate the Hessian given the rejected trajectory, use it to estimate step-size
            invh_est = inverse_Hessian_approx(np.array(qvec[::-1]), np.array(gvec[::-1]))
            h_est = np.linalg.inv(invh_est)
            eigv = np.linalg.eigvals(h_est)
            eps = min(step_size/2, np.sqrt(1/ eigv.max().real))
            epsf1 = get_stepsize_dist(eps, step_size)
            step_size2 = epsf1.rvs(size=1)
            
            # Make the second proposal
            q1, p1, _, _ = self.leapfrog(q0, p0, int(nleap*step_size/step_size2), step_size2)
            H0 = self.H(q0, p0)
            H1 = self.H(q1, p1)
            vanilla_prob = np.exp(H0 - H1)

            # Ghost trajectory for the second proposal
            q1_ghost, p1_ghost, qvec2, gvec2 = self.leapfrog(q1, -p1, nleap, step_size)
            H0 = self.H(q1, p1)
            H1 = self.H(q1_ghost, p1_ghost)
            prob_accept2 = min(1., np.exp(H0 - H1))
            if verbose: print("first and ghost accept probs: ", prob_accept1, prob_accept2)

            # Estimate Hessian and step-size distribution for ghost trajectory
            invh_est2 = inverse_Hessian_approx(np.array(qvec2[::-1]), np.array(gvec2[::-1]))
            h_est2 = np.linalg.inv(invh_est2)
            eigv2 = np.linalg.eigvals(h_est2)
            eps2 = min(step_size/2, np.sqrt(1/ eigv2.max().real))
            epsf2 = get_stepsize_dist(eps, step_size)
            if verbose: print(step_size, step_size2, eps, eps2)

            # Calcualte different Hastings corrections
            prob_delayed = (1-prob_accept2) / (1- prob_accept1)
            prob_eps = np.exp(epsf2.logpdf(step_size2) - epsf1.logpdf(step_size2) )        
            prob = vanilla_prob* prob_eps * prob_delayed
            if verbose: print(f"prob : {prob}\nvanilla_prob : {vanilla_prob}\nprob_eps : {prob_eps}\nprob_delayed : {prob_delayed}")

            u =  np.random.uniform(0., 1., size=1)
            if np.isnan(prob) or np.isinf(prob) or (q0-q1).sum()==0:
                if verbose: print("reject")
                return q0, p0, -1, [H0, H1]
            elif  u > min(1., prob):
                if verbose: print("reject")
                return q0, p0, 0., [H0, H1]
            else: 
                if verbose: print("accept")
                return q1, p1, 2., [H0, H1]
            
        except Exception as e:
            print("exception : ", e)
            return q0, p0, -1, [H0, H1]
            
        
    def step(self, q):

        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
        KE = self.setup_KE(self.mass_matrix)
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
        q1, p1, qvec, gvec = self.leapfrog(q, p, self.nleap, self.step_size)
        qf, pf, accepted, Hs = self.metropolis([q, p], [q1, p1])
        prob_accept1 = min(1., np.exp(Hs[0] - Hs[1]))
        
        if (accepted == 0 ) & self.delayed_proposals:
            qf, pf, accepted, Hs= self.delayed_step(q, p, qvec, gvec, self.nleap, self.step_size, prob_accept1=prob_accept1)
        return qf, pf, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount]


    def sample(self, q, p=None, callback=None, epsadapt=0, nsamples=100, delayed_proposals=True, burnin=0, step_size=0.1, nleap=10, verbose=False):

        self.nsamples = nsamples
        self.burnin = burnin
        self.step_size = step_size
        self.nleap = nleap
        self.delayed_proposals = delayed_proposals
        self.verbose = verbose

        #self._parse_kwargs_sample(**kwargs)

        state = Sampler()

        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt) 

        for i in range(self.nsamples + self.burnin):
            q, p, acc, Hs, count = self.step(q) 
            state.i += 1
            state.accepts.append(acc)
            if (i > self.burnin):
                state.samples.append(q)
                state.Hs.append(Hs)
                state.counts.append(count)
                if callback is not None: callback(state)

        state.to_array()
        return state
