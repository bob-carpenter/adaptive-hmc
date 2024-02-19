import sys
import numpy as np
from scipy.stats import expon, beta, multivariate_normal

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
            q, p, acc, Hs, count, _ = self.step(q, self.nleap, step_size, delayed=False)
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
            

    def get_stepsize_dist(self, q0, p0, qvec, gvec, step_size, nleap, n_lbfgs=10, attempts=5):

        est_hessian = True
        i = 0
        vb = False
        qs, gs = [], []
        while (est_hessian) & (i < attempts): # if we hit nans rightaway, reduce step size and try again
            if i:
                step_size /= 2.
                if self.verbose: print(f'{i}, halve stepsize', step_size)
                nleap = n_lbfgs + 1
                q1, p1, qvec, gvec = self.leapfrog(q0, p0, N=nleap, step_size=step_size)
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
            h_est, skipped_steps = Hessian_approx(np.array(qs[::-1]), np.array(gs[::-1]), h_est)
            if (nleap - skipped_steps < n_lbfgs) :
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
        return eps, epsf
        
        

    def delayed_step(self, q0, p0, qvec, gvec, nleap, step_size, log_prob_accept1):
        
        verbose = self.verbose
        if verbose: print(f"trying delayed step")
        H0, H1 = 0., 0.
        try:
            if q0[0] > 50: self.verbose = True
            else: self.verbose = False
            # Estimate the Hessian given the rejected trajectory, use it to estimate step-size
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
            q1_ghost, p1_ghost, qvec_ghost, gvec_ghost = self.leapfrog(q1, -p1, nleap, step_size)
            log_prob_accept2 = self.accept_log_prob([q1, -p1], [q1_ghost, p1_ghost])

            # Estimate Hessian and step-size distribution for ghost trajectory
            eps2, epsf2 = self.get_stepsize_dist(q1, -p1, qvec_ghost, gvec_ghost, step_size, nleap)
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
        return qf, pf, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount], steplist


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
            q, p, acc, Hs, count, steplist = self.step(q) 
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
    


###########################################################################
## Earlier version of the above code. Keeping around for now.
# class DRHMC_AdaptiveStepsize():

#     def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None):

#         self.D = D
#         self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
#         self.V = lambda x : self.log_prob(x)*-1.

#         if mass_matrix is None: self.mass_matrix = np.eye(D)
#         else: self.mass_matrix = mass_matrix
#         self.inv_mass_matrix = np.linalg.inv(self.mass_matrix)
        
#         self.leapcount = 0
#         self.Vgcount = 0
#         self.Hcount = 0


#     def adapt_stepsize(self, q, epsadapt, target_accept=0.65):
#         print("Adapting step size for %d iterations"%epsadapt)
#         step_size = self.step_size
#         epsadapt_kernel = DualAveragingStepSize(step_size, target_accept=target_accept)

#         for i in range(epsadapt+1):
#             qprev = q.copy()
#             q, p, acc, Hs, count, _ = self.step(q, self.nleap, step_size, delayed=False)
#             if (qprev == q).all():
#                 prob = 0 
#             else:
#                 prob = np.exp(Hs[0] - Hs[1])
                
#             if i < epsadapt:
#                 if np.isnan(prob) or np.isinf(prob): 
#                     prob = 0.
#                     continue
#                 if prob > 1: prob = 1.
#                 step_size, avgstepsize = epsadapt_kernel.update(prob)
#             elif i == epsadapt:
#                 _, step_size = epsadapt_kernel.update(prob)
#                 print("Step size fixed to : ", step_size)
#                 self.step_size = step_size
#         return q
        

#     def V_g(self, x):
#         self.Vgcount += 1
#         v_g = self.grad_log_prob(x)
#         return v_g *-1.

    
#     def H(self, q, p, M=None):
#         self.Hcount += 1
#         Vq = self.V(q)
#         KE, _ = self.setup_KE(M)
#         Kq = KE(p)
#         return Vq + Kq


#     def setup_KE(self, M):      # This is unnecessary if we are not adapting mass matrix. 
#         if M is None:
#             M = self.mass_matrix 
#         KE =  lambda p : 0.5*np.dot(p, np.dot(M, p))
#         KE_g =  lambda p : np.dot(M, p)
#         return KE, KE_g
        

#     def leapfrog(self, q, p, N, step_size, M=None):
#         self.leapcount += 1
        
#         KE, KE_g = self.setup_KE(M)    
#         qvec, gvec = [], []
#         q0, p0 = q, p
        
#         try:
#             g =  self.V_g(q)
#             p = p - 0.5*step_size * g
#             qvec.append(q)
#             gvec.append(g)
#             for i in range(N-1):
#                 q = q + step_size * KE_g(p)
#                 g = self.V_g(q)
#                 p = p - step_size * g
#                 qvec.append(q)
#                 gvec.append(g)
#             q = q + step_size * KE_g(p)
#             g = self.V_g(q)
#             p = p - 0.5*step_size * g
#             qvec.append(q)
#             gvec.append(g)            
#             return q, p, qvec, gvec

#         except Exception as e:  # Sometimes nans happen. 
#             #print("exception : ", e)
#             return q0, p0, qvec, gvec


#     def metropolis(self, qp0, qp1, M=None):

#         q0, p0 = qp0
#         q1, p1 = qp1
#         H0 = self.H(q0, p0)
#         H1 = self.H(q1, p1)
#         log_prob = H0 - H1

#         u =  np.random.uniform(0., 1., size=1)
#         if np.isnan(log_prob) or np.isinf(log_prob) or (q0-q1).sum()==0:
#             return q0, p0, -1, [H0, H1]
#         elif  np.log(u) > min(0., log_prob):
#             return q0, p0, 0., [H0, H1]
#         else: return q1, p1, 1., [H0, H1]


#     def delayed_step(self, q0, p0, qvec, gvec, nleap, step_size, log_prob_accept1):
        
#         verbose = self.verbose
#         if verbose: print(f"trying delayed step")

#         try:
#             # Estimate the Hessian given the rejected trajectory, use it to estimate step-size
#             h_est = np.eye(self.D) #/ step_size *np.linalg.norm(gvec[-1])  #initial value
#             h_est, skipped_steps = Hessian_approx(np.array(qvec[::-1]), np.array(gvec[::-1]), h_est)
#             if (nleap - skipped_steps < 5) or np.isnan(h_est).any():
#                 print("not enough points for correct h_est or nans in h_est")
#                 raise
#             #eigv = np.linalg.eigvals(h_est + np.eye(self.D)*1e-6)
#             eigv = power_iteration(h_est)[0]
#             eps = min(0.5*step_size, 0.5*np.sqrt(1/ eigv.max().real))
#             epsf1 = get_beta_dist(eps, step_size)
#             step_size2 = epsf1.rvs(size=1)[0]
            
#             # Make the second proposal
#             #nleap2 = int(min(nleap*step_size/step_size2, nleap*100))
#             nleap2 = int(nleap)
#             q1, p1, _, _ = self.leapfrog(q0, p0, nleap2, step_size2)
#             H0 = self.H(q0, p0)
#             H1 = self.H(q1, p1)
#             vanilla_log_prob = H0 - H1
#             if np.isnan(vanilla_log_prob)  or (q0-q1).sum()==0:
#                 vanilla_log_prob = -np.inf
#             vanilla_log_prob = min(0., vanilla_log_prob)
            
#             # Ghost trajectory for the second proposal
#             q1_ghost, p1_ghost, qvec2, gvec2 = self.leapfrog(q1, -p1, nleap, step_size)
#             H0 = self.H(q1, p1)
#             H1 = self.H(q1_ghost, p1_ghost)
#             log_prob_accept2 = (H0 - H1)
#             if np.isnan(log_prob_accept2)  or (q1_ghost-q1).sum()==0:
#                 log_prob_accept2 = -np.inf
#             log_prob_accept2 = min(0., log_prob_accept2)
#             if verbose: print("first and ghost accept probs: ", log_prob_accept1, log_prob_accept2)

#             # Estimate Hessian and step-size distribution for ghost trajectory
#             h_est2 =  np.eye(self.D) #/ step_size *np.linalg.norm(gvec2[-1])   #initial value
#             h_est2, skipped_steps = Hessian_approx(np.array(qvec2[::-1]), np.array(gvec2[::-1]), h_est2)
#             if (nleap - skipped_steps < 5) or np.isnan(h_est2).any():
#                 print("not enough points for correct h_est2 or nans in h_est2")
#                 raise
#             #eigv2 = np.linalg.eigvals(h_est2 + np.eye(self.D)*1e-6)
#             eigv2 = power_iteration(h_est2)[0]
#             eps2 = min(0.5*step_size, 0.5*np.sqrt(1/ eigv2.max().real))
#             epsf2 = get_beta_dist(eps2, step_size)
#             if verbose:
#                 print("original, new step size : ", step_size, step_size2)
#                 print("max allowed step size: : ", eps, eps2)

                
#             # Calcualte different Hastings corrections
#             if log_prob_accept2 == 0:
#                 return q0, p0, 0, [H0, H1], [eps, eps2, step_size2]
            
#             log_prob_delayed = np.log( (1-np.exp(log_prob_accept2)) / (1- np.exp(log_prob_accept1)) )
#             log_prob_eps = epsf2.logpdf(step_size2) - epsf1.logpdf(step_size2)
#             log_prob = vanilla_log_prob + log_prob_eps + log_prob_delayed
#             #if verbose: print(f"vanilla_prob : {vanilla_prob}\nprob_eps : {prob_eps}\nprob_delayed : {prob_delayed}\nprob : {prob}")
#             if verbose: print(f"vanilla_log_prob : {vanilla_log_prob}\nlog_prob_eps : {log_prob_eps}\nlog_prob_delayed : {log_prob_delayed}\nlog_prob : {log_prob}")
            
#             u =  np.random.uniform(0., 1., size=1)
#             if np.isnan(log_prob) or (q0-q1).sum()==0:
#                 if verbose: print("reject\n")
#                 return q0, p0, -1, [H0, H1], [eps, eps2, step_size2]
#             elif  np.log(u) > min(0., log_prob):
#                 if verbose: print("reject\n")
#                 return q0, p0, 0, [H0, H1], [eps, eps2, step_size2]
#             else: 
#                 if verbose: print("accept\n")
#                 return q1, p1, 2., [H0, H1], [eps, eps2, step_size2]
            
#         except Exception as e:
#             #PrintException()
#             #print("exception : ", e)
#             return q0, p0, -1, [0, 0], [0, 0, 0]
            
        
#     def step(self, q, nleap=None, step_size=None, delayed=None):

#         if nleap is None: nleap = self.nleap
#         if step_size is None: step_size = self.step_size
#         if delayed is None: delayed = self.delayed_proposals

#         self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
#         KE = self.setup_KE(self.mass_matrix)
#         p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
#         q1, p1, qvec, gvec = self.leapfrog(q, p, N=nleap, step_size=step_size)
#         qf, pf, accepted, Hs = self.metropolis([q, p], [q1, p1])
#         log_prob_accept1 = (Hs[0] - Hs[1])
#         if np.isnan(log_prob_accept1):
#             log_prob_accept1 = -np.inf
#         log_prob_accept1 = min(0, log_prob_accept1)
                    
#         if (accepted <= 0 ) & delayed:
#             qf, pf, accepted, Hs, steplist = self.delayed_step(q, p, qvec, gvec, nleap=nleap, step_size=step_size, log_prob_accept1=log_prob_accept1)
#         else:
#             steplist = [0., 0., 0.]
#         return qf, pf, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount], steplist


#     def sample(self, q, p=None,
#                nsamples=100, burnin=0, step_size=0.1, nleap=10, delayed_proposals=True, 
#                epsadapt=0, target_accept=0.65,
#                callback=None, verbose=False):

#         self.nsamples = nsamples
#         self.burnin = burnin
#         self.step_size = step_size
#         self.nleap = nleap
#         self.delayed_proposals = delayed_proposals
#         self.verbose = verbose

#         #self._parse_kwargs_sample(**kwargs)

#         state = Sampler()
#         state.steplist = []
        
#         if epsadapt:
#             q = self.adapt_stepsize(q, epsadapt, target_accept=target_accept) 

#         for i in range(self.nsamples + self.burnin):
#             q, p, acc, Hs, count, steplist = self.step(q) 
#             state.i += 1
#             if (i > self.burnin):
#                 state.accepts.append(acc)
#                 state.samples.append(q)
#                 state.Hs.append(Hs)
#                 state.counts.append(count)
#                 state.steplist.append(steplist)
#                 if callback is not None: callback(state)

#         state.to_array()
#         return state




    
###########################################################################
# class DRHMC_AdaptiveStepsize_autotune():

#     def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None):

#         self.D = D
#         self.log_prob, self.grad_log_prob = log_prob, grad_log_prob
#         self.V = lambda x : self.log_prob(x)*-1.

#         if mass_matrix is None: self.mass_matrix = np.eye(D)
#         else: self.mass_matrix = mass_matrix
#         self.inv_mass_matrix = np.linalg.inv(self.mass_matrix)
        
#         self.leapcount = 0
#         self.Vgcount = 0
#         self.Hcount = 0



#     def uturn(self, theta, rho, step_size):
#             theta_next = theta
#             rho_next = rho
#             last_dist = 0
#             N = 0
#             H0 = self.H(theta, rho)
#             while True:
#                 theta_next, rho_next, _, _ = self.leapfrog(theta_next, rho_next, 1, step_size)
#                 H1 = self.H(theta_next, rho_next)
#                 prob = np.exp(H0 - H1)
#                 if np.isnan(prob) or np.isinf(prob) or prob < 0.01:
#                     return 0, theta
#                 else:
#                     dist = np.sum((theta_next - theta)**2)
#                     if (dist <= last_dist) or (N > 500):
#                         theta_new = self.metropolis([theta, rho], [theta_next, rho_next])[0]
#                         return N, theta_new
#                     last_dist = dist
#                     N += 1
                
#     def adapt_trajectory_length(self, q, n_adapt):
#         print("Adapting trajectory length for %d iterations"%n_adapt)
#         nleaps = []
#         step_size = self.step_size
#         for i in range(n_adapt):
#             p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
#             #p = np.dot(self.inv_mass_matrix_L, np.random.normal(size=q.size).reshape(q.shape))    
#             N, q = self.uturn(q, p, step_size)
#             if N > 0: nleaps.append(N)
            
#         self.nleap_array = np.array(nleaps)
#         print(self.nleap_array)
#         self.nleap = int(np.percentile(self.nleap_array, 25))
#         self.traj_length = step_size * self.nleap
#         self.nleap_dist = lambda x : np.random.randint(low=int(np.percentile(self.nleap_array, 15)), 
#                                                        high=int(np.percentile(self.nleap_array, 40)))
#         print("Number of steps fixed to : ", self.nleap)
#         return q

    
#     def adapt_stepsize(self, q, epsadapt, target_accept=0.65):
#         print("Adapting step size for %d iterations"%epsadapt)
#         step_size = self.step_size
#         epsadapt_kernel = DualAveragingStepSize(step_size, target_accept=target_accept)

#         for i in range(epsadapt+1):
#             qprev = q.copy()
#             nleap = self.nleap_dist(1)
#             q, p, acc, Hs, count = self.step(q, nleap, step_size)
#             if (qprev == q).all():
#                 prob = 0 
#             else:
#                 prob = np.exp(Hs[0] - Hs[1])
                
#             if i < epsadapt:
#                 if np.isnan(prob) or np.isinf(prob): 
#                     prob = 0.
#                     continue
#                 if prob > 1: prob = 1.
#                 step_size, avgstepsize = epsadapt_kernel.update(prob)
#             elif i == epsadapt:
#                 _, step_size = epsadapt_kernel.update(prob)
#                 print("Step size fixed to : ", step_size)
#                 self.step_size = step_size
#         return q
        

#     def V_g(self, x):
#         self.Vgcount += 1
#         v_g = self.grad_log_prob(x)
#         return v_g *-1.

    
#     def H(self, q, p, M=None):
#         self.Hcount += 1
#         Vq = self.V(q)
#         KE, _ = self.setup_KE(M)
#         Kq = KE(p)
#         return Vq + Kq


#     def setup_KE(self, M):      # This is unnecessary if we are not adapting mass matrix. 
#         if M is None:
#             M = self.mass_matrix 
#         KE =  lambda p : 0.5*np.dot(p, np.dot(M, p))
#         KE_g =  lambda p : np.dot(M, p)
#         return KE, KE_g
        

#     def leapfrog(self, q, p, N, step_size, M=None):
#         self.leapcount += 1
        
#         KE, KE_g = self.setup_KE(M)    
#         qvec, gvec = [], []
#         q0, p0 = q, p
        
#         try:
#             g =  self.V_g(q)
#             p = p - 0.5*step_size * g
#             qvec.append(q)
#             gvec.append(g)
#             for i in range(N-1):
#                 q = q + step_size * KE_g(p)
#                 g = self.V_g(q)
#                 p = p - step_size * g
#                 qvec.append(q)
#                 gvec.append(g)
#             q = q + step_size * KE_g(p)
#             g = self.V_g(q)
#             p = p - 0.5*step_size * g
#             qvec.append(q)
#             gvec.append(g)            
#             return q, p, qvec, gvec

#         except Exception as e:  # Sometimes nans happen. 
#             #print("exception : ", e)
#             return q0, p0, qvec, gvec


#     def metropolis(self, qp0, qp1, M=None):

#         q0, p0 = qp0
#         q1, p1 = qp1
#         H0 = self.H(q0, p0)
#         H1 = self.H(q1, p1)
#         log_prob = H0 - H1

#         u =  np.random.uniform(0., 1., size=1)
#         if np.isnan(log_prob) or np.isinf(log_prob) or (q0-q1).sum()==0:
#             return q0, p0, -1, [H0, H1]
#         elif  np.log(u) > min(0., log_prob):
#             return q0, p0, 0., [H0, H1]
#         else: return q1, p1, 1., [H0, H1]


#     def delayed_step(self, q0, p0, qvec, gvec, nleap, step_size, prob_accept1):
        
#         verbose = self.verbose
#         if verbose: print(f"trying delayed step")

#         try:
#             # Estimate the Hessian given the rejected trajectory, use it to estimate step-size
#             invh_est, skipped_steps = inverse_Hessian_approx(np.array(qvec[::-1]), np.array(gvec[::-1]))
#             if nleap - skipped_steps < 5:
#                 print("not enough for correct Hessian")
#                 raise
#             h_est = np.linalg.inv(invh_est)
#             eigv = np.linalg.eigvals(h_est)
#             eps = min(0.5*step_size, 0.5*np.sqrt(1/ eigv.max().real))
#             if verbose: print(0.5*np.sqrt(1/ eigv.max().real))
#             epsf1 = get_stepsize_dist(eps, step_size)
#             step_size2 = epsf1.rvs(size=1)
            
#             # Make the second proposal
#             q1, p1, _, _ = self.leapfrog(q0, p0, int(nleap*step_size/step_size2), step_size2)
#             H0 = self.H(q0, p0)
#             H1 = self.H(q1, p1)
#             vanilla_prob = np.exp(H0 - H1)
#             if np.isnan(vanilla_prob) or np.isinf(vanilla_prob) or (q0-q1).sum()==0:
#                 vanilla_prob = 0.
            
#             # Ghost trajectory for the second proposal
#             q1_ghost, p1_ghost, qvec2, gvec2 = self.leapfrog(q1, -p1, nleap, step_size)
#             H0 = self.H(q1, p1)
#             H1 = self.H(q1_ghost, p1_ghost)
#             prob_accept2 = np.exp(H0 - H1)
#             if np.isnan(prob_accept2) or np.isinf(prob_accept2) or (q1_ghost-q1).sum()==0:
#                 prob_accept2 = 0.
#             else:
#                 prob_accept2 = min(1., prob_accept2)
#             if verbose: print("first and ghost accept probs: ", prob_accept1, prob_accept2)

#             # Estimate Hessian and step-size distribution for ghost trajectory
#             invh_est2, skipped_steps = inverse_Hessian_approx(np.array(qvec2[::-1]), np.array(gvec2[::-1]))
#             if nleap - skipped_steps < 5:
#                 print("not enough for correct Hessian")
#                 raise
#             h_est2 = np.linalg.inv(invh_est2)
#             eigv2 = np.linalg.eigvals(h_est2)
#             eps2 = min(0.5*step_size, 0.5*np.sqrt(1/ eigv2.max().real))
#             if verbose: print(0.5*np.sqrt(1/ eigv2.max().real))
#             epsf2 = get_stepsize_dist(eps2, step_size)
#             if verbose:
#                 print("original, new step size : ", step_size, step_size2)
#                 print("max allowed step size: : ", eps, eps2)

#             # Calcualte different Hastings corrections
#             prob_delayed = (1-prob_accept2) / (1- prob_accept1)
#             prob_eps = np.exp(epsf2.logpdf(step_size2) - epsf1.logpdf(step_size2) )        
#             prob = vanilla_prob* prob_eps * prob_delayed
#             if verbose: print(f"vanilla_prob : {vanilla_prob}\nprob_eps : {prob_eps}\nprob_delayed : {prob_delayed}\nprob : {prob}")
#             u =  np.random.uniform(0., 1., size=1)
#             if np.isnan(prob) or np.isinf(prob) or (q0-q1).sum()==0:
#                 if verbose: print("reject\n")
#                 return q0, p0, -1, [H0, H1]
#             elif  u > min(1., prob):
#                 if verbose: print("reject\n")
#                 return q0, p0, 0, [H0, H1]
#             else: 
#                 if verbose: print("accept\n")
#                 return q1, p1, 2., [H0, H1]
            
#         except Exception as e:
#             #print("exception : ", e)
#             return q0, p0, -1, [H0, H1]
            
        
#     def step(self, q, nleap=None, step_size=None):

#         if nleap is None: nleap = self.nleap
#         if step_size is None: step_size = self.step_size

#         self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
#         KE = self.setup_KE(self.mass_matrix)
#         p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
#         q1, p1, qvec, gvec = self.leapfrog(q, p, N=nleap, step_size=step_size)
#         qf, pf, accepted, Hs = self.metropolis([q, p], [q1, p1])
#         prob_accept1 = min(1., np.exp(Hs[0] - Hs[1]))
        
#         if (accepted == 0 ) & self.delayed_proposals:
#             qf, pf, accepted, Hs= self.delayed_step(q, p, qvec, gvec, nleap=nleap, step_size=step_size, prob_accept1=prob_accept1)
#         return qf, pf, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount]


#     def sample(self, q, p=None,
#                nsamples=100, burnin=0, step_size=0.1, nleap=10, delayed_proposals=True, 
#                epsadapt=0, nleap_adapt=0, target_accept=0.65,
#                callback=None, verbose=False):

#         self.nsamples = nsamples
#         self.burnin = burnin
#         self.step_size = step_size
#         self.nleap = nleap
#         self.delayed_proposals = delayed_proposals
#         self.verbose = verbose
#         self.nleap_dist = lambda x:  self.nleap

#         state = Sampler()
        
#         if epsadapt:
#             q = self.adapt_stepsize(q, epsadapt)
#             first_step_size = self.step_size * 1.
            
#         if nleap_adapt:
#             q = self.adapt_trajectory_length(q, nleap_adapt)
#             curr_nleap = self.nleap * 1.
#             q = self.adapt_stepsize(q, epsadapt)

#             # adjust N to maintain constant trajetory length
#             scale_nsteps = first_step_size / self.step_size
#             self.nleap = int(curr_nleap * scale_nsteps)
#             self.nleap_dist = lambda x : np.random.randint(low=int(np.percentile(self.nleap_array * scale_nsteps, 15)), 
#                                                         high=int(np.percentile(self.nleap_array  * scale_nsteps, 40)))
#             print("finally update number of steps again : ", self.nleap)


#         for i in range(self.nsamples + self.burnin):
#             nleap = self.nleap_dist(1)
#             q, p, acc, Hs, count = self.step(q, nleap, self.step_size)
#             state.i += 1
#             state.accepts.append(acc)
#             if (i > self.burnin):
#                 state.samples.append(q)
#                 state.Hs.append(Hs)
#                 state.counts.append(count)
#                 if callback is not None: callback(state)

#         state.to_array()
#         return state
    
