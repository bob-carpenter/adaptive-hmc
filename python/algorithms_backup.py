import sys
import numpy as np
from scipy.stats import expon, beta, multivariate_normal, uniform, binom

from util import Sampler, inverse_Hessian_approx, Hessian_approx, DualAveragingStepSize, PrintException, power_iteration, get_beta_dist

from hmc import HMC

# Setup MPI environment
from mpi4py import MPI
comm = MPI.COMM_WORLD
wrank = comm.Get_rank()
wsize = comm.Get_size()



class HMC_Uturn(HMC):

    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None, min_nleap=4, max_nleap=1024,
                 distribution='uniform', offset=0.5, p_binom=0.5,symmetric=True, mode='angles', **kwargs):
        super(HMC_Uturn, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob, mass_matrix=mass_matrix)        
        self.min_nleap = min_nleap
        self.max_nleap = max_nleap
        self.distribution = distribution
        self.offset = offset
        self.p_binom = p_binom
        self.symmetric = symmetric
        self.mode = mode
        self.exceed_max_steps = 0
        # parse kwargs
        for key, val in kwargs.items():
            setattr(self, key, val)

        if not self.symmetric:
            print("Assymmetric U-turn condition")
            print("Not implemented")
            raise
                    
    def nuts_criterion(self, theta, rho, step_size, Noffset=0, theta_next=None, rho_next=None, check_goodness=True):
        if theta_next is None: theta_next = theta
        if rho_next is None: rho_next = rho
        N = Noffset
        qs, ps, gs = [], [], []
        H0 = self.H(theta, rho)
        g_next = None
        old_distance = 0
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
                if np.isnan(log_prob) or np.isinf(log_prob) or (log_prob < -10):
                    return N, qs, ps, gs, False
            qs.append(theta_next)
            ps.append(rho_next)
            gs.append(g_next)
            # condition
            if self.mode == 'angles':
                # if (np.dot((theta_next - theta), rho_next) > 0)  and (N < self.max_nleap) :
                #     if self.symmetric & (not (np.dot((theta - theta_next), -rho) > 0)): return N, qs, ps, gs, True
                #     else:  N += 1
                # else: return N, qs, ps, gs, True
                if (np.dot((theta_next - theta), rho_next) > 0)  and (np.dot((theta - theta_next), -rho) > 0) and (N < self.max_nleap) :
                    N += 1
                else:
                    if (N >= self.max_nleap):
                        #print("Exceeded max steps")
                        self.exceed_max_steps += 1
                    return N, qs, ps, gs, True

            elif self.mode == 'distance':
                distance = np.sum((theta_next - theta)**2)
                if distance <= old_distance or (N >= self.max_nleap):
                    if (N >= self.max_nleap):
                        #print("Exceeded max steps")
                        self.exceed_max_steps += 1
                    return N, qs, ps, gs, True
                else:
                    old_distance = distance
                    N += 1
            else:
                print("Mode for u-turn criterion should be angles or distance")
                sys.exit()
            
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
        # if self.distribution == 'binomial':
        #     if N == 1:
        #         if nleap is None: nleap = 1
        #         if nleap == 1: return int(nleap), 0.
        #         else: return int(nleap), -np.inf
        #     else:
        #         if nleap is None:
        #             nleap = 1 + self.rng.binomial(N-2, self.p_binom) # inlcusive N-1
        #         lp = binom.logpmf(nleap-1, N-2, self.p_binom)
        #         if nleap-1 > N-2:
        #             lp = -np.inf
                
        return int(nleap), lp

    
    def step(self, q, step_size=None):

        if step_size is None: step_size = self.step_size
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)

        # Go forward
        Nuturn, qs, ps, gs, success = self.nuts_criterion(q, p, step_size)
        if (Nuturn == 0) or (not success):
            return q, p, -1, [0, 0], [self.Hcount, self.Vgcount, self.leapcount], [0, 0, 0], 0

        nleap, lp1 = self.nleap_dist(Nuturn)
        q1, p1, qvec, gvec = qs[nleap], ps[nleap], qs, gs
        
        # Go backward
        Nuturn_rev, _, _, _, _ = self.nuts_criterion(q1, -p1, step_size)
        
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
        niterations = self.nsamples + self.burnin
        
        state = Sampler()
        state.stepcount = []
        state.mhfac = []
        
        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt, target_accept=target_accept)
            
        for i in range(self.nsamples + self.burnin):
            q, p, acc, Hs, count, stepcount, mhfac = self.step(q, self.step_size)
            state.i += 1
            if (i%(self.nsamples//10) == 0):
                print(f"In rank {wrank}: iteration {i} of {niterations}")
            if (i > self.burnin):
                state.accepts.append(acc)
                state.samples.append(q)
                state.Hs.append(Hs)
                state.counts.append(count)
                state.stepcount.append(stepcount)
                state.mhfac.append(mhfac)
                if callback is not None: callback(state)

        state.to_array()
        
        return state



###################################################
class HMC_Uturn_Jitter(HMC_Uturn):
    """adaptive algorithm which makes delayed proposal with prob (1-pcheck), otherwise runs u-turn sampler.
    pcheck is evaluated at the point of u-turn. The number of steps in independent of "u-turn" and sampled from pre-stored u-turn trajectories.
    """
    
    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None,
                 min_nleap=5, max_nleap=1024,
                 distribution='uniform', offset=0.5, p_binom=0.5,symmetric=True,
                 **kwargs):
        super(HMC_Uturn_Jitter, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob, mass_matrix=mass_matrix,
                                               min_nleap=min_nleap, max_nleap=max_nleap,
                                               distribution=distribution, offset=offset, p_binom=p_binom,symmetric=symmetric, **kwargs)
            
        print(self.min_nleap, self.max_nleap)


    def adapt_trajectory_length(self, q, n_adapt, target_accept, check_goodness=True):
        
        print("Adapting trajectory length for %d iterations"%n_adapt)
        self.traj_array = [] #np.zeros(n_adapt)
        nleaps, traj = [], []
        step_size = self.step_size
        for i in range(n_adapt):            
            p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
            qprev = q.copy()
            H0 = self.H(q, p)
            N, qs, ps, gs, success = self.nuts_criterion(q, p, step_size, check_goodness=check_goodness)
            if success:
                #self.traj_array[i] = N * step_size
                self.traj_array.append(N * step_size)
            else:
                self.traj_array.append(0.)

        # construct a distribution of leap frog steps
        self.traj_array = np.array(self.traj_array) * 1.
        return q

    
    def nleap_jitter(self, lowp=10, highp=90, adjustment_factor=1.):

        if not hasattr(self, "trajectories"):
            l, h = np.percentile(self.traj_array, lowp), np.percentile(self.traj_array, highp)
            if l == h:
                l -= 1
                h += 1
            trajectories = self.traj_array.copy()
            trajectories = trajectories[trajectories > l]
            trajectories = trajectories[trajectories < h]
            self.trajectories = trajectories * adjustment_factor # Adjustment factor can be 2/3 factor to not make a full U-turn
            print("average number of steps  : ", (self.trajectories/self.step_size).mean())
            if self.trajectories.size == 0 :
                print("error in trajectories")
                raise
        self.nleap_jitter_dist = lambda step_size : min(max(int(np.random.choice(self.trajectories, 1) / step_size), self.min_nleap), self.max_nleap)

    

    def sample(self, q, p=None,
               nsamples=100, burnin=0, step_size=0.1, nleap=10,
               constant_trajectory=False, adapt_delayed_trajectory=False,
               epsadapt=0, nleap_adapt=100,
               target_accept=0.65,
               callback=None, verbose=False, seed=99):

        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.nsamples = nsamples
        self.burnin = burnin
        self.step_size = step_size
        self.nleap = nleap
        self.verbose = verbose
        niterations = self.nsamples + self.burnin
        
        state = Sampler()
        state.steplist = []
        state.stepcount = []
        
        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt, target_accept=target_accept) 

        if nleap_adapt:
            print("adapting trajectory lengths")
            q = self.adapt_trajectory_length(q, nleap_adapt, target_accept)
            comm.Barrier()
            all_traj_array = np.zeros(len(self.traj_array) * wsize)
            all_traj_array_tmp = comm.gather(self.traj_array, root=0)
            if wrank == 0 :
                all_traj_array = np.concatenate(all_traj_array_tmp)
                print(f"Shape of trajectories in rank 0 : ", all_traj_array.shape)
                
            comm.Bcast(all_traj_array, root=0)
            self.traj_array = all_traj_array*1.
            self.traj_array = self.traj_array[ self.traj_array!=0]
            comm.Barrier()
            print(f"Shape of trajectories after bcast  in rank {wrank} : ", self.traj_array.shape)
            self.nleap_jitter()
            
            
        for i in range(niterations):
            nleap = self.nleap_jitter_dist(step_size)
            q, p, acc, Hs, count = self.step(q, nleap=nleap) 
            state.i += 1
            if (i%(self.nsamples//10) == 0):
                print(f"In rank {wrank}: iteration {i} of {niterations}")
            if (i > self.burnin):
                state.accepts.append(acc)
                state.samples.append(q)
                state.Hs.append(Hs)
                state.counts.append(count)
                if callback is not None: callback(state)

        state.to_array()
        return state


###############################################################
class DRHMC_AdaptiveStepsize(HMC_Uturn_Jitter):

    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None, n_lbfgs=10, attempts=10, **kwargs):
        super(DRHMC_AdaptiveStepsize, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob, mass_matrix=mass_matrix, **kwargs)        
        self.n_lbfgs=n_lbfgs
        self.attempts = attempts
        
    def get_stepsize_dist(self, q0, p0, qvec, gvec, step_size, ):

        est_hessian = True
        i = 0
        vb = False
        n_lbfgs, attempts = self.n_lbfgs, self.attempts
        qs, gs = [], []
        while (est_hessian) & (i < attempts): # if we hit nans rightaway, reduce step size and try again
            if i > 0:
                step_size /= 2.
                if self.verbose: print(f'{i}, halve stepsize', step_size)
                q1, p1, qvec, gvec = self.leapfrog(q0, p0, N=n_lbfgs + 1, step_size=step_size)
            qs, gs = [], []
            if len(qvec) < n_lbfgs:
                if self.verbose: print('empty qvec')
                i+= 1
                continue
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
                    
        if (accepted <= 0) & delayed:
            qf, pf, accepted, Hs, steplist = self.delayed_step(q, p, qvec, gvec, nleap=nleap, step_size=step_size, log_prob_accept1=log_prob_accept1)
        else:
            steplist = [0, 0, step_size]
        return qf, pf, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount], steplist, None


    def sample(self, q, p=None,
               nsamples=100, burnin=0, step_size=0.1, nleap=10,
               delayed_proposals=True, constant_trajectory=False,
               epsadapt=0, nleap_adapt=0, target_accept=0.65,
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
           
        if nleap_adapt:
            print("adapting trajectory lengths")
            q = self.adapt_trajectory_length(q, nleap_adapt, target_accept)
            comm.Barrier()
            all_traj_array = np.zeros(len(self.traj_array) * wsize)
            all_traj_array_tmp = comm.gather(self.traj_array, root=0)
            if wrank == 0 :
                all_traj_array = np.concatenate(all_traj_array_tmp)
                print(f"Shape of trajectories in rank 0 : ", all_traj_array.shape)
                
            comm.Bcast(all_traj_array, root=0)
            self.traj_array = all_traj_array*1.
            self.traj_array = self.traj_array[ self.traj_array!=0]
            comm.Barrier()
            print(f"Shape of trajectories after bcast  in rank {wrank} : ", self.traj_array.shape)
            self.nleap_jitter()
        else:
            self.nleap_jitter_dist = lambda x:  self.nleap
            
        for i in range(self.nsamples + self.burnin):
            nleap = self.nleap_jitter_dist(step_size)
            q, p, acc, Hs, count, steplist, mhfac = self.step(q, nleap=nleap) 
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





# ###############################################################
# class DRHMC_AdaptiveStepsize(HMC):

#     def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None, n_lbfgs=10, attempts=10):
#         super(DRHMC_AdaptiveStepsize, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob, mass_matrix=mass_matrix)        
#         self.n_lbfgs=n_lbfgs
#         self.attempts = attempts
        
#     def get_stepsize_dist(self, q0, p0, qvec, gvec, step_size, ):

#         est_hessian = True
#         i = 0
#         vb = False
#         n_lbfgs, attempts = self.n_lbfgs, self.attempts
#         qs, gs = [], []
#         while (est_hessian) & (i < attempts): # if we hit nans rightaway, reduce step size and try again
#             if i > 0:
#                 step_size /= 2.
#                 if self.verbose: print(f'{i}, halve stepsize', step_size)
#                 q1, p1, qvec, gvec = self.leapfrog(q0, p0, N=n_lbfgs + 1, step_size=step_size)
#             qs, gs = [], []
#             if len(qvec) < n_lbfgs:
#                 if self.verbose: print('empty qvec')
#                 i+= 1
#                 continue
#             for ig, g in enumerate(gvec):  # sometimes nans might be at the end of the trajectory. Discard them
#                 if np.isnan(g).any():
#                     pass
#                 else:
#                     qs.append(qvec[ig])
#                     gs.append(gvec[ig])
#             if len(qs) < n_lbfgs:
#                 if self.verbose: print('nans in g')
#                 i+= 1
#                 continue
#             h_est = None
#             h_est, points_used = Hessian_approx(np.array(qs[::-1]), np.array(gs[::-1]), h_est)
#             if (points_used < n_lbfgs) :
#                 if self.verbose: print('skipped too many')
#                 i += 1
#                 continue
#             elif  np.isnan(h_est).any():
#                 if self.verbose: print("nans in H")
#                 i+=1
#                 continue
#             else:
#                 est_hessian = False

#         if est_hessian:
#             print(f"step size reduced to {step_size} from {self.step_size}")
#             print("Exceeded max attempts to estimate Hessian")
#             raise
#         eigv = power_iteration(h_est + np.eye(self.D)*1e-6)[0]
#         if eigv < 0:
#             print("negative eigenvalue : ", eigv)
#             raise
#         eps = min(0.5*step_size, 0.5*np.sqrt(1/ eigv))
#         epsf = get_beta_dist(eps, step_size)
#         # eps = 0.5*np.sqrt(1/ eigv)
#         # if eps < step_size:
#         #     epsf = get_beta_dist(eps, step_size)
#         # else:
#         #     #print("increase step size?")
#         #     epsf = get_beta_dist(step_size, eps)
#         return eps, epsf
        
        

#     def delayed_step(self, q0, p0, qvec, gvec, nleap, step_size, log_prob_accept1):
        
#         verbose = self.verbose
#         if verbose: print(f"trying delayed step")
#         H0, H1 = 0., 0.
#         try:
#             if q0[0] > 50: self.verbose = True
#             else: self.verbose = False
#             # Estimate the Hessian given the rejected trajectory, use it to estimate step-size
#             eps1, epsf1 = self.get_stepsize_dist(q0, p0, qvec, gvec, step_size)
#             step_size_new = epsf1.rvs(size=1)[0]
            
#             # Make the second proposal
#             if self.constant_trajectory:
#                 nleap_new = int(min(nleap*step_size/step_size_new, nleap*100))
#             else:
#                 nleap_new = int(nleap)
#             q1, p1, _, _ = self.leapfrog(q0, p0, nleap_new, step_size_new)
#             vanilla_log_prob = self.accept_log_prob([q0, p0], [q1, p1])
            
#             # Ghost trajectory for the second proposal
#             q1_ghost, p1_ghost, qvec_ghost, gvec_ghost = self.leapfrog(q1, -p1, nleap, step_size)
#             log_prob_accept2 = self.accept_log_prob([q1, -p1], [q1_ghost, p1_ghost])

#             # Estimate Hessian and step-size distribution for ghost trajectory
#             eps2, epsf2 = self.get_stepsize_dist(q1, -p1, qvec_ghost, gvec_ghost, step_size)
#             steplist = [eps1, eps2, step_size_new]

#             # Calcualte different Hastings corrections
#             if log_prob_accept2 == 0:
#                 if verbose: print("first and ghost accept probs: ", log_prob_accept1, log_prob_accept2)
#                 return q0, p0, 0, [H0, H1], steplist
#             else:            
#                 log_prob_delayed = np.log((1-np.exp(log_prob_accept2))) - np.log((1- np.exp(log_prob_accept1)))
#             log_prob_eps = epsf2.logpdf(step_size_new) - epsf1.logpdf(step_size_new)
#             log_prob = vanilla_log_prob + log_prob_eps + log_prob_delayed
            
#             if verbose:
#                 print("first and ghost accept probs: ", log_prob_accept1, log_prob_accept2)
#                 print("original, new step size : ", step_size, step_size_new)
#                 print("max allowed step size: : ", eps1, eps2)
#                 print(f"vanilla_log_prob : {vanilla_log_prob}\nlog_prob_eps : {log_prob_eps}\nlog_prob_delayed : {log_prob_delayed}\nlog_prob : {log_prob}")
            
#             u =  np.random.uniform(0., 1., size=1)
#             if np.isnan(log_prob) or (q0-q1).sum()==0:
#                 if verbose: print("reject\n")
#                 return q0, p0, -1, [H0, H1], steplist
#             elif  np.log(u) > min(0., log_prob):
#                 if verbose: print("reject\n")
#                 return q0, p0, 0, [H0, H1], steplist
#             else: 
#                 if verbose: print("accept\n")
#                 return q1, p1, 2., [H0, H1], steplist
            
#         except Exception as e:
#             PrintException()
#             print("exception : ", e)
#             return q0, p0, -1, [0, 0], [0., 0., 0.]
            
        
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
#         if np.isnan(log_prob_accept1) or (q-q1).sum() == 0 :
#             log_prob_accept1 = -np.inf
#         log_prob_accept1 = min(0, log_prob_accept1)
                    
#         if (accepted <= 0) & delayed:
#             qf, pf, accepted, Hs, steplist = self.delayed_step(q, p, qvec, gvec, nleap=nleap, step_size=step_size, log_prob_accept1=log_prob_accept1)
#         else:
#             steplist = [0, 0, step_size]
#         return qf, pf, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount], steplist, None


#     def sample(self, q, p=None,
#                nsamples=100, burnin=0, step_size=0.1, nleap=10,
#                delayed_proposals=True, constant_trajectory=False,
#                epsadapt=0, target_accept=0.65,
#                callback=None, verbose=False):

#         self.nsamples = nsamples
#         self.burnin = burnin
#         self.step_size = step_size
#         self.nleap = nleap
#         self.delayed_proposals = delayed_proposals
#         self.constant_trajectory = constant_trajectory
#         self.verbose = verbose

#         state = Sampler()
#         state.steplist = []
        
#         if epsadapt:
#             q = self.adapt_stepsize(q, epsadapt, target_accept=target_accept) 

#         for i in range(self.nsamples + self.burnin):
#             q, p, acc, Hs, count, steplist, mhfac = self.step(q) 
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

    



###################################################
class HMC_Adaptive_Binary(DRHMC_AdaptiveStepsize):
    """adaptive algorithm which 
    --- runs to a u-turn and evaluates "pcheck" 
    --- delayed proposal with prob (1-pcheck), otherwise runs u-turn sampler.
    --- The trajectory length in the delayed stage are decided by "u-turn" to pcheck
    """
    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None,  min_nleap=5, max_nleap=1024,
                 distribution='uniform', offset=0.5, p_binom=0.5,symmetric=True):
        super(HMC_Adaptive_Binary, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob, mass_matrix=mass_matrix, min_nleap=min_nleap, max_nleap=max_nleap,
                                             distribution=distribution, offset=offset, p_binom=p_binom,symmetric=symmetric)
            

    def pre_step(self, q, p, step_size):
        # Go forward
        Nuturn, qvec, pvec, gvec, success = self.nuts_criterion(q, p, step_size)
       
        if (Nuturn == 0) or (not success):
            log_prob_check = -np.inf
            qvec, gvec = [], []
        else:
            qcheck, pcheck = qvec[-1], pvec[-1]
            log_prob_check, H0, Hcheck = self.accept_log_prob([q, p], [qcheck, pcheck], return_H=True)
            
        if self.CHECK_U_TURN: log_prob_check = 0. 
        if self.CHECK_DELAYED: log_prob_check = -np.inf
        #log_prob_check = log_prob_check - np.log(self.pfactor)
        if self.pfactor != 1:
            log_prob_check = self.pfactor * log_prob_check
        if self.pthresh !=0:
            if log_prob_check > np.log(self.pthresh):
                log_prob_check = 0.
            else:
                log_prob_check = -np.inf
            
        return Nuturn, qvec, pvec, gvec, log_prob_check

    
    def delayed_step(self, q0, p0, qvec, gvec, step_size, log_prob_delayed, Nuturn):
        ##CHECK DELAYED STEP IS FAILING FOR offset=0.5, uniform distribution
        ##
        verbose = self.verbose
        if verbose: print(f"trying delayed step")
        H0, H1 = 0., 0.
        try:
            # Estimate the Hessian given the rejected trajectory, use it to estimate step-size
            eps1, epsf1 = self.get_stepsize_dist(q0, p0, qvec, gvec, step_size)
            step_size_new = epsf1.rvs(size=1)[0]

            # Make the second proposal
            if self.constant_trajectory:
                nleap_new = int(min(self.nleap*step_size / step_size_new, self.nleap*100))
            else:
                if self.adapt_delayed_trajectory and (Nuturn > 0):
                    nleap_new, lp_N = self.nleap_dist(int(Nuturn * step_size / step_size_new))
                else: nleap_new, lp_N = int(self.nleap), 0.
                    
            q1, p1, _, _ = self.leapfrog(q0, p0, nleap_new, step_size_new)
            
            # Ghost trajectory for the second proposal
            Nuturn_ghost, qvec_ghost, pvec_ghost, gvec_ghost, log_prob_check_ghost = self.pre_step(q1, -p1, step_size)
            eps2, epsf2 = self.get_stepsize_dist(q1, -p1, qvec_ghost, gvec_ghost, step_size)
            log_prob_delayed_ghost = np.log(1 - np.exp(log_prob_check_ghost))
            if np.isnan(log_prob_delayed_ghost) : log_prob_delayed_ghost = -np.inf
            if self.CHECK_DELAYED: log_prob_delayed_ghost = 0.
            
            if self.adapt_delayed_trajectory and (Nuturn_ghost > 0):
                nleap_new_ghost, lp_N_ghost = self.nleap_dist(int(Nuturn_ghost * step_size / step_size_new), nleap=nleap_new)
                assert  nleap_new == nleap_new_ghost
            else: lp_N_ghost = 0.
        
            # Calcualte different Hastings corrections
            log_prob_accept, H0, H1 = self.accept_log_prob([q0, p0], [q1, p1], return_H=True)
            log_prob_delayed = log_prob_delayed_ghost - log_prob_delayed
            log_prob_eps = epsf2.logpdf(step_size_new) - epsf1.logpdf(step_size_new)
            log_prob_N = lp_N_ghost - lp_N
            log_prob = log_prob_accept + log_prob_delayed + log_prob_eps + log_prob_N
            steplist = [eps1, eps2, step_size_new]
            stepcount = [Nuturn, Nuturn_ghost, nleap_new]

            u =  np.random.uniform(0., 1., size=1)
            if np.isnan(log_prob) or (q0-q1).sum()==0:
                qf, pf, acc = q0, p0, -99
            elif  np.log(u) > min(0., log_prob):
                qf, pf, acc = q0, p0, -2
            else: 
                qf, pf, acc = q1, p1, 2
                
            return qf, pf, acc, [H0, H1], [self.Hcount, self.Vgcount, self.leapcount], steplist, stepcount
            
        except Exception as e:
            PrintException()
            print("exception : ", e)
            return q0, p0, -99, [0, 0], [0., 0., 0.], [0, 0, 0], [0, 0, 0]

        
    def step(self, q, nleap=None, step_size=None):

        if step_size is None: step_size = self.step_size
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
        KE = self.setup_KE(self.mass_matrix)
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
        
        # Go forward
        Nuturn, qvec, pvec, gvec, log_prob_check = self.pre_step(q, p, step_size)
        
        u =  np.random.uniform(0., 1., size=1)
        if  np.log(u) > log_prob_check:
            log_prob_delayed = np.log(1 - np.exp(log_prob_check))
            return self.delayed_step(q, p, qvec, gvec, step_size=step_size, log_prob_delayed=log_prob_delayed, Nuturn=Nuturn)
            
        else:
            # uturn step
            nleap, lp_N = self.nleap_dist(Nuturn)
            q1, p1 = qvec[nleap], pvec[nleap]
        
            # Go backward
            Nuturn_rev, qvec_rev, pvec_rev, gvec_rev, log_prob_check_rev = self.pre_step(q1, -p1, step_size)
            nleap2, lp_N_rev = self.nleap_dist(Nuturn_rev, nleap=nleap)
            assert nleap2 == nleap
            stepcount = [Nuturn, Nuturn_rev, nleap]
            steplist = [step_size]*3

            # Hastings
            log_prob_accept, H0, H1 = self.accept_log_prob([q, p], [q1, p1], return_H=True)
            log_prob_accept_total = log_prob_accept + lp_N_rev + log_prob_check_rev - lp_N - log_prob_check
            log_prob_accept_total = min(0, log_prob_accept_total)
            u =  np.random.uniform(0., 1., size=1)
            if  np.log(u) > min(0., log_prob_accept_total):
                qf, pf = q, p
                accepted = 0
            else:
                qf, pf = q1, p1
                accepted = 1
                
            return qf, pf, accepted, [H0, H1], [self.Hcount, self.Vgcount, self.leapcount], steplist, stepcount


    def sample(self, q, p=None,
               nsamples=100, burnin=0, step_size=0.1, nleap=10,
               constant_trajectory=False, adapt_delayed_trajectory=False,
               epsadapt=0, nleap_adapt=0,
               target_accept=0.65, pfactor=1, pthresh=0.,
               check_delayed=False, check_uturn=False,
               callback=None, verbose=False, seed=99):

        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.nsamples = nsamples
        self.burnin = burnin
        self.step_size = step_size
        self.nleap = nleap
        self.constant_trajectory = constant_trajectory
        self.adapt_delayed_trajectory = adapt_delayed_trajectory
        self.verbose = verbose
        self.CHECK_DELAYED = check_delayed
        self.CHECK_U_TURN = check_uturn
        self.pfactor = pfactor
        self.pthresh = pthresh
        niterations = self.nsamples + self.burnin
        
        state = Sampler()
        state.steplist = []
        state.stepcount = []
        
        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt, target_accept=target_accept) 

        for i in range(niterations):
            q, p, acc, Hs, count, steplist, stepcount = self.step(q) 
            state.i += 1
            if (i%(self.nsamples//10) == 0):
                print(f"In rank {wrank}: iteration {i} of {niterations}")
            if (i > self.burnin):
                state.accepts.append(acc)
                state.samples.append(q)
                state.Hs.append(Hs)
                state.counts.append(count)
                state.steplist.append(steplist)
                state.stepcount.append(stepcount)
                if callback is not None: callback(state)

        state.to_array()
        return state
    


    

###################################################
class HMC_Adaptive_Binary_Autotune(DRHMC_AdaptiveStepsize):
    """adaptive algorithm which makes delayed proposal with prob (1-pcheck), otherwise runs u-turn sampler.
    pcheck is evaluated at the point of u-turn. The number of steps in independent of "u-turn" and sampled from pre-stored u-turn trajectories.
    """
    
    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None,  min_nleap=5, max_nleap=1024,
                 distribution='uniform', offset=0.5, p_binom=0.5,symmetric=True):
        super(HMC_Adaptive_Binary_Autotune, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob, mass_matrix=mass_matrix, min_nleap=min_nleap, max_nleap=max_nleap,
                                             distribution=distribution, offset=offset, p_binom=p_binom,symmetric=symmetric)
            
        print(self.min_nleap, self.max_nleap)

        
    def pre_step(self, q, p, step_size):
        # Go forward
        Nuturn, qvec, pvec, gvec, success = self.nuts_criterion(q, p, step_size)
       
        if (Nuturn == 0) or (not success):
            log_prob_check = -np.inf
            qvec, gvec = [], []
        else:
            qcheck, pcheck = qvec[-1], pvec[-1]
            log_prob_check, H0, Hcheck = self.accept_log_prob([q, p], [qcheck, pcheck], return_H=True)
            
        if self.CHECK_U_TURN: log_prob_check = 0. 
        if self.CHECK_DELAYED: log_prob_check = -np.inf
        #log_prob_check = log_prob_check - np.log(self.pfactor)
        if self.pfactor != 1:
            log_prob_check = self.pfactor * log_prob_check
        if self.pthresh !=0:
            if log_prob_check > np.log(self.pthresh):
                log_prob_check = 0.
            else:
                log_prob_check = -np.inf
            
        return Nuturn, qvec, pvec, gvec, log_prob_check

    
    def delayed_step(self, q0, p0, qvec, gvec, step_size, log_prob_delayed, Nuturn):
        ##CHECK DELAYED STEP IS FAILING FOR offset=0.5, uniform distribution
        ##
        verbose = self.verbose
        if verbose: print(f"trying delayed step")
        H0, H1 = 0., 0.
        try:
            # Estimate the Hessian given the rejected trajectory, use it to estimate step-size
            eps1, epsf1 = self.get_stepsize_dist(q0, p0, qvec, gvec, step_size)
            step_size_new = epsf1.rvs(size=1)[0]

            # Make the second proposal
            nleap_new = self.nleap_jitter_dist(step_size_new)
            q1, p1, _, _ = self.leapfrog(q0, p0, nleap_new, step_size_new)
            
            # Ghost trajectory for the second proposal
            Nuturn_ghost, qvec_ghost, pvec_ghost, gvec_ghost, log_prob_check_ghost = self.pre_step(q1, -p1, step_size)
            eps2, epsf2 = self.get_stepsize_dist(q1, -p1, qvec_ghost, gvec_ghost, step_size)
            log_prob_delayed_ghost = np.log(1 - np.exp(log_prob_check_ghost))
            if np.isnan(log_prob_delayed_ghost) : log_prob_delayed_ghost = -np.inf
            if self.CHECK_DELAYED: log_prob_delayed_ghost = 0.
        
            # Calcualte different Hastings corrections
            log_prob_accept, H0, H1 = self.accept_log_prob([q0, p0], [q1, p1], return_H=True)
            log_prob_delayed = log_prob_delayed_ghost - log_prob_delayed
            log_prob_eps = epsf2.logpdf(step_size_new) - epsf1.logpdf(step_size_new)
            log_prob = log_prob_accept + log_prob_delayed + log_prob_eps
            steplist = [eps1, eps2, step_size_new]
            stepcount = [nleap_new, nleap_new, nleap_new]

            u =  np.random.uniform(0., 1., size=1)
            if np.isnan(log_prob) or (q0-q1).sum()==0:
                qf, pf, acc = q0, p0, -99
            elif  np.log(u) > min(0., log_prob):
                qf, pf, acc = q0, p0, -2
            else: 
                qf, pf, acc = q1, p1, 2
                
            return qf, pf, acc, [H0, H1], [self.Hcount, self.Vgcount, self.leapcount], steplist, stepcount
            
        except Exception as e:
            PrintException()
            print("exception : ", e)
            return q0, p0, -99, [0, 0], [0., 0., 0.], [0, 0, 0], [0, 0, 0]

        
    def step(self, q, nleap=None, step_size=None):

        if step_size is None: step_size = self.step_size
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
        KE = self.setup_KE(self.mass_matrix)
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
        
        # Go forward
        Nuturn, qvec, pvec, gvec, log_prob_check = self.pre_step(q, p, step_size)
        
        u =  np.random.uniform(0., 1., size=1)
        if  np.log(u) > log_prob_check:
            log_prob_delayed = np.log(1 - np.exp(log_prob_check))
            return self.delayed_step(q, p, qvec, gvec, step_size=step_size, log_prob_delayed=log_prob_delayed, Nuturn=Nuturn)
            
        else:
            # uturn step
            nleap, lp_N = self.nleap_dist(Nuturn)
            q1, p1 = qvec[nleap], pvec[nleap]
        
            # Go backward
            Nuturn_rev, qvec_rev, pvec_rev, gvec_rev, log_prob_check_rev = self.pre_step(q1, -p1, step_size)
            nleap2, lp_N_rev = self.nleap_dist(Nuturn_rev, nleap=nleap)
            assert nleap2 == nleap
            stepcount = [Nuturn, Nuturn_rev, nleap]
            steplist = [step_size]*3

            # Hastings
            log_prob_accept, H0, H1 = self.accept_log_prob([q, p], [q1, p1], return_H=True)
            log_prob_accept_total = log_prob_accept + lp_N_rev + log_prob_check_rev - lp_N - log_prob_check
            log_prob_accept_total = min(0, log_prob_accept_total)
            u =  np.random.uniform(0., 1., size=1)
            if  np.log(u) > min(0., log_prob_accept_total):
                qf, pf = q, p
                accepted = 0
            else:
                qf, pf = q1, p1
                accepted = 1
                
            return qf, pf, accepted, [H0, H1], [self.Hcount, self.Vgcount, self.leapcount], steplist, stepcount


    def sample(self, q, p=None,
               nsamples=100, burnin=0, step_size=0.1, nleap=10,
               constant_trajectory=False, adapt_delayed_trajectory=False,
               epsadapt=0, nleap_adapt=0,
               target_accept=0.65, pfactor=1, pthresh=0.,
               check_delayed=False, check_uturn=False,
               callback=None, verbose=False, seed=99):

        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.nsamples = nsamples
        self.burnin = burnin
        self.step_size = step_size
        self.nleap = nleap
        self.constant_trajectory = constant_trajectory
        self.adapt_delayed_trajectory = adapt_delayed_trajectory
        self.verbose = verbose
        self.CHECK_DELAYED = check_delayed
        self.CHECK_U_TURN = check_uturn
        self.pfactor = pfactor
        self.pthresh = pthresh
        niterations = self.nsamples + self.burnin
        
        state = Sampler()
        state.steplist = []
        state.stepcount = []
        
        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt, target_accept=target_accept) 

        if nleap_adapt:
            print("adapting trajectory lengths")
            q = self.adapt_trajectory_length(q, nleap_adapt, target_accept)
            comm.Barrier()
            all_traj_array = np.zeros(len(self.traj_array) * wsize)
            all_traj_array_tmp = comm.gather(self.traj_array, root=0)
            if wrank == 0 :
                all_traj_array = np.concatenate(all_traj_array_tmp)
                print(f"Shape of trajectories in rank 0 : ", all_traj_array.shape)
                
            comm.Bcast(all_traj_array, root=0)
            self.traj_array = all_traj_array*1.
            self.traj_array = self.traj_array[ self.traj_array!=0]
            comm.Barrier()
            print(f"Shape of trajectories after bcast  in rank {wrank} : ", self.traj_array.shape)
            self.nleap_jitter()
            
            
        for i in range(niterations):
            q, p, acc, Hs, count, steplist, stepcount = self.step(q) 
            state.i += 1
            if (i%(self.nsamples//10) == 0):
                print(f"In rank {wrank}: iteration {i} of {niterations}")
            if (i > self.burnin):
                state.accepts.append(acc)
                state.samples.append(q)
                state.Hs.append(Hs)
                state.counts.append(count)
                state.steplist.append(steplist)
                state.stepcount.append(stepcount)
                if callback is not None: callback(state)

        state.to_array()
        return state
    


    

##############################################
###################################################
class DRHMC_Adaptive(DRHMC_AdaptiveStepsize):
    """adaptive algorithm which runs a u-turn, and upon rejection, makes delayed proposal with same number of steps,
    and estimates rejection probability for the same number of steps.
    If number of steps is unreliable, it samples a from u-turn trajectories stored during warmup
    """

    def __init__(self, D, log_prob, grad_log_prob, mass_matrix=None,  min_nleap=5, max_nleap=1024,
                 distribution='uniform', offset=0.5, p_binom=0.5, symmetric=True, mode='angles', **kwargs):
        super(DRHMC_Adaptive, self).__init__(D=D, log_prob=log_prob, grad_log_prob=grad_log_prob, mass_matrix=mass_matrix,
                                              min_nleap=min_nleap, max_nleap=max_nleap,
                                              distribution=distribution, offset=offset, p_binom=p_binom,symmetric=symmetric, **kwargs)
            
        
    def first_step(self, q, p, step_size, nleap=None, verbose=False):
        # Go forward
        Nuturn, qvec, pvec, gvec, success = self.nuts_criterion(q, p, step_size)
        if (Nuturn == 0) or (not success):
            if verbose: print("zero nuturn: ", Nuturn, nleap)
            log_prob_accept = -np.inf
            return [q, p], [qvec, pvec, gvec], log_prob_accept, [0, 0., 0.], [step_size]*3, [0, 0], 0.
        if (nleap is not None):
            if (Nuturn < nleap): #if ghost trajectory can never reach nleap proposals, it can never give correct DR proposal
                log_prob_accept = 0.
                return [q, p], [qvec, pvec, gvec], log_prob_accept, [Nuturn, Nuturn, nleap], [step_size]*3, [0, 0], nleap
            
        nleap, lp_N = self.nleap_dist(Nuturn, nleap=nleap)
        q1, p1 = qvec[nleap], pvec[nleap]
        
        # Go backward
        Nuturn_rev, qvec_rev, pvec_rev, gvec_rev, success = self.nuts_criterion(q1, -p1, step_size)
        if Nuturn_rev == 0 :
            if verbose: print("zero nuturn rev: ", Nuturn_rev, Nuturn, nleap) 
        nleap2, lp_N_rev = self.nleap_dist(Nuturn_rev, nleap=nleap)
        assert nleap2 == nleap
        stepcount = [Nuturn, Nuturn_rev, nleap]
        steplist = [step_size]*3
        self.Vgcount -= min(Nuturn_rev, nleap) #adjust for number of common steps when going backward
        
        # Hastings
        log_prob_accept, H0, H1 = self.accept_log_prob([q, p], [q1, p1], return_H=True)
        log_prob_accept_total = log_prob_accept + lp_N_rev - lp_N 
        log_prob_accept_total = min(0, log_prob_accept_total)
        
        return [q1, p1], [qvec, pvec, gvec], log_prob_accept_total, stepcount, steplist, [H0, H1], nleap
            

        
    def delayed_step(self, q0, p0, qvec, gvec, step_size, nleap, log_prob_accept_first):
        ##
        verbose = self.verbose
        if verbose: print(f"trying delayed step")
        try:
            # Estimate the Hessian given the rejected trajectory, use it to estimate step-size
            eps1, epsf1 = self.get_stepsize_dist(q0, p0, qvec, gvec, step_size)
            step_size_new = epsf1.rvs(size=1)[0]

            # Make the second proposal
            #if nleap < self.min_nleap: nleap_prime =  self.nleap_jitter_dist(step_size)
            #if len(qvec) < self.min_nleap: nleap_prime =  self.nleap_jitter_dist(step_size)
            if nleap == 0 : nleap_prime =  self.nleap_jitter_dist(step_size)
            else: nleap_prime = nleap
            if self.constant_trajectory:
                #nleap_new = int(min(self.max_nleap, max(self.min_nleap, min(nleap_prime*step_size / step_size_new, nleap*100))))
                nleap_new = int(min(self.max_nleap, max(self.min_nleap, nleap_prime*step_size / step_size_new)))
            else:
                nleap_new = int(min(self.max_nleap, max(self.min_nleap, nleap_prime)))
                    
            q1, p1, _, _ = self.leapfrog(q0, p0, nleap_new, step_size_new)
            log_prob_accept, H0, H1 = self.accept_log_prob([q0, p0], [q1, p1], return_H=True)
            
            # Ghost trajectory for the second proposal
            qp_ghost, vecs_ghost, log_prob_accept_ghost, stepcount_ghost, steplist_ghost, Hs_ghost, nleap_ghost = self.first_step(q1, -p1, step_size, nleap=nleap)
            qvec_ghost, pvec_ghost, gvec_ghost = vecs_ghost
            eps2, epsf2 = self.get_stepsize_dist(q1, -p1, qvec_ghost, gvec_ghost, step_size)
            
            # Hastings
            log_prob_delayed = np.log((1-np.exp(log_prob_accept_ghost))) - np.log((1- np.exp(log_prob_accept_first)))
            log_prob_eps = epsf2.logpdf(step_size_new) - epsf1.logpdf(step_size_new)
            log_prob_accept_total = log_prob_accept + log_prob_delayed + log_prob_eps
            log_prob_accept_total = min(0, log_prob_accept_total)
            stepcount = stepcount_ghost
            steplist = [eps1, eps2, step_size_new]

            u =  np.random.uniform(0., 1., size=1)
            if np.isnan(log_prob_accept_total) or (q0-q1).sum()==0:
                qf, pf, acc = q0, p0, -99
            elif  np.log(u) > min(0., log_prob_accept_total):
                if log_prob_delayed == -np.inf:
                    qf, pf, acc = q0, p0, -2
                    #print(log_prob_delayed, Hs_ghost, Hs_ghost[0]-Hs_ghost[1], stepcount_ghost)
                elif log_prob_eps == -np.inf: qf, pf, acc = q0, p0, -3
                else: qf, pf, acc = q0, p0, -1
            else: 
                qf, pf, acc = q1, p1, 2
                
            return qf, pf, acc, [H0, H1], [self.Hcount, self.Vgcount, self.leapcount], steplist, stepcount
            
        except Exception as e:
            PrintException()
            print("exception : ", e)
            return q0, p0, -99, [0, 0], [0., 0., 0.], [0, 0, 0], [0, 0, 0]

        

    def step(self, q, nleap=None, step_size=None):

        if step_size is None: step_size = self.step_size
        self.leapcount, self.Vgcount, self.Hcount = 0, 0, 0
        
        KE = self.setup_KE(self.mass_matrix)
        p =  multivariate_normal.rvs(mean=np.zeros(self.D), cov=self.inv_mass_matrix, size=1)
        
        qp, vecs, log_prob_accept_total, stepcount, steplist, Hs, nleap = self.first_step(q, p, step_size)
        q1, p1 = qp
        qvec, pvec, gvec = vecs
        
        # Hastings
        u =  np.random.uniform(0., 1., size=1)
        if  np.log(u) > min(0., log_prob_accept_total):
            qf, pf = q, p
            accepted = 0
            return self.delayed_step(q, p, qvec, gvec, step_size = step_size, nleap = nleap, log_prob_accept_first = log_prob_accept_total)
        else:
            qf, pf = q1, p1
            accepted = 1
            return qf, pf, accepted, Hs, [self.Hcount, self.Vgcount, self.leapcount], steplist, stepcount
        

    
    def sample(self, q, p=None,
               nsamples=100, burnin=0, step_size=0.1, nleap=10,
               constant_trajectory=False, 
               epsadapt=0, nleap_adapt=0,
               target_accept=0.65, 
               check_delayed=False, check_uturn=False,
               callback=None, verbose=False, seed=99):

        np.random.seed(seed)
        self.rng = np.random.default_rng(seed)
        self.nsamples = nsamples
        self.burnin = burnin
        self.step_size = step_size
        #self.nleap = nleap
        self.constant_trajectory = constant_trajectory
        self.verbose = verbose
        niterations = self.nsamples + self.burnin
        
        state = Sampler()
        state.steplist = []
        state.stepcount = []
        
        if epsadapt:
            q = self.adapt_stepsize(q, epsadapt, target_accept=target_accept) 

        if nleap_adapt:
            print("adapting trajectory lengths")
            q = self.adapt_trajectory_length(q, nleap_adapt, target_accept)
            comm.Barrier()
            all_traj_array = np.zeros(len(self.traj_array) * wsize)
            all_traj_array_tmp = comm.gather(self.traj_array, root=0)
            if wrank == 0 :
                all_traj_array = np.concatenate(all_traj_array_tmp)
                print(f"Shape of trajectories in rank 0 : ", all_traj_array.shape)
                
            comm.Bcast(all_traj_array, root=0)
            self.traj_array = all_traj_array*1.
            self.traj_array = self.traj_array[ self.traj_array!=0]
            comm.Barrier()
            print(f"Shape of trajectories after bcast  in rank {wrank} : ", self.traj_array.shape)
            self.nleap_jitter(lowp=10, highp=90, adjustment_factor=1.)
            #self.nleap_jitter_dist = lambda x: np.random.uniform(5, 15)
            if wrank == 0 : state.trajectories = self.trajectories
                
            
        for i in range(niterations):
            q, p, acc, Hs, count, steplist, stepcount = self.step(q) 
            state.i += 1
            if (i%(self.nsamples//10) == 0):
                print(f"In rank {wrank}: iteration {i} of {niterations}")
            if (i > self.burnin):
                state.accepts.append(acc)
                state.samples.append(q)
                state.Hs.append(Hs)
                state.counts.append(count)
                state.steplist.append(steplist)
                state.stepcount.append(stepcount)
                if callback is not None: callback(state)

        state.to_array()
        print(f"In rank {wrank}, exceeded max steps : {self.exceed_max_steps}, i.e {self.exceed_max_steps/len(state.accepts)} times")
        return state
    

