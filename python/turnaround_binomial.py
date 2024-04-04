import numpy as np
import scipy as sp
import hmc
import traceback

class TurnaroundSampler(hmc.HmcSamplerBase):
    """Adaptive HMC algorithm for selecting number of leapfrog steps.

    Samples number of steps from 1 up to U-turn with probability
    proportional to number of steps, flips momentum, then balances
    with reverse proposal probability. 
    """
    def __init__(self, model, stepsize, theta, rng, uturn_condition, path_fraction,
                     max_leapfrog = 1024):
        super().__init__(model, stepsize, rng)
        self._max_leapfrog_steps = max_leapfrog
        self._theta = theta
        self._cannot_get_back_rejects = 0  # DIAGNOSTIC
        self._fwds = []                    # DIAGNOSTIC
        self._bks = []                     # DIAGNOSTIC
        self._divergences = 0              # DIAGNOSTIC

    def uturn(self, theta, rho):
        log_joint_theta_rho = self.log_joint(theta, rho)
        theta_next = theta
        rho_next = rho
        old_distance = 0
        for n in range(self._max_leapfrog_steps):
            theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            log_joint_next = self.log_joint(theta_next, rho_next)
            if np.abs(log_joint_theta_rho - log_joint_next) > 20.0:
                self._divergences += 1
                return n + 1
            distance = np.sum((theta_next - theta)**2)
            if distance <= old_distance:
                return n + 1
            old_distance = distance
        return self._max_leapfrog_steps

    def range_probs(self, L):
        lengths = np.arange(L)
        # q = lengths
        q = np.sqrt(lengths)
        # q = np.log(1 + lengths)
        # q = np.log2(1 + lengths)
        # q = np.log10(1 + lengths)
        p = q / np.sum(q)
        return lengths, p

    def sample_length(self, L):
        lengths, p = self.range_probs(L)
        n = self._rng.binomial(L, 0.6)
        return n

    def length_log_prob(self, N, L):
        # print(f"{N=} {L=}")
        _, p = self.range_probs(L)
        if 0 <= N and N < L:
            # logp = np.log(p[N])
            logp = sp.stats.binom.logpmf(N, L, 0.6)
            return logp
        else:
            return np.NINF
        
    def draw(self):
        try:
            self._rho = self._rng.normal(size=self._model.param_unc_num())
            theta = self._theta
            rho = self._rho
            log_joint_theta_rho = self.log_joint(theta, rho)
            L = self.uturn(theta, rho)
            # print(f"{L=}")
            N = self.sample_length(L)
            # print(f"{L=} {N=}")
            theta_star, rho_star = self.leapfrog(theta, rho, N)
            rho_star = -rho_star
            Lstar = self.uturn(theta_star, rho_star)
            self._fwds.append(L)                     # DIAGNOSTIC
            self._bks.append(Lstar)                  # DIAGNOSTIC
            if not(1 <= N and N < Lstar):
                self._cannot_get_back_rejects += 1   # DIAGNOSTIC
                return self._theta, self._rho        # cannot balance w/o return
            log_accept = ((self.log_joint(theta_star, rho_star)
                               - self.length_log_prob(N, Lstar))
                          - (log_joint_theta_rho
                                 - self.length_log_prob(N, L)))
            if np.log(self._rng.uniform()) < log_accept:
                self._theta = theta_star
                self._rho = rho_star
        except Exception as e:
            print(f"{e=}")
            # traceback.print_exc()
            self._divergences += 1
        return self._theta, self._rho

