import adaptive_hmc as ahmc
import numpy as np
import scipy as sp

UNIFORM = True

class UTurnSampler(ahmc.AdaptiveHmcSampler):
    def __init__(
        self, model, stepsize=0.9, numsteps=4, seed=None, theta0=None, rho0=None
    ):
        super().__init__(model, stepsize, numsteps, seed, theta0, rho0)
        self._gradient_calls = 0
        self._leapfrog_steps = 0
        self._NMsteps = []

    def uturn(self, theta, rho):
        theta_next = theta
        rho_next = rho
        last_dist = 0
        N = 0
        while True:
            theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            dist = np.sum((theta_next - theta)**2)
            if dist <= last_dist:
                return N
            last_dist = dist
            N += 1

    def sample_tuning(self):
        N = self.uturn(self._theta, self._rho)
        self._numsteps = self._rng.integers(1, N + 1)  # uniform(1, N) inclusive
        self._gradient_calls -= self._numsteps         # adjust for overlap
        self._leapfrog_steps += self._numsteps

    def logp_tune(self, theta, rho):
        N = self.uturn(theta, rho)
        self._NMsteps.append(N)
        self._gradient_calls += N
        return -np.log(N)                              # = log uniform(L | 1, N)

class UTurnBinomialSampler(ahmc.AdaptiveHmcSampler):
    def __init__(
        self, model, stepsize=0.9, numsteps=4, seed=None, theta0=None, rho0=None
    ):
        super().__init__(model, stepsize, numsteps, seed, theta0, rho0)
        self._gradient_calls = 0
        self._leapfrog_steps = 0
        self._NMsteps = []
        self._successProb = 0.95

    def uturn(self, theta, rho):
        theta_next = theta
        rho_next = rho
        last_dist = 0
        N = 0
        while True:
            theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            dist = np.sum((theta_next - theta)**2)
            if dist <= last_dist:
                return N
            last_dist = dist
            N += 1

    def sample_tuning(self):
        N = self.uturn(self._theta, self._rho)
        self._numsteps = self._rng.binomial(N - 1, self._successProb)
        self._gradient_calls -= self._numsteps
        self._leapfrog_steps += self._numsteps

    def logp_tune(self, theta, rho):
        N = self.uturn(theta, rho)
        self._NMsteps.append(N)
        self._gradient_calls += N
        return sp.stats.binom.logpmf(self._numsteps, N - 1, self._successProb)
    
    
