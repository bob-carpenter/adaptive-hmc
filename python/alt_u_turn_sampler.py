import adaptive_hmc as ahmc
import numpy as np
import scipy as sp

UNIFORM = True

class AltUTurnSampler(ahmc.AdaptiveHmcSampler):
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
        # return -np.log(N)                              # = log uniform(L | 1, N)
        return N

    def draw(self):
        self._rho = self._rng.normal(size=self._model.dims())
        logp = self.joint_logp(self._theta, self._rho)

        self.sample_tuning()
        N = self.logp_tune(self._theta, self._rho)
        logp_tune = -np.log(N)

        theta_prop, rho_prop = self.leapfrog()
        rho_prop = -rho_prop

        logp_prop = self.joint_logp(theta_prop, rho_prop)
        B = self.logp_tune(theta_prop, rho_prop)
        logp_tune_prop = -np.log(B + 1)

        self._proposed += 1
        if np.log(self._rng.uniform()) < ((logp_prop + logp_tune_prop) - (logp + logp_tune)):
            self._accepted += 1
            self._theta = theta_prop
            self._rho = rho_prop
        return self._theta, self._rho
