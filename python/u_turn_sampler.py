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

    def uturn(self, theta, rho):
        theta_next = theta
        rho_next = rho
        last_dist = 0
        N = 0
        while True:
            theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            dist = np.sum((theta_next - theta)**2) # sp.spatial.distance.euclidean(theta_next, theta)
            if dist <= last_dist:
                return N
            last_dist = dist
            N += 1

    def sample_tuning(self):
        N = self.uturn(self._theta, self._rho)
        self._numsteps = self._rng.integers(1, N + 1)
        self._gradient_calls -= self._numsteps  # adjustment for overlap
        self._leapfrog_steps += self._numsteps

    def logp_tune(self, theta, rho):
        N = self.uturn(theta, rho)
        # if self._numsteps > self.uturn_to_steps(N):
        #    return np.log(0) # U-turn before return
        self._gradient_calls += N
        return -np.log(N)
