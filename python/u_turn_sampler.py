import adaptive_hmc as ahmc
import numpy as np
import scipy as sp

class UTurnSampler(ahmc.AdaptiveHmcSampler):
    def __init__(
        self, model, stepsize=0.5, numsteps=4, seed=None, theta0=None, rho0=None
    ):
        super().__init__(model, stepsize, numsteps, seed, theta0, rho0)
        self._gradient_calls = 0
        self._leapfrog_steps = 0

    def uturn_to_steps(self, N):
        return N + 1

    def uturn(self, theta, rho):
        theta_next = theta
        rho_next = rho
        last_dist = 0
        L = 0
        while True:
            L += 1
            theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            dist = sp.spatial.distance.euclidean(theta_next, theta)
            if dist <= last_dist:
                return L
            last_dist = dist

    def sample_tuning(self):
        N = self.uturn(self._theta, self._rho)
        steps = self.uturn_to_steps(N)

        self._numsteps = self._rng.integers(1, steps)
        # (WEIGHT) p = np.arange(1, steps + 1)
        # (WEIGHT) p = p / np.sum(p)
        # (WEIGHT) self._numsteps = np.random.choice(a = np.arange(1, steps + 1), p = p)

        if self._numsteps <= N:
            self._gradient_calls -= self._numsteps  # adjustment for overlap
        else:
            self._gradient_calls += self._numsteps  # rev subseq of fwd
        self._leapfrog_steps += self._numsteps

    def logp_tune(self, theta, rho):
        N = self.uturn(theta, rho)
        if self._numsteps > self.uturn_to_steps(N):
            return np.log(0)
        if self._numsteps <= N:  # add forward and reverse
            self._gradient_calls += N
        steps = self.uturn_to_steps(N)

        return -np.log(self.uturn_to_steps(N) - 1)  # steps selected exclusive of end
        # (WEIGHT) p = np.arange(1, steps + 1)
        # (WEIGHT) p = p / np.sum(p)
        # (WEIGHT) return np.log(p[self._numsteps - 1])
