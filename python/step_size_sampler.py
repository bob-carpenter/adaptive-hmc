import numpy as np
import scipy as sp
import adaptive-hmc as ahmc

class StepSizeSampler(ahmc.AdaptiveHmcSampler):
    def __init__(
        self, model, tolerance, integration_time, stepsize, numsteps, seed, theta0, rho0
    ):
        super().__init__(model, stepsize, numsteps, seed, theta0, rho0)
        self._max_stepsize = stepsize
        self._tolerance = tolerance

    def expected_steps(self):
        numsteps_save = self._numsteps
        self._numsteps = 2
        H_0 = self.logp(self._theta, self._rho)
        while True:
            self._stepsize = self._max_stepsize / self._numsteps
            theta_star, rho_star = self.leapfrog()
            H_star = self.logp_joint(theta_star, rho_star)
            if np.abs(H_0 - H_star) < self._tolerance:
                break
            self._numsteps += 1
        num_steps_out = self._numsteps
        self._numsteps = numsteps_save  # horrible abuse of OO member as local var
        return num_steps_out

    def sample_tuning(self):
        N = self.expected_steps(self._theta, self._rho)
        self._numsteps = np.random.poisson(N)
        self._stepsize = self._max_stepsize / self._numsteps

    def logp_tune(self, theta, rho):
        N = self.expected_steps(theta, rho)
        return sp.stats.poisson.logpmf(self._numsteps, N)
