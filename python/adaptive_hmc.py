import numpy as np
import scipy as sp
import util
import models

class AdaptiveHmcSampler:
    def __init__(self, model, stepsize, numsteps, seed, theta0, rho0):
        self._model = model
        self._stepsize = stepsize
        self._numsteps = numsteps
        self._rng = (
            np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        )
        self._theta = theta0 if theta0 is not None else self._rng.normal(size=model.dims())
        self._rho = rho0 if rho0 is not None else self._rng.normal(size=model.dims())
        self._proposed = 0
        self._accepted = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.draw()

    def joint_logp(self, theta, rho):
        return self._model.log_density(theta) - 0.5 * np.dot(rho, rho)

    def leapfrog_step(self, theta, rho):
        _, grad = self._model.log_density_gradient(theta)
        rho2 = rho + 0.5 * self._stepsize * grad
        theta2 = theta + self._stepsize * rho2
        _, grad = self._model.log_density_gradient(theta2)
        rho2 += 0.5 * self._stepsize * grad
        return theta2, rho2

    def leapfrog(self):
        theta = self._theta
        rho = self._rho
        for _ in range(self._numsteps):
            theta, rho = self.leapfrog_step(theta, rho)
        return theta, rho

    def draw(self):
        self._rho = self._rng.normal(size=self._model.dims())
        logp = self.joint_logp(self._theta, self._rho)

        self.sample_tuning()
        logp_tune = self.logp_tune(self._theta, self._rho)

        theta_prop, rho_prop = self.leapfrog()
        rho_prop = -rho_prop

        logp_prop = self.joint_logp(theta_prop, rho_prop)
        logp_tune_prop = self.logp_tune(theta_prop, rho_prop)

        self._proposed += 1
        if np.log(self._rng.uniform()) < ((logp_prop + logp_tune_prop) - (logp + logp_tune)):
            self._accepted += 1
            self._theta = theta_prop
            self._rho = rho_prop
        return self._theta, self._rho
        
    def sample(self, M):
        thetas = np.empty((M, self._model.dims()), dtype=np.float64)
        thetas[0, :] = self._theta
        for m in range(1, M):
            thetas[m, :], _ = self.draw()
        return thetas
