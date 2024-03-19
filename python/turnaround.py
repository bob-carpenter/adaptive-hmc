import numpy as np
import scipy as sp
import util
import models

class TurnaroundSampler:
    def __init__(self, model, stepsize, rng):
        self._model = model
        self._stepsize = stepsize
        self._rng = rng
        self._theta = self._rng.normal(size=model.param_unc_num())
        self._rho = self._rng.normal(size=model.param_unc_num())

    def __iter__(self):
        return self

    def __next__(self):
        return self.draw()

    def leapfrog_step(self, theta, rho):
        _, grad = self._model.log_density_gradient(theta)
        rho2 = rho + 0.5 * self._stepsize * grad
        theta2 = theta + self._stepsize * rho2
        _, grad = self._model.log_density_gradient(theta2)
        rho2 += 0.5 * self._stepsize * grad
        return theta2, rho2

    def leapfrog(self, theta, rho, numsteps):
        theta, rho = theta.copy(), rho.copy()
        for _ in range(numsteps):
            theta, rho = self.leapfrog_step(theta, rho)
        return theta, rho
    
    def log_joint(self, theta, rho):
        return self._model.log_density(theta) - 0.5 * sum(rho**2)

    def uturn(self, theta, rho):
        theta_next = theta.copy()
        rho_next = rho.copy()
        old_distance = 0
        N = 0
        while True:
            theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            N += 1
            distance = np.sum((theta_next - theta)**2)
            if distance <= old_distance:
                return N
            old_distance = distance
    
    def draw(self):
        self._rho = self._rng.normal(size=self._model.param_unc_num())
        logp = self.log_joint(self._theta, self._rho)
        N = self.uturn(self._theta, self._rho)
        L = self._rng.integers(1, N - 1)  # exclude initial and final points
        theta_prop, rho_prop = self.leapfrog(self._theta, self._rho, L)
        rho_prop = -rho_prop
        logp_prop = self.log_joint(theta_prop, rho_prop)
        M = self.uturn(theta_prop, rho_prop)
        logp_out = -np.log(N - 1)
        logp_back = -np.log(M - 1)
        if np.log(self._rng.uniform()) < (logp_prop - logp) + (logp_out - logp_back):
            self._theta = theta_prop
            self._rho = rho_prop
        return self._theta, self._rho

    def sample(self, M):
        thetas = np.empty((M, self._model.param_unc_num()), dtype=np.float64)
        thetas[0, :] = self._theta
        for m in range(1, M):
            thetas[m, :], _ = self.draw()
        return thetas
