import numpy as np


class AdaptiveHmcSampler:
    def __init__(self, model, stepsize, numsteps, seed, theta0, rho0):
        self._model = model
        self._stepsize = stepsize
        self._numsteps = numsteps
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self._theta = theta0 if theta0 is not None else rng.normal(size=model.dims())
        self._rho = rho0 if rho0 is not None else rng.normal(size=model.dims())
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
        self._rho = rng.normal(size = self._model.dims())
        logp = self.joint_logp(self._theta, self._rho)

        self.sample_tuning()
        logp_tune = self.logp_tune(self._theta, self._rho)

        theta_prop, rho_prop = self.leapfrog()
        rho_prop = -rho_prop

        logp_prop = self.joint_logp(theta_prop, rho_prop)
        logp_tune_prop = self.logp_tune(theta_prop, rho_prop)

        self._proposed += 1
        if np.log(rng.uniform()) < (logp_prop - logp) + (logp_tune_prop - logp_tune):
            self._accepted += 1
            self._theta = theta_prop
            self._rho = rho_prop
        return self._theta, self._rho
    
    def sample(self, M):
        thetas = np.empty((M, model.dims()), dtype=np.float64)
        thetas[0, :] = self._theta
        for m in range(1, M):
            thetas[m, :], _ = self.draw()
        return thetas

class UTurnSampler(AdaptiveHmcSampler):
    def __init__(self, model, stepsize = 0.5, numsteps = 4, seed = None, theta0 = None, rho0 = None):
        super().__init__(model, stepsize, numsteps, seed, theta0, rho0)
        self._gradient_calls = 0
        self._leapfrog_steps = 0
        
    def uturn(self, theta, rho):
        theta_next = theta
        rho_next = rho
        last_dist_sq = 0
        L = 0
        while True:
            L += 1
            theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            diff = theta_next - theta
            dist_sq = np.sum(diff**2)
            if dist_sq <= last_dist_sq:
                return L  # L >= 2 because 1 step can't u-turn
            last_dist_sq = dist_sq

            
    def uturn_to_steps(self, N):
        return N
            
    def sample_tuning(self):
        N = self.uturn(self._theta, self._rho)
        # exclude start, include U-turn
        self._numsteps = rng.integers(1, self.uturn_to_steps(N))
        # careful impl would share these steps forward/reverse
        self._gradient_calls += self._numsteps - N
        self._leapfrog_steps += self._numsteps

    def logp_tune(self, theta, rho):
        N = self.uturn(theta, rho)
        if self._numsteps > self.uturn_to_steps(N):
            return np.log(0)
        # called once forward, once reverse
        self._gradient_calls += N
        # log uniform(self._numsteps | 1, N - 1) = log(1 / (N - 1))
        return -np.log(self.uturn_to_steps(N))

        
class StdNormal:
    def __init__(self, dims = 1):
        self._dims = dims

    def log_density(self, x):
        return -0.5 * np.dot(x, x)

    def log_density_gradient(self, x):
        return self.log_density(x), -x

    def dims(self):
        return self._dims

M = 1000
theta0 = np.array([0.2, -1.3, -0.1, -3.9, 4.8])
rng = np.random.default_rng()
stepsize = 0.9
L = 10

D = 5
model = StdNormal(D)
sampler = UTurnSampler(model, stepsize)
sample = sampler.sample(M)

np.set_printoptions(precision=3)
print(f"   gradient calls: {sampler._gradient_calls}")
print(f"gradient calls/it: {sampler._gradient_calls / M:6.2f}\n")

print(f"   leapfrog steps: {sampler._leapfrog_steps}")
print(f"leapfrog steps/it: {sampler._leapfrog_steps / M:6.2f}\n")

print(f"   mean: {np.mean(sample, axis=0)}")
print(f"std dev: {np.std(sample, axis=0, ddof=1)}\n")

print(f"   mean (sq): {np.mean(sample**2, axis=0)}")
print(f"std dev (sq): {np.std(sample**2, axis=0, ddof=1)}\n")

def mean_sq_jump_distance(sample):
    sq_jump = np.empty(M - 1, np.float64)
    for m in range(M - 1):
        jump = sample[m + 1, :] - sample[m, :]
        sq_jump[m] = jump.dot(jump)
    return np.mean(sq_jump)


print(f"mean squared jump distance = {mean_sq_jump_distance(sample):6.2f}")

iid_sample = np.random.normal(size = (M, D))
print(f"ind. sample mean sq. jump distance = {mean_sq_jump_distance(iid_sample):6.2f}")

print(f" accept: {sampler._accepted / sampler._proposed:4.2f}")

import plotnine as pn
import pandas as pd
import scipy as sp
df = pd.DataFrame({'x': sample[1:M, 1] })

plot = ( pn.ggplot(df, pn.aes(x = 'x'))
         + pn.geom_histogram(pn.aes(y='..density..'), bins=50,
                             color='black', fill = 'white')
         + pn.stat_function(fun=sp.stats.norm.pdf,
                            args={'loc': 0, 'scale': 1},
                            color='red', size=1)
       )
# print(plot)
