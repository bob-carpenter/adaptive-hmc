import hmc
import numpy as np
import scipy as sp
import traceback

class MultinomialSampler(hmc.HmcSamplerBase):
    def __init__(self, rng, model, stepsize, steps, theta, max_leapfrog=1024):
        super().__init__(model, stepsize, rng)
        self._steps = steps
        self._theta = theta
        self._max_leapfrog_steps = max_leapfrog

    def draw(self):
        theta, rho = self._theta, self._rho
        try:
            return draw_divergent(self)
        except Exception as e:
            self._theta, self._rho = theta, rho
            return theta, rho
         
    def draw_divergent(self):
        rng = self._rng
        L = self._steps
        self._rho =  self._rng.normal(size=self._model.param_unc_num())
        F = rng.integers(0, L)
        theta, rho = self._theta, self._rho
        lps = np.zeros(L + 1)
        states = [(None, None)] * (L + 1)
        lps[0] = self.log_joint(theta, rho)
        states[0] = (theta, rho)
        for n in range(1, L + 1):
            if n == F + 1:
                theta, rho = self._theta, -self._rho
            theta, rho = self.leapfrog_step(theta, rho)
            lps[n] = self.log_joint(theta, rho)
            states[n] = (theta, rho)
        ps = sp.special.softmax(lps)
        n = rng.sample(range(L + 1), ps)
        self._theta, self._rho = states[n]
        return self._theta, self._rho

    def draw_divergent_online(self):
        F = rng.integers(0, self._steps)
        self._rho =  self._rng.normal(size=self._model.param_unc_num())
        theta, rho = self._theta, self._rho
        theta_star, rho_star = theta, rho
        lp_sum = self.log_joint(theta, rho)
        for n in range(1, self._steps + 1):
            if n == F + 1:
                theta, rho = self._theta, -self._rho
            theta, rho = self.leapfrog_step(theta, rho)
            lp = self.log_joint(theta, rho)
            lp_sum = logaddexp(lp_sum, lp)
            if np.log(self._rng.uniform(0, 1)) < lp - lp_sum:
                theta_star = theta
                rho_star = rho
        self._theta, self._rho = theta_star, rho_star
        return self._theta, self._rho
    
