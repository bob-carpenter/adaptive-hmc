import numpy as np
import scipy as sp

import hmc


class StepAdaptSampler(hmc.HmcSamplerBase):
    def __init__(self, model, rng, integration_time, theta, min_accept_prob):
        super().__init__(model, 0.0, rng)
        self._T = integration_time
        self._theta = theta
        self._log_min_accept_prob = np.log(min_accept_prob)
        self._steps = np.unique(np.array(np.sqrt(2) ** range(21), dtype=int))

    def reject(self, min_lp, max_lp):
        return min_lp - max_lp < self._log_min_accept_prob

    def stable_steps(self, theta0, rho0, lp0, T):
        for L in self._steps:
            theta, rho = theta0, rho0
            min_lp, max_lp = lp0, lp0
            self._stepsize = T / L
            for _ in range(L):
                theta, rho = self.leapfrog_step(theta, rho)
                lp = self.log_joint(theta, rho)
                min_lp, max_lp = min(lp, min_lp), max(lp, max_lp)
                if self.reject(min_lp, max_lp):
                    break
            if not self.reject(min_lp, max_lp):
                return L
        return 40

    def draw(self):
        self._rho = self._rng.normal(size=self._model.param_unc_num())

        T = self._rng.uniform(0, self._T)

        theta0, rho0 = self._theta, self._rho
        lp0 = self.log_joint(theta0, rho0)
        L = self.stable_steps(theta0, rho0, lp0, T)
        N = self._rng.poisson(L) + 1
        lp_N = sp.stats.poisson.logpmf(N - 1, L)

        self._stepsize = T / N
        theta_star, rho_star = self.leapfrog(self._theta, self._rho, N)
        rho_star = -rho_star
        lp_star = self.log_joint(theta_star, rho_star)
        L_star = self.stable_steps(theta_star, rho_star, lp_star, T)
        lp_star_N = sp.stats.poisson.logpmf(N - 1, L_star)

        accept_prob = np.exp((lp_star + lp_star_N) - (lp0 + lp_N))
        if self._rng.uniform(0, 1) < accept_prob:
            self._theta, self._rho = theta_star, rho_star

        return self._theta, self._rho
