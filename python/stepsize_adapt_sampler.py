import hmc
import numpy as np
import scipy as sp

class StepAdaptSampler(hmc.HmcSamplerBase):
    def __init__(self, model, rng, integration_time, theta, min_accept_prob):
        super().__init__(model, 0.0, rng)
        self._T = integration_time
        self._theta = theta
        self._min_accept_prob = min_accept_prob


    def stable_steps(self, theta0, rho0, lp0):
        for L in [2, 3, 4, 5, 8, 11, 16, 20, 32]:
            theta, rho = theta0, rho0
            min_lp, max_lp = lp0, lp0
            self._stepsize = self._T / L
            for _ in range(L):
                theta, rho = self.leapfrog_step(theta, rho)
                lp = self.log_joint(theta, rho)
                if lp < min_lp:
                    min_lp = lp
                if lp > max_lp:
                    max_lp = lp
            if np.exp(min_lp - max_lp) > self._min_accept_prob:
                return L
        return 40

    def draw(self):
        self._rho = self._rng.normal(size=self._model.param_unc_num())

        theta0, rho0 = self._theta, self._rho
        lp0 = self.log_joint(theta0, rho0)
        L = self.stable_steps(theta0, rho0, lp0)
        N = self._rng.poisson(L) + 1
        lp_N = sp.stats.poisson.logpmf(N - 1, L)

        self._stepsize = self._T / N
        theta_star, rho_star = self.leapfrog(self._theta, self._rho, N)
        rho_star = -rho_star
        lp_star = self.log_joint(theta_star, rho_star)
        L_star = self.stable_steps(theta_star, rho_star, lp_star)
        lp_star_N = sp.stats.poisson.logpmf(N - 1, L_star)

        accept_prob = np.exp((lp_star + lp_star_N) - (lp0 + lp_N))
        if self._rng.uniform(0, 1) < accept_prob:
            self._theta, self._rho = theta_star, rho_star

        return self._theta, self._rho
