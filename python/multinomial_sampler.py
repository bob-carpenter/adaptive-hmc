import hmc
import numpy as np

class MultinomialSampler(hmc.HmcSamplerBase):
    def __init__(self, model, rng, stepsize, steps, theta):
        print("MULTINOMIAL SAMPLER CONSTRUCTION")
        super().__init__(model, stepsize, rng)
        self._steps = steps
        self._theta = theta

    def draw(self):
        N = self._steps
        F = self._rng.integers(0, N)
        self._rho =  self._rng.normal(size=self._model.param_unc_num())
        theta, rho = self._theta, self._rho
        theta_star, rho_star = theta, rho
        lp = self.log_joint(theta, rho)
        log_sum_exp_lps = lp
        # forward F steps, backward B = N - F steps
        for n in range(N):
            if n == F:
                theta, rho = self._theta, -self._rho
            theta, rho = self.leapfrog_step(theta, rho)
            lp = self.log_joint(theta, rho)
            log_sum_exp_lps = np.logaddexp(log_sum_exp_lps, lp)
            switch_prob = np.exp(lp - log_sum_exp_lps)
            if self._rng.uniform(0, 1) < switch_prob:
                theta_star = theta
                rho_star = rho
        self._theta, self._rho = theta_star, rho_star
        return self._theta, self._rho
