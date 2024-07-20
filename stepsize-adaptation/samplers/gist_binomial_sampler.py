import numpy as np
import scipy as sp

import hmc


class GistBinomialSampler(hmc.HmcSamplerBase):
    def __init__(self, model, stepsize, theta, rng, frac, max_leapfrog=1024):
        super().__init__(model, stepsize, rng)
        self._max_leapfrog_steps = max_leapfrog
        self._theta = theta
        self._success_prob = frac
        self._cannot_get_back_rejects = 0  # DIAGNOSTIC
        self._fwds = []  # DIAGNOSTIC
        self._bks = []  # DIAGNOSTIC
        self._divergences = 0  # DIAGNOSTIC
        self._gradient_evals = 0  # DIAGNOSTIC

    def uturn(self, theta, rho):
        log_joint_theta_rho = self.log_joint(theta, rho)
        theta_next = theta
        rho_next = rho
        old_distance = 0
        for n in range(self._max_leapfrog_steps):
            theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            log_joint_next = self.log_joint(theta_next, rho_next)
            if np.abs(log_joint_theta_rho - log_joint_next) > 50.0:
                self._divergences += 1
                return n + 1
            distance = np.sum((theta_next - theta) ** 2)
            if distance <= old_distance:
                return n + 1
            old_distance = distance
        return self._max_leapfrog_steps

    def sample_length(self, L):
        n = self._rng.binomial(L - 1, self._success_prob)
        return n

    def length_log_prob(self, N, L):
        if 0 <= N and N < L:
            logp = sp.stats.binom.logpmf(N, L - 1, self._success_prob)
            return logp
        else:
            return np.NINF

    def draw(self):
        try:
            self._rho = self._rng.normal(size=self._model.param_unc_num())
            theta = self._theta
            rho = self._rho
            log_joint_theta_rho = self.log_joint(theta, rho)
            L = self.uturn(theta, rho)
            N = self.sample_length(L)
            theta_star, rho_star = self.leapfrog(theta, rho, N)
            rho_star = -rho_star
            Lstar = self.uturn(theta_star, rho_star)
            self._fwds.append(L)  # DIAGNOSTIC
            self._bks.append(Lstar)  # DIAGNOSTIC
            if not (1 <= N and N < Lstar):
                self._gradient_evals += L  # DIAGNOSTIC
                self._cannot_get_back_rejects += 1  # DIAGNOSTIC
                return self._theta, self._rho  # cannot balance w/o return
            self._gradient_evals += L + Lstar - N  # DIAGNOSTIC
            log_accept = (
                    self.log_joint(theta_star, rho_star)
                    + self.length_log_prob(N, Lstar)
                    - (log_joint_theta_rho + self.length_log_prob(N, L))
            )
            if np.log(self._rng.uniform()) < log_accept:
                self._theta = theta_star
                self._rho = rho_star
        except Exception as e:
            # traceback.print_exc()
            self._divergences += 1
        return self._theta, self._rho
