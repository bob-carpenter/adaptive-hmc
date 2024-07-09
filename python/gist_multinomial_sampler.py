import scipy as sp
import numpy as np
import hmc
import traceback

class GistSampler(hmc.HmcSamplerBase):
    def __init__(self, model, stepsize, theta, rng, frac, max_leapfrog=1024):
        super().__init__(model, stepsize, rng)
        self._frac = -1 # n/a here but didn't want to change interface
        self._max_leapfrog_steps = max_leapfrog
        self._theta = theta
        self._cannot_get_back_rejects = 0  # DIAGNOSTIC
        self._fwds = []  # DIAGNOSTIC
        self._bks = []  # DIAGNOSTIC
        self._divergences = 0  # DIAGNOSTIC
        self._gradient_evals = 0  # DIAGNOSTIC

    def uturn(self, theta, rho):
        log_joint_theta_rho = self.log_joint(theta, rho)
        # lps = [-100_000]
        lps = [-100_000 + log_joint_theta_rho]  # include initial lp, but avoid trying it
        theta_next = theta
        rho_next = rho
        old_distance = 0
        for n in range(self._max_leapfrog_steps):
            theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            log_joint_next = self.log_joint(theta_next, rho_next)
            if np.abs(log_joint_theta_rho - log_joint_next) > 50.0:
                self._divergences += 1
                return n + 1, lps
            distance = np.sum((theta_next - theta) ** 2)
            if distance <= old_distance:
                return n + 1, lps
            old_distance = distance
            lps.append(np.log(distance) + log_joint_next)  # include if didn't U-turn or diverge
        return self._max_leapfrog_steps, lps

    def draw(self):
        try:
            self._rho = self._rng.normal(size=self._model.param_unc_num())
            theta = self._theta
            rho = self._rho
            log_joint_theta_rho = self.log_joint(theta, rho)
            L, lps = self.uturn(theta, rho)
            logsumexp_lps = sp.special.logsumexp(lps)
            probs = sp.special.softmax(lps)
            N = np.random.choice(len(probs), p = probs)
            theta_star, rho_star = self.leapfrog(theta, rho, N)
            rho_star = -rho_star
            Lstar, lps_star = self.uturn(theta_star, rho_star)
            logsumexp_lps_star = sp.special.logsumexp(lps_star)
            if len(lps_star) - 1 < N:
                self._gradient_evals += L  # DIAGNOSTIC
                self._cannot_get_back_rejects += 1  # DIAGNOSTIC
                return self._theta, self._rho  # cannot balance w/o return
            self._gradient_evals += L + Lstar - N  # DIAGNOSTIC
            log_accept = logsumexp_lps - logsumexp_lps_star
            if np.log(self._rng.uniform()) < log_accept:
                self._theta = theta_star
                self._rho = rho_star
        except Exception as e:
            traceback.print_exc()
            self._divergences += 1
        return self._theta, self._rho
