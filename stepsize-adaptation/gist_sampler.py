import numpy as np

import hmc


class GistSampler(hmc.HmcSamplerBase):
    def __init__(self, model, stepsize, theta, rng, frac, max_leapfrog=1024):
        super().__init__(model, stepsize, rng)
        self._frac = frac
        self._max_leapfrog_steps = max_leapfrog
        self._theta = theta
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

    def lower_step_bound(self, L):
        return np.max([1, int(np.floor(self._frac * L))])

    def draw(self):
        try:
            self._rho = self._rng.normal(size=self._model.param_unc_num())
            theta = self._theta
            rho = self._rho
            log_joint_theta_rho = self.log_joint(theta, rho)
            L = self.uturn(theta, rho)
            LB = self.lower_step_bound(L)
            N = self._rng.integers(LB, L)
            theta_star, rho_star = self.leapfrog(theta, rho, N)
            rho_star = -rho_star
            Lstar = self.uturn(theta_star, rho_star)
            LBstar = self.lower_step_bound(Lstar)
            self._fwds.append(L)  # DIAGNOSTIC
            self._bks.append(Lstar)  # DIAGNOSTIC
            if not (LBstar <= N and N < Lstar):
                self._gradient_evals += L  # DIAGNOSTIC
                self._cannot_get_back_rejects += 1  # DIAGNOSTIC
                return self._theta, self._rho  # cannot balance w/o return
            self._gradient_evals += L + Lstar - N  # DIAGNOSTIC
            log_accept = (
                    self.log_joint(theta_star, rho_star)
                    - np.log(Lstar - LBstar)
                    - (log_joint_theta_rho - np.log(L - LB))
            )
            if np.log(self._rng.uniform()) < log_accept:
                self._theta = theta_star
                self._rho = rho_star
        except Exception as e:
            # traceback.print_exc()
            self._divergences += 1
        return self._theta, self._rho
