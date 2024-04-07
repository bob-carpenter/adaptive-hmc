import numpy as np
import hmc
import traceback

class ProgressiveTurnaroundSampler(hmc.HmcSamplerBase):
    def __init__(self, model, stepsize, theta, rng, max_leapfrog = 512):
        super().__init__(model, stepsize, rng)
        self._uturn_condition = "blah" # uturn_condition
        self._path_fraction = "blah" # path_fractifon
        self._max_leapfrog_steps = max_leapfrog
        self._theta = theta
        self._cannot_get_back_rejects = 0  # DIAGNOSTIC
        self._fwds = []                    # DIAGNOSTIC
        self._bks = []                     # DIAGNOSTIC
        self._divergences = 0              # DIAGNOSTIC
        self._gradient_evals = 0           # DIAGNOSTIC

    def uturn(self, theta, rho):
        H0 = self.log_joint(theta, rho)
        theta_next = theta
        rho_next = rho
        old_distance = 0
        theta_prop = theta
        rho_prop = rho
        lsw = -np.inf
        L = 0
        for n in range(self._max_leapfrog_steps):
            theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            H = self.log_joint(theta_next, rho_next)
            if np.abs(H0 - H) > 10.0:
                return n, theta_prop, rho_prop, L, lsw
            lsw = np.logaddexp(lsw, H)
            log_alpha = H - lsw
            if np.log(self._rng.uniform()) < np.minimum(0.0, log_alpha):
                theta_prop = theta_next
                rho_prop = rho_next
                L = n
            distance = np.sum((theta_next - theta) ** 2)
            if distance <= old_distance:
                return n + 1, theta_prop, rho_prop, L + 1, lsw
            old_distance = distance
        return n + 1, theta_prop, rho_prop, L + 1, lsw

    def draw(self):
        try:
            self._rho = self._rng.normal(size=self._model.param_unc_num())
            theta = self._theta
            rho = self._rho

            N, theta_star, rho_star, L, lsw_N = self.uturn(theta, rho)
            rho_star = -rho_star
            M, _, _, _, lsw_M = self.uturn(theta_star, rho_star)

            self._gradient_evals += N + M
            self._fwds.append(N)
            self._bks.append(M)

            if not(1 <= L and L <= M):
                self._cannot_get_back_rejects += 1
                return self._theta, self._rho  # unbalance-able

            log_alpha = lsw_N - lsw_M
            if np.log(self._rng.uniform()) < np.minimum(0.0, log_alpha):
                self._theta = theta_star
                self._rho = rho_star
        except Exception as e:
            # traceback.print_exc()
            self._divergences += 1
        return self._theta, self._rho
