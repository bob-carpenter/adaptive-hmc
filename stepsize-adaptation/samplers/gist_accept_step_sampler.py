import numpy as np
import hmc

STEPS_PROGRESSION = [1, 2, 3, 4, 6, 8, 11, 16, 22, 32, 45, 64, 90, 128, 181, 256, 362, 512, 724, 1024]
LOG_MIN_ACCPET = np.log(0.8)


class GistWishartSampler(hmc.HmcSamplerBase):
    def __init__(self, model, theta, rng, integration_time):
        super().__init__(model, -1.0, rng)
        self._theta = theta
        self._max_integration_time = max_integration_time
        self._divergences = 0  # DIAGNOSTIC
        self._no_return = 0  # DIAGNOSTIC

    def draw_step_lp(self, theta, rho):
        lp_theta_rho = self.log_joint(theta, rho)
        for steps in STEPS_PROGRESSION:
            self._stepsize = self._max_integration_time / steps
            theta_star, rho_star = self.leapfrog(theta, rho, steps)
            lp_theta_rho_star = self.log_joint(theta_star, rho_star)
            if lp_theta_rho_star - lp_theta_rho > LOG_MIN_ACCEPT:
                return self._rng.uniform(0, self._stepsize), -np.log(self._stepsize)
        return self._stepsize, 0

    def step_lp(self, step, theta, rho):
        orig_stepsize = self._stepsize
        lp_theta_rho = self.log_joint(theta, rho)
        for steps in STEPS_PROGRESSION:
            self._stepsize = self._max_integration_time / steps
            theta_star, rho_star = self.leapfrog(theta, rho, steps)
            lp_theta_rho_star = self.log_joint(theta_star, rho_star)
            if lp_theta_rho_star - lp_theta_rho > LOG_MIN_ACCEPT:
                if step > self._stepsize:
                    return np.NINF
                stepsize = self._stepsize
                self._stepsize = orig_stepsize
                return -np.log(0.5 * stepsize)
        self._stepsize = orig_stepsize
        return 1

    def draw(self):
        try:
            self._rho = self._rng.normal(size=self._model.param_unc_num())
            theta = self._theta
            rho = self._rho
            lp_theta_rho = self.log_joint(theta, rho)
            step, lp_step_theta = self.draw_num_steps(theta, rho)
            self._stepsize = step
            num_steps = self._max_integration_time / self._stepsize
            theta_star, rho_star = self.leapfrog(theta, rho, num_steps)
            lp_theta_rho_star = self.log_joint(theta_star, rho_star)
            lp_step_theta_star = self.stepsize_lp(self._stepsize, theta_star)
            if lp_step_theta_star == np.NINF:
                self._no_return += 1  # DIAGNOSTIC
                return self._theta, self._rho
            log_accept = (
                    (lp_theta_rho_star + lp_step_theta_star)
                    - (lp_theta_rho + lp_step_theta)
            )
            if not np.log(self._rng.uniform()) > log_accept:
                self._theta = theta_star
                self._rho = rho_star
        except Exception as e:
            print(f"  REJECT: EXCEPTION {e = }")
            self._divergences += 1  # DIAGNOSTIC
        return self._theta, self._rho
