import numpy as np
import hmc

class GistSpectralStepSampler(hmc.HmcSamplerBase):
    def __init__(self, model, theta, rng, lb_frac, max_leapfrog=1024):
        super().__init__(model, -1.0, rng)
        self._theta = theta
        self._lb_frac = 0.0
        self._max_leapfrog_steps = max_leapfrog
        self._divergences = 0  # DIAGNOSTIC
        self._no_return = 0  # DIAGNOSTIC

    def step_bounds(self, theta):
        _, _, H = self._model.log_density_hessian(theta)
        # try:
        eigvals, _ = np.linalg.eigh(-H)
        lambda_max = np.max(eigvals)
        # except Exception as e:
        #  lambda_max = np.max(np.diag(H))
        # ub_step = 2.0 / np.sqrt(lambda_max)  # this is stability limit
        ub_step = 1.5 / np.sqrt(lambda_max)
        lb_step = self._lb_frac * ub_step
        return lb_step, ub_step
        
    def draw_step(self, theta):
        min_step, max_step = self.step_bounds(theta)
        step = self._rng.uniform(min_step, max_step)
        p = 1.0 / (max_step - min_step)
        lp = np.log(p)
        return step, lp

    def stepsize_lp(self, step, theta):
        min_step, max_step = self.step_bounds(theta)
        if step < min_step or step > max_step:
            return np.NINF
        return -np.log(max_step - min_step)
        
    def draw(self):
        try:
            self._rho = self._rng.normal(size=self._model.param_unc_num())
            theta = self._theta
            rho = self._rho
            lp_theta_rho = self.log_joint(theta, rho)
            step, lp_step_theta = self.draw_step(theta)
            num_steps = 8
            self._stepsize = step
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
