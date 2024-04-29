import numpy as np
import hmc

class GistSampler(hmc.HmcSamplerBase):
    def __init__(self, model, theta, rng, lb_frac, max_leapfrog=1024):
        super().__init__(model, -1.0, rng)
        self._theta = theta
        self._lb_frac = 0.5
        self._max_leapfrog_steps = max_leapfrog
        self._divergences = 0  # DIAGNOSTIC
        self._no_return = 0  # DIAGNOSTIC
        print("GistSampler created")
        
    def draw_stepsize(self, theta):
        _, _, H = self._model.log_density_hessian(self._theta)
        eigvals, _ = np.linalg.eigh(-H)
        lambda_max = np.max(eigvals)
        max_stepsize = 2.0 / np.sqrt(lambda_max)
        min_stepsize = self._lb_frac * max_stepsize
        # print(f"{min_stepsize=}  {max_stepsize=}")
        self._stepsize = self._rng.uniform(min_stepsize, max_stepsize)
        # print(f"{self._stepsize=}")
        p = 1.0 / (max_stepsize - min_stepsize)
        log_p = np.log(p)
        print(f"{p=}  {log_p=}")
        return stepsize, log_p

    def stepsize_lp(self, stepsize, theta):
        lp, grad, H = self._model.log_density_hessian(self._theta)
        eigvals, _ = np.linalg.eigh(H)
        lambda_max = np.max(eigvals)
        max_stepsize = 2.0 / np.sqrt(lambda_max)
        min_stepsize = self._lb_frac * max_stable_stepsize
        if stepsize < min_stepsize or stepsize > max_stepsize:
            return np.NINF
        return -np.log(max_stepsize - min_stepsize)
        
    def draw(self):
        try:
            self._rho = self._rng.normal(size=self._model.param_unc_num())
            theta = self._theta
            rho = self._rho
            log_joint_theta_rho = self.log_joint(theta, rho)
            print(f"{log_joint_theta_rho=}")
            stepsize, log_p = self.draw_stepsize(theta)
            print(f"{stepsize=}  {log_p=}")
            num_leapfrog_steps = 1
            print(f"  {self._stepsize=}")
            theta_star, rho_star = self.leapfrog(theta, rho, num_leapfrog_steps)
            lp_star = self.stepsize_lp(stepsize, theta_star)
            print(f"  {lp_star=}")
            if lp_star == np.NINF:
                self._no_return += 1  # DIAGNOSTIC
                print("  REJECT: NO RETURN")
                return self._theta, self._rho
            log_accept = (
                self.log_joint(theta_star, rho_star) + lp_star
                - (log_joint_theta_rho + lp)
            )
            if np.log(self._rng.uniform()) < log_accept:
                print("  ACCEPT")
                self._theta = theta_star
                self._rho = rho_star
        except Exception as e:
            print("  REJECT: EXCEPTION")
            self._divergences += 1  # DIAGNOSTIC
        return self._theta, self._rho
