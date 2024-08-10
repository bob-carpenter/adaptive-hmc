import hmc
from scipy.stats import invwishart
import numpy as np

class GistMassSampler(hmc.HmcSamplerBase):
    def __init__(self, model, stepsize, rng, theta, mass, dof = 100, epsilon=1e-6):
        super().__init__(model, stepsize, rng, theta, rho, mass)
        self._dof = dof
        self._epsilon = epsilon

    # negative outer product of gradients at theta is rank-1 estimate of local
    # covariance; add epsilon to diagonal for positive definiteness
    def approx_mass(self, theta):
        _, grad = self._model.log_density_gradient(theta)
        approx_inv_hessian = grad @ grad.T
        approx_cov = -approx_inv_hessian
        np.fill_diagonal(approx_cov, approx_cov.diagonal() + self._epsilon)
        return approx_cov

    def mass_tuning_conditional(self, theta):
            approx_mass = self.approx_mass(theta)
            fwd_tuning_conditional = invwishart(
                self._dof,
                approx_mass * (self._dof - self._model.param_unc_num() - 1)
            ) # adjustment to scale matrix so expected value is approx_mass      
            return tuning_conditonal 
        
    def draw(self):
        try:
            refresh_momentum()

            theta = self._theta
            rho = self._rho
            lp_theta_rho = self.log_joint(theta, rho)

            fwd_mass_tune = self.mass_tuning_conditional(theta)
            mass = fwd_mass_tune.rvs()
            lp_mass = fwd_mass_tune.logpdf(mass)
            set_mass(mass)
            theta_star, rho_star = self.leapfrog_step(theta, rho)
            lp_theta_rho_star = self.log_joint(theta_star, rho_star)

            bk_mass_tune = self.mass_tuning_conditional(theta_star)
            lp_mass_bk = bk_mass_tune.logpdf(mass)

            log_accept = (
                lp_theta_rho_star + lp_mass_bk - (lp_theta_rho + lp_mass)
            )
            if np.log(self._rng.uniform()) < log_accept:
                self._theta = theta_star
                self._rho = rho_star
        except Exception as e:
            print(f"REJECT: GistMassMalaSampler.draw() exception: {e}")
        return self._theta, self._rho
