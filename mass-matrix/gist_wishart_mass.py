import hmc
from scipy.stats import invwishart
import numpy as np


# APPROXIMATION:
# negative outer product of gradients at theta is rank-1 estimate of
# local covariance; add epsilon to diagonal for positive definiteness

# SCALING WISHART:
# mean of invWishart(nu, Sigma) is Sigma / (nu - D - 1) in D dimensions

class GistMassSampler(hmc.HmcSamplerBase):
    def __init__(self, model, stepsize, rng, theta, inv_mass_matrix, dof = 100, epsilon=1e-6):
        super().__init__(model, stepsize, rng, theta, theta, inv_mass_matrix)
        self._dof = dof
        self._epsilon = epsilon

    def approx_inv_mass(self, theta):
        _, grad = self._model.log_density_gradient(theta)
        approx_cov = -grad @ grad.T
        np.fill_diagonal(approx_cov, approx_cov.diagonal() + self._epsilon)
        return approx_cov

    def tuning_conditional(self, theta):
        approx_inv_mass = self.approx_inv_mass(theta)
        expectation_adjustment = self._dof - self._model.param_unc_num() - 1
        return invwishart(self._dof, approx_inv_mass * expectation_adjustment)
        
    def draw(self):
        self.refresh_momentum()

        theta = self._theta
        rho = self._rho
        lp_theta_rho = self.log_joint(theta, rho)

        tune = self.tuning_conditional(theta)
        inv_mass = tune.rvs()
        lp_inv_mass = tune.logpdf(mass)

        set_inv_mass(inv_mass)
        theta_star, rho_star = self.leapfrog_step(theta, rho)
        lp_theta_rho_star = self.log_joint(theta_star, rho_star)

        tune_star = self.tuning_conditional(theta_star)
        lp_inv_mass_star = tune_star.logpdf(inv_mass)

        log_accept = (
            lp_theta_rho_star + lp_inv_mass_star - (lp_theta_rho + lp_inv_mass)
        )
        if np.log(self._rng.uniform()) < log_accept:
            self._theta = theta_star
            self._rho = rho_star

        return self._theta, self._rho
