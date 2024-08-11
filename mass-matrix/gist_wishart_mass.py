import hmc
from scipy.stats import invwishart
import numpy as np


# APPROXIMATION:
# negative outer product of gradients at theta is rank-1 estimate of
# local covariance; add epsilon to diagonal for positive definiteness

# SCALING WISHART:
# mean of invWishart(nu, Sigma) is Sigma / (nu - D - 1) in D dimensions

class GistMassSampler(hmc.HmcSamplerBase):
    def __init__(self, model, stepsize, rng, theta, inv_mass_matrix, dof = 100, epsilon = 1e-6):
        super().__init__(model, stepsize, rng, theta, theta, inv_mass_matrix)
        self._dof = dof
        self._epsilon = epsilon
        self._accepts = 0
        self._tries = 0
        
    def approximate_covariance(self, theta):
        _, _, hess = self._model.log_density_hessian(theta)
        return np.linalg.inv(-hess)

    def approximate_covariance_rv(self, theta):
        approx_cov = self.approximate_covariance(theta)
        expectation_adjustment = self._dof - self._model.param_unc_num() - 1
        return invwishart(self._dof, approx_cov * expectation_adjustment)

    def accept_rate(self):
        return self._accepts / self._tries
    
    def draw(self):
        self.refresh_momentum()

        theta = self._theta
        rho = self._rho
        lp_theta_rho = self.log_joint(theta, rho)

        approx_cov_rv = self.approximate_covariance_rv(theta)
        inv_mass = approx_cov_rv.rvs()
        # print(f"{inv_mass=}")
        self.set_inv_mass(inv_mass)
        lp_inv_mass = approx_cov_rv.logpdf(inv_mass)

        theta_star, rho_star = self.leapfrog_step(theta, rho)
        lp_theta_rho_star = self.log_joint(theta_star, rho_star)

        approx_cov_rv_star = self.approximate_covariance_rv(theta_star)
        lp_inv_mass_star = approx_cov_rv_star.logpdf(inv_mass)

        log_accept = (
            lp_theta_rho_star + lp_inv_mass_star - (lp_theta_rho + lp_inv_mass)
        )
        self._tries += 1
        if np.log(self._rng.uniform()) < log_accept:
            self._accepts += 1
            self._theta = theta_star
            self._rho = rho_star

        return self._theta, self._rho
