import hmc
from scipy.stats import wishart
import numpy as np

# This works to generate a covariance matrix near S
# p = wishart(df = 1000, scale=S / 1000)

class GistMassMalaSampler(hmc.HmcSamplerBase):
    def __init__(self, model, stepsize, rng, theta, dof = 2):
        super().__init__(model, stepsize, rng, theta)
        self._epsilon_vec = 1e-6 * np.ones(model.param_unc_num())
        self._dof = dof
        
    def approx_inv_mass(self, theta):
        _, grad = self._model.log_density_gradient(theta)
        approx_inv_mass = -(theta @ theta.T)
        np.fill_diagonal(approx_inv_mass, approx_inv_mass.diagonal() + self._epsilon_vec)
        return approx_inv_mass
        
    def draw(self):
        try:
            theta = self._theta
            approx_inv_mass = self.approx_mass(theta)
            fwd_wishart = wishart(df = self._dof, approx_inv_mass)
            inv_mass = fwd_wishart.rvs()
            lp_wishart = fwd_wishart.logpdf(inv_mass)
            set_inv_mass(inv_mass)
            refresh_momentum()
            theta = self._theta
            rho = self._rho
            log_joint_theta_rho = self.log_joint(theta, rho)
            theta_star, rho_star = self.leapfrog_step(theta, rho)
            approx_rev_inv_mass = approx_inv_mass(theta_star)
            lp_rev_wishart = wishart(df = self._dof, approx_rev_inv_mass).logpdf(inv_mass)
            log_joint_theta_rho_star = self.log_joint(theta_star, rho_star)
            log_accept = (
                log_joint_theta_rho_star + lp_rev_wishart
                - (log_joint_theta_rho + lp_wishart)
            )
            if np.log(self._rng.uniform()) < log_accept:
                self._theta = theta_star
                self._rho = rho_star
        except Exception as e:
            print(f"{GistMassMalaSampler.draw() exception: {e}")
        return self._theta, self._rho
