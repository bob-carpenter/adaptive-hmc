import hmc
import numpy as np

class EuclideanMala(hmc.HmcSamplerBase):
    def __init__(self, model, stepsize, rng, theta, mass_matrix):
        super().__init__(model, stepsize, rng, theta, theta, mass_matrix)
        self.set_mass(mass_matrix)

    def draw(self):
        self.refresh_momentum()
        theta = self._theta
        rho = self._rho
        lp_theta_rho = self.log_joint(theta, rho)

        theta_star, rho_star = self.leapfrog_step(theta, rho)
        lp_theta_rho_star = self.log_joint(theta_star, rho_star)

        log_accept = lp_theta_rho_star - lp_theta_rho
        if np.log(self._rng.uniform()) < log_accept:
            self._theta = theta_star
            self._rho = rho_star
        return self._theta, self._rho
        

