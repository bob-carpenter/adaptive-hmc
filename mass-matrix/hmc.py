import numpy as np
import traceback

class HmcSamplerBase:
    def __init__(self, model, stepsize, rng, theta, rho, inv_mass_matrix):
        self._model = model
        self._stepsize = stepsize
        self._rng = rng
        self._theta = theta
        self._rho = rho
        self._ZEROS = np.zeros(model.param_unc_num())
        self.set_inv_mass(inv_mass_matrix if inv_mass_matrix is not None else np.eye(model.param_unc_num()))
        
    def __iter__(self):
        return self

    def __next__(self):
        return self.safe_draw()

    def set_inv_mass(self, inv_mass_matrix):
        self._inv_mass = inv_mass_matrix
        self._mass = np.linalg.inv(inv_mass_matrix)

    def set_mass(self, mass_matrix):
        self._mass = mass_matrix
        self._inv_mass = np.linalg.inv(mass_matrix)
    
    def refresh_momentum(self):
        self._rho = self._rng.multivariate_normal(self._ZEROS, self._mass)
    
    def safe_draw(self):
        theta, rho = self._theta, self._rho
        try:
            return self.draw()
        except Exception as e:
            print(f"EXCEPTION in HmcSamplerBase.safe_draw(): {e}")
            traceback.print_exc()
            self._theta, self._rho = theta, rho
            return self._theta, self._rho

    def leapfrog_step(self, theta, rho):
        _, grad = self._model.log_density_gradient(theta)
        rho2 = rho + 0.5 * self._stepsize * grad
        theta2 = theta + self._stepsize * (self._inv_mass @ rho2)
        _, grad2 = self._model.log_density_gradient(theta2)
        rho2 += 0.5 * self._stepsize * grad2
        return theta2, rho2

    # TODO(bob-carpenter): make this more efficient by composing
    def leapfrog(self, theta, rho, numsteps):
        for _ in range(numsteps):
            theta, rho = self.leapfrog_step(theta, rho)
        return theta, rho

    def log_joint(self, theta, rho):
        try:
            return self._model.log_density(theta) - 0.5 * (rho.T @ self._inv_mass @ rho)
        except ExceptionType as e:
            print(f"hmc.log_joint() exception: {e}")
            return np.NINF

    def sample(self, M):
        D = self._model.param_unc_num()
        thetas = np.empty((M, D), dtype=np.float64)
        thetas[0, :] = self._theta
        for m in range(1, M):
            thetas[m, :], _ = self.safe_draw()
        return thetas

    def sample_constrained(self, M):
        D = self._model.param_num()
        thetas = np.empty((M, D), dtype=np.float64)
        thetas[0, :] = self._model.param_constrain(self._theta)
        for m in range(1, M):
            theta_m, _ = self.safe_draw()
            thetas[m, :] = self._model.param_constrain(theta_m)
        return thetas
