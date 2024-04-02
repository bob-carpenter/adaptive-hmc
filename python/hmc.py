import numpy as np

class HmcSamplerBase:
    def __init__(self, model, stepsize, rng):
        self._model = model
        self._stepsize = stepsize
        self._rng = rng
        self._theta = self._rng.normal(size=model.param_unc_num())
        self._rho = self._rng.normal(size=model.param_unc_num())

    def __iter__(self):
        return self

    def __next__(self):
        return self.draw()

    def leapfrog_step(self, theta, rho):
        _, grad = self._model.log_density_gradient(theta)
        rho2 = rho + 0.5 * self._stepsize * grad
        theta2 = theta + self._stepsize * rho2
        _, grad = self._model.log_density_gradient(theta2)
        rho2 += 0.5 * self._stepsize * grad
        return theta2, rho2

    def leapfrog(self, theta, rho, numsteps):
        for _ in range(numsteps):
            theta, rho = self.leapfrog_step(theta, rho)
        return theta, rho
    
    def log_joint(self, theta, rho):
        try:
            return self._model.log_density(theta) - 0.5 * sum(rho**2)
        except ExceptionType as e:
            return np.NINF

    # def draw(self):
    #     self._rho = self._rng.normal(size=self._model.param_unc_num())
    #     logp = self.log_joint(self._theta, self._rho)
    #     theta_prop, rho_prop = self.leapfrog(self._theta, self._rho)
    #     rho_prop = -rho_prop
    #     logp_prop = self.log_joint(theta_prop, rho_prop)
    #     if np.log(self._rng.uniform()) < logp_prop - logp:
    #         self._theta = theta_prop
    #         self._rho = rho_prop
    #     return self._theta, self._rho

    def sample(self, M):
        D = self._model.param_unc_num()
        thetas = np.empty((M, D), dtype=np.float64)
        thetas[0, :] = self._theta;
        for m in range(1, M):
            thetas[m, :], _ = self.draw()
        return thetas

    def sample_constrained(self, M):
        D = self._model.param_num()
        thetas = np.empty((M, D), dtype=np.float64)
        thetas[0, :] = self._model.param_constrain(self._theta)
        for m in range(1, M):
            theta_m, _ = self.draw()
            thetas[m, :] = self._model.param_constrain(theta_m)
        return thetas
