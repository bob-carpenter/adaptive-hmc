import numpy as np
import scipy as sp


class NegArray():
    def __init__(self, size):
        self._size = size
        self._vals = np.empty(size * 2)
    def get(n):
        return self._vals[self._size + n]
    def set(n, x)
        self._vals[self._size + n] = x
        
class UTurnCategoricalSampler():
    def __init__(self, model, stepsize, seed = None, theta0 = None, rho0 = None):
        self._model = model
        self._stepsize = stepsize
        self._rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        self._theta = theta0 if theta0 is not None else self._rng.normal(size=model.dims())
        self._rho = rho0 if rho0 is not None else self._rng.normal(size=model.dims())

    def __iter__(self):
        return self

    def __next__(self):
        return self.draw()

    def joint_logp(self, theta, rho):
        return self._model.log_density(theta) - 0.5 * np.dot(rho, rho)

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

    def max_steps_before_uturn(self, theta, rho):
        theta_next = theta
        rho_next = rho
        last_dist = 0
        N = 0
        while True:
            theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            dist = np.sum((theta_next - theta)**2)
            if dist <= last_dist:
                return N
            last_dist = dist
            N += 1


    def draw(self):
        # refresh momentum
        self._rho = self._rng.normal(size=self._model.dims())
        N = self.max_steps_before_uturn(self._theta, self._rho)
        lp = np.empty(N + 1)
        theta = self._theta
        rho = self._rho
        lp[0] = self.joint_logp(theta, rho)
        for n in range(1, N + 1):
            theta, rho = self.leapfrog_step(theta, rho)
            lp[n] = self.joint_logp(theta, rho)
        p = sp.special.softmax(lp)
        L = self._rng.choice(N + 1, p)
        theta_star, neg_rho_star = self.leapfrog(theta, rho, n)
        lp_jump = lp[L] - log_sum_exp(lp)
        rho_star = -neg_rho_star
        M = self.max_steps_before_uturn(theta_star, rho_star)
        lp_back = np.empty(M + 1)
        lp_back[0] = self.joint_logp(theta_star, rho_star)
        theta = theta_star
        rho = rho_star
        for m in range(1, M + 1):
            theta, rho = self.leapfrog_step(theta, rho)
            lp_back[m] = self.joint_logp(theta, rho)
        lp_zero = lp[0]
        lp_star = lp[L]
        lp_jump_back = lp_back[L] - log_sum_exp(lp_back)
        log_p_accept = (lp_star + lp_jump_back) - (lp_zero + lp_jump)
        alpha = self._rng.uniform(0, 1)
        if np.log(alpha) < log_p_accept:
            return theta_star, rho_star
        return self._theta, self._rho
