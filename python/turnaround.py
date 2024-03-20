import numpy as np

class TurnaroundSampler:
    def __init__(self, model, stepsize, rng):
        self._model = model
        self._stepsize = stepsize
        self._rng = rng
        self._theta = self._rng.normal(size=model.param_unc_num())
        self._rho = self._rng.normal(size=model.param_unc_num())
        self._too_short_rejects = 0
        self._fwds = []
        self._bks = []

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
        return self._model.log_density(theta) - 0.5 * sum(rho**2)

    def uturn(self, theta, rho):
        return self.uturn_distance(theta, rho)
    
    def uturn_angle(self, theta, rho):
        theta_next = theta
        rho_next = rho
        N = 0
        while True:
            theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            N += 1
            if np.dot(rho, rho_next) < 0:
                return N + 1  # N + 1 is point nearer to start
    
    def uturn_distance(self, theta, rho):
        theta_next = theta
        rho_next = rho
        old_distance = 0
        N = 0
        while True:
            theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            N += 1
            distance = np.sum((theta_next - theta)**2)
            if distance <= old_distance:
                return N
            old_distance = distance
    
    def draw(self):
        self._rho = self._rng.normal(size=self._model.param_unc_num())
        theta0 = self._theta
        rho0 = self._rho
        L = self.uturn(theta0, rho0)
        N1 = self._rng.integers(1, L)
        theta1_star, rho1_star = self.leapfrog(self._theta, self._rho, N1) # F^(N1)(theta0, rho0)
        rho1_star = -rho1_star # S.F^(N1)(theta0, rho0)

        Lstar = self.uturn(theta1_star, rho1_star)
        self._fwds.append(L)
        self._bks.append(Lstar)
        if Lstar - 1 < N1:
            self._too_short_rejects += 1
            return self._theta, self._rho  # unbalance-able
        
        log_alpha = ( (self.log_joint(theta1_star,rho1_star) + -np.log(Lstar - 1))
                          - (self.log_joint(theta0, rho0) + -np.log(L - 1) ) )
        
        if np.log(self._rng.uniform()) < log_alpha:
            self._theta = theta1_star
            self._rho = rho1_star
        return self._theta, self._rho

    def sample(self, M):
        D = self._model.param_unc_num()
        thetas = np.empty((M, D), dtype=np.float64)
        thetas[0, :] = self._theta
        for m in range(1, M):
            thetas[m, :], _ = self.draw()
        return thetas
