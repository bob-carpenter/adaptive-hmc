import numpy as np
import hmc

class TurnaroundSampler(hmc.HmcSamplerBase):
    """Adaptive HMC algorithm for selecting number of leapfrog steps.

    Uniformly samples number of steps from 1 up to U-turn, flips
    momentum, then balances with reverse proposal probability.
    """
    def __init__(self, model, stepsize, rng,
                     uturn_condition='distance'):
        super().__init__(model, stepsize, rng)
        self._uturn_condition = uturn_condition
        self._cannot_get_back_rejects = 0  # DIAGNOSTIC
        self._fwds = []                    # DIAGNOSTIC
        self._bks = []                     # DIAGNOSTIC

    def uturn(self, theta, rho):
        if self._uturn_condition == 'distance':
            return self.uturn_distance(theta, rho)
        elif self._uturn_condition == 'angle':
            return self.uturn_angle(theta, rho)
        elif self._uturn_condition == 'sym_distance':
            return self.uturn_sym_distance(theta, rho)
        else:
            raise ValueError(f"unknown uturn condition: {self._uturn_condition}")

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
    
    def uturn_angle(self, theta, rho):
        theta_next = theta
        rho_next = rho
        N = 0
        while True:
            theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            N += 1
            if np.dot(rho, rho_next) < 0:
                return N + 1  # N + 1 is point nearer to start
    
    def uturn_sym_distance(self, theta, rho):
        theta_next = theta
        rho_next = rho
        old_distance = 0
        N = 0
        while True:
            if self.uturn_distance(theta_next, -rho_next) < N + 1:
                return N
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
        lps = np.zeros(L)
        pairs = []
        for steps in range(L):
            theta_star, rho_star = self.leapfrog(self._theta, self._rho, steps)
            pairs.append((theta_star, rho_star))
            lps[steps] = self.log_joint(theta_star, rho_star)
        ps = np.exp(lps)
        theta_star, rho_star = np.random.choice(a = pairs, p=ps)
        N1 = self._rng.integers(LB, L)
        theta1_star, rho1_star = self.leapfrog(self._theta, self._rho, N1)
        rho1_star = -rho1_star
        Lstar = self.uturn(theta1_star, rho1_star)
        self._fwds.append(L)                     # DIAGNOSTIC
        self._bks.append(Lstar)                  # DIAGNOSTIC
        if not(LB <= N1 and N1 <= Lstar - 1):
            self._cannot_get_back_rejects += 1   # DIAGNOSTIC
            return self._theta, self._rho        # cannot balance w/o return
        log_accept_prob = ( (self.log_joint(theta1_star, rho1_star) + -np.log(Lstar - 1))
                          - (self.log_joint(theta0, rho0) + -np.log(L - 1) ) )
        if np.log(self._rng.uniform()) < log_accept_prob:
            self._theta = theta1_star
            self._rho = rho1_star
        return self._theta, self._rho

