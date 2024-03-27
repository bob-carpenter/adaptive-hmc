import numpy as np
import hmc
import traceback

class TurnaroundSampler(hmc.HmcSamplerBase):
    """Adaptive HMC algorithm for selecting number of leapfrog steps.

    Uniformly samples number of steps from 1 up to U-turn, flips
    momentum, then balances with reverse proposal probability.
    """
    def __init__(self, model, stepsize, rng, uturn_condition, path_fraction,
                     max_leapfrog = 512):
        super().__init__(model, stepsize, rng)
        self._uturn_condition = uturn_condition
        self._path_fraction = path_fraction
        self._max_leapfrog_steps = max_leapfrog
        self._cannot_get_back_rejects = 0  # DIAGNOSTIC
        self._fwds = []                    # DIAGNOSTIC
        self._bks = []                     # DIAGNOSTIC
        self._divergences = 0              # DIAGNOSTIC

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
        for _ in range(self._max_leapfrog_steps):
            try:
                theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            except Exception as e:
                self._divergences += 1
                return N + 1
            N += 1
            distance = np.sum((theta_next - theta)**2)
            if distance <= old_distance:
                return N
            old_distance = distance
        return self._max_leapfrog_steps
    
    def uturn_angle(self, theta, rho):
        theta_next = theta
        rho_next = rho
        N = 0
        for _ in range(self._max_leapfrog_steps):
            try:
                theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            except Exception as e:
                self._divergences += 1
                return N + 1
            N += 1
            if np.dot(rho, rho_next) < 0:
                return N + 1
        return self._max_leapfrog_steps
    
    def uturn_sym_distance(self, theta, rho):
        theta_next = theta
        rho_next = rho
        old_distance = 0
        N = 0
        for _ in range(self._max_leapfrog_steps):
            try:
                if self.uturn_distance(theta_next, -rho_next) < N + 1:
                    return N
                theta_next, rho_next = self.leapfrog_step(theta_next, rho_next)
            except Exception as e:
                self._divergences += 1
                return N + 1
            N += 1
            distance = np.sum((theta_next - theta)**2)
            if distance <= old_distance:
                return N
            old_distance = distance
        return self._max_leapfrog_steps
     
    def lower_step_bound(self, L):
        if self._path_fraction == 'full':
            return 1
        elif self._path_fraction == 'half':
            return L // 2
        elif self._path_fraction == 'quarter':
            return 3 * L // 4
        else:
            raise ValueError(f"unknown path fraction: {self._path_fraction}")

    def draw(self):
        try:
            self._rho = self._rng.normal(size=self._model.param_unc_num())
            theta = self._theta
            rho = self._rho
            L = self.uturn(theta, rho)
            LB = self.lower_step_bound(L)
            N = self._rng.integers(LB, L)
            theta_star, rho_star = self.leapfrog(theta, rho, N)
            rho_star = -rho_star
            Lstar = self.uturn(theta_star, rho_star)
            LBstar = self.lower_step_bound(Lstar)
            self._fwds.append(L)                     # DIAGNOSTIC
            self._bks.append(Lstar)                  # DIAGNOSTIC
            if not(LBstar <= N and N < Lstar):
                self._cannot_get_back_rejects += 1   # DIAGNOSTIC
                return self._theta, self._rho        # cannot balance w/o return
            log_accept = (
                self.log_joint(theta_star, rho_star) - np.log(Lstar - LBstar)
                - (self.log_joint(theta, rho) - np.log(L - LB))
            )
            if np.log(self._rng.uniform()) < log_accept:
                self._theta = theta_star
                self._rho = rho_star
        except Exception as e:
            print("shouldn't get here")
            traceback.print_exc()
            self._divergences += 1
            return self._theta, self._rho
        return self._theta, self._rho

