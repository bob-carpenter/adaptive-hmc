import numpy as np

import hmc
import traceback

class GISTV(hmc.HmcSamplerBase):
    """Virial GIST

    Use sign switches of the virial as stopping criterion.
    switch_limit can be greater than 1. Proposals are biased
    towards the end of the trajectory, with biasing analogous to Stan.

    """
    def __init__(self, model, stepsize,
                 theta = None, seed = None,
                 switch_limit = 4, **kwargs):
        super().__init__(model, stepsize, seed = seed)
        self.sampler_name = "GIST-Virial"
        if theta is None:
            self.theta = self.rng.normal(size = self.D)
        else:
            self.theta = theta
        self.switch_limit = switch_limit

        self.steps = 0
        self.prop_accepted = 0.0
        self.draws = 0
        self.divergences = 0

    def virial(self, theta, rho):
        return 2 * rho.dot(theta)

    def logsubexp(self, a, b):
        if a > b:
            return a + np.log1p(-np.exp(b - a))
        elif a < b:
            return b + np.log1p(-np.exp(a - b))
        else:
            return np.inf

    def trajectory(self, theta, rho, lsw = 0.0, switch_discount = 0):
        theta_star = theta
        rho_star = rho
        theta_prop = theta
        rho_prop = rho
        v = self.virial(theta, rho)
        sv = np.sign(v)
        H_0 = self.log_joint(theta, rho)
        switches = 0
        steps = 0
        lsw_segment = -np.inf
        lsw_star = -np.inf
        switches_passed = 0
        while True:
            theta, rho = self.leapfrog_step(theta, rho)
            H = self.log_joint(theta, rho)
            delta = H - H_0
            if np.abs(-delta) > 50.0:
              self.divergences += 1
              break

            v = self.virial(theta, rho)
            if sv * np.sign(v) < 0:
                switches += 1
                sv = np.sign(v)

                log_beta = lsw - lsw_segment
                if np.log(self.rng.uniform()) < np.minimum(0.0, log_beta):
                    theta_star = theta_prop
                    rho_star = rho_prop
                    lsw_star = lsw
                lsw_segment = -np.inf

            lsw_segment = np.logaddexp(lsw_segment, delta)
            lsw = np.logaddexp(lsw, delta)
            log_alpha = delta - lsw
            if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
                theta_prop = theta
                rho_prop = rho
                switches_passed = switches

            if switches >= self.switch_limit - switch_discount:
                break

            steps += 1

        self.steps += steps
        return theta_star, rho_star, lsw_star, lsw, switches_passed

    def draw(self):
        self.draws += 1
        try:
            theta = self.theta
            rho = self.rng.normal(size = self.D)
            H_0 = self.log_joint(theta, rho)

            theta_star, rho_star, lsw_star, FW, switches_passed = self.trajectory(theta, rho)

            _, _, _, BW, _ = self.trajectory(theta, -rho,
                                              lsw = lsw_star,
                                              switch_discount = switches_passed)
            BW = self.logsubexp(BW, 0.0)

            # H_star - H_0 + (H_0 - BW) - (H_star - FW)
            log_alpha = FW - BW

            accepted = 0
            if np.log(self.rng.uniform()) < np.minimum(0.0, log_alpha):
                accepted = 1
                self.theta = theta_star

            self.prop_accepted += (accepted - self.prop_accepted) / self.draws
        except Exception as e:
            # traceback.print_exc()
            pass
        return self.theta
