import numpy as np
import scipy as sp

import hmc


def lognormal_lpdf(y, mu, sigma):
    return sp.stats.lognorm.logpdf(x=y, loc=0, shape=sigma, scale=np.exp(mu))


def lognormal_rng(mu, sigma):
    return rng.lognormal(mean=mu, sigma=sigma)


class TurnaroundSampler(hmc.HmcSamplerBase):
    def __init__(self, model, stepsize, rng, theta_init, steps,
                 stepdown, min_accept_prob, max_retries=5, lognormal_scale=0.5):
        super().__init__(model, stepsize, rng)
        self._theta = theta_init
        self._steps = steps
        self._stepdown = stepdown
        self._min_accept_log_prob = np.log(min_accept_prob)
        self._max_retries = 5
        self._lognormal_scale = 0.5
        self._gradient_evals = 0  # DIAGNOSTIC

    def generate_steps(self, stage):
        median_stepsize = self._stepsize / 2 ** stage
        mu = np.log(median_stepsize)
        sigma = self._lognormal_scale
        stepsize = lognormal_rng(mu, sigma)
        lp_stepsize = lognormal_lpdf(stepsize, mu, sigma)
        return stepsize, lp_stepsize

    def draw(self):
        self._rho = self._rng.normal(size=self._model.param_unc_num())
        steps = self._rng.integers(1, self._steps + 1)
        lp = self.log_joint(theta, rho)
        stepsize = self._stepsize
        for stage in range(self._max_retries):
            stepsize, lp_stepsize = self.generate_steps(stage)
            theta = self._theta
            rho = self._rho
            for _ in range(steps):
                theta, rho = self.leapfrog(theta, rho)
                lp_next = self.log_joint(theta, rho)
                if lp_next - lp >= self._min_accept_log_prob:
                    break

                theta_next
        stepsize = self._rng.lognormal(np.log(self.stepsize

        theta_start = self._theta
        rho_start = self._rho
        log_joint_start = self.log_joint(theta, rho)
        while True:
            steps = np.ceil(self._integration_time / self._stepsize)
            L = self.uturn(theta, rho)
            LB = self.lower_step_bound(L)
            N = self._rng.integers(LB, L)
            theta_star, rho_star = self.leapfrog(theta, rho, N)
            rho_star = -rho_star
            Lstar = self.uturn(theta_star, rho_star)
            LBstar = self.lower_step_bound(Lstar)
            self._fwds.append(L)  # DIAGNOSTIC
            self._bks.append(Lstar)  # DIAGNOSTIC
            if not (LBstar <= N and N < Lstar):
                self._gradient_evals += L  # DIAGNOSTIC
                self._cannot_get_back_rejects += 1  # DIAGNOSTIC
                return self._theta, self._rho  # cannot balance w/o return
            self._gradient_evals += L + Lstar - N  # DIAGNOSTIC
            log_accept = (
                    self.log_joint(theta_star, rho_star) - np.log(Lstar - LBstar)
                    - (log_joint_theta_rho - np.log(L - LB))
            )
            if np.log(self._rng.uniform()) < log_accept:
                self._theta = theta_star
                self._rho = rho_star
        except Exception as e:
        # traceback.print_exc()
        self._divergences += 1

    return self._theta, self._rho
