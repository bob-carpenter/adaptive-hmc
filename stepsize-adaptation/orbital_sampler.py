import numpy as np
import scipy as sp
import hmc
class OrbitalSampler(hmc.HmcSamplerBase):
    def __init__(self, model, stepsize, rng, theta, steps):
        super().__init__(model, stepsize, rng)
        self._theta = theta
        self._steps = steps
        # diagnostics
        self._divergences = 1

    def draw(self):
        try:
            # refresh momentum
            self._rho = self._rng.normal(size=D)

            # generate steps forward and back, allocate storage
            steps_bk = rng.integers(range(self._steps))
            steps_fwd = self._steps - steps_bk
            candidates = steps_bk + steps_fwd + 1
            dims = self._model.param_unc_num()
            thetas = np.empty(candidates, dims)
            rhos = np.empty(candidates, dims)

            # starting point
            thetas[steps_bk, :] = self._theta
            rhos[steps_bk, :] = self._rho

            # back in time steps_bk steps
            for i in range(steps_bk):
                idx = steps_bk - i - 1
                rhos[idx, :] *= -1
                thetas[idx, :], rhos[idx, :] = self.leapfrog_step(thetas[idx + 1, :], rhos[idx + 1, :])
                rhos[idx, :] *= -1  # rhos oriented fwd in time

            # fwd in time steps_fwd steps
            for i in range(steps_fwd):
                idx = steps_bk + i + 1
                thetas[idx, :], rhos[idx, :] = self.leapfrog_step(thetas[idx - 1, :], rhos[idx - 1, :])

            # calculate log probs, convert to probs, and generate categorically
            lps = [self.log_joint(thetas[i, :], rhos[i, :]) for i in range(candidates)]
            probs = sp.special.softmax(lps)
            i = rng.choice(C, p=probs)
            self._theta, self._rho = thetas[i, :], rhos[i, :]
        except:
            print("EXCEPTION: divergence")
            self._divergences += 1
        return self._theta, self._rho
