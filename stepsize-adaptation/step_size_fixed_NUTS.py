import NUTSOrbit
import hmc


class FixedStepSizeNUTS(hmc.HmcSamplerBase):
    def __init__(self,
                 model,
                 rng,
                 theta,
                 rho,
                 stepsize,
                 max_nuts_depth):
        super().__init__(model, stepsize, rng)
        self._theta = theta
        self._rho = rho
        self._max_nuts_search_depth = max_nuts_depth

    def draw(self):
        self._rho = self._rng.normal(size=self._model.param_unc_num())
        theta, rho = self._theta, self._rho
        nuts_orbit = self.NUTS(theta, rho, self._max_nuts_search_depth)
        theta_prime, rho_prime = nuts_orbit.sample_coordinates()
        self._theta = theta_prime
        self._rho = rho_prime
        return theta_prime, rho_prime

    def NUTS(self, theta, rho, max_height):
        bernoulli_sequence = tuple(self._rng.binomial(1, 0.5, max_height))
        sample_orbit = NUTSOrbit.NUTSOrbit(self,
                                           self._rng,
                                           theta,
                                           rho,
                                           self._stepsize,
                                           1,
                                           bernoulli_sequence)
        return sample_orbit
