import traceback
from dataclasses import dataclass

import numpy as np

import NUTSOrbit as NUTSOrbit
import hmc as hmc

class StepAdaptNUTSMetro(hmc.HmcSamplerBase):
    def __init__(self,
                 model,
                 rng,
                 theta,
                 rho,
                 min_acceptance_probability,
                 max_stepsize,
                 max_stepsize_search_depth,
                 max_nuts_depth,
                 orbit_class=NUTSOrbit.NUTSOrbit
                 ):
        super().__init__(model, 0, rng)
        self._theta = theta
        self._rho = rho
        self._energy_gap_bound = -np.log(min_acceptance_probability)
        self._max_stepsize = max_stepsize
        self._max_nuts_depth = max_nuts_depth
        self._max_stepsize_search_depth = max_stepsize_search_depth
        self._no_return_rejections = 0
        self._adapted_step_sizes = []
        self._name = "NUTS_b_prime_transform"
        self._acceptance_ratios = []
        self._orbit_constructor = orbit_class

    def draw(self):
        theta, rho = self._theta, self.refresh_velocity()
        bernoulli_sequence = self.refresh_bernoulli_sequence()
        adapted_step_size_from_initial, nuts_orbit = self.adapt_step_size_and_sample(theta, rho, bernoulli_sequence)
        self._adapted_step_sizes.append(adapted_step_size_from_initial._gamma)
        theta_prime, rho_prime = nuts_orbit.sample_coordinates()
        bernoulli_sequence_prime = nuts_orbit.sample_bernoulli_sequence()
        adapted_step_size_from_sample = self.adapt_step_size_without_sample(theta_prime, rho_prime,
                                                                            bernoulli_sequence_prime)
        accept_probability = self.compute_acceptance_probability(adapted_step_size_from_initial,
                                                                 adapted_step_size_from_sample)
        self._acceptance_ratios.append(accept_probability)
        if self._rng.uniform() < accept_probability:
            self._theta = theta_prime
            self._rho = rho_prime

        else:
            self._no_return_rejections += 1
            print(f"Rejected {self._no_return_rejections} times")

        return self._theta, self._rho

    def refresh_velocity(self):
        return self._rng.normal(size=self._model.param_unc_num())

    def refresh_bernoulli_sequence(self):
        return tuple(self._rng.binomial(1, 0.5, self._max_nuts_depth))

    def compute_acceptance_probability(self, adapted_step_size_from_initial, adapted_step_size_from_sample):
        log_likelihood_from_initial = adapted_step_size_from_initial.step_size_log_likelihood(
            adapted_step_size_from_initial)
        log_likelihood_from_sample = adapted_step_size_from_sample.step_size_log_likelihood(
            adapted_step_size_from_initial)
        return np.exp2(log_likelihood_from_sample - log_likelihood_from_initial)

    def adapt_step_size_and_sample(self, theta, rho, bernoulli_sequence):
        for i in range(1, self._max_stepsize_search_depth + 1):
            try:
                stepsize = self._max_stepsize * np.exp2(-i)
                nuts_orbit = self._orbit_constructor(self,
                                                     self._rng,
                                                     theta,
                                                     rho,
                                                     stepsize,
                                                     2 ** i,
                                                     bernoulli_sequence)
                if nuts_orbit.energy_gap() < self._energy_gap_bound:
                    adapted_step_size = AdaptedStepSize.from_h_max_and_gamma(self._max_stepsize,
                                                                             i,
                                                                             self._rng)
                    orbit_with_adapted_step_size = self._orbit_constructor(self,
                                                                           self._rng,
                                                                           theta,
                                                                           rho,
                                                                           adapted_step_size.proposal_step_size(),
                                                                           2 ** adapted_step_size.log_number_intermediate_steps(),
                                                                           bernoulli_sequence)
                    return adapted_step_size, orbit_with_adapted_step_size

            except Exception:
                traceback.print_exc()

        adapted_step_size = AdaptedStepSize.from_h_max_and_gamma(self._max_stepsize,
                                                                 self._max_stepsize_search_depth,
                                                                 self._rng)
        orbit_with_adapted_step_size = self._orbit_constructor(self,
                                                               self._rng,
                                                               theta,
                                                               rho,
                                                               adapted_step_size.proposal_step_size(),
                                                               2 ** adapted_step_size.log_number_intermediate_steps(),
                                                               bernoulli_sequence)
        return adapted_step_size, orbit_with_adapted_step_size

    def adapt_step_size_without_sample(self, theta, rho, bernoulli_sequence):
        for i in range(1, self._max_stepsize_search_depth + 1):
            try:
                stepsize = self._max_stepsize * np.exp2(-i)
                nuts_orbit = self._orbit_constructor(self,
                                                     self._rng,
                                                     theta,
                                                     rho,
                                                     stepsize,
                                                     2 ** i,
                                                     bernoulli_sequence)
                if nuts_orbit.energy_gap() < self._energy_gap_bound:
                    adapted_step_size = AdaptedStepSize.from_h_max_and_gamma(self._max_stepsize,
                                                                             i,
                                                                             self._rng)
                    return adapted_step_size
            except Exception:
                traceback.print_exc()

        adapted_step_size = AdaptedStepSize.from_h_max_and_gamma(self._max_stepsize,
                                                                 self._max_stepsize_search_depth,
                                                                 self._rng)
        return adapted_step_size

    def acceptance_ratios(self):
        return self._acceptance_ratios


@dataclass
class AdaptedStepSize:
    _h_max: float
    _gamma: float
    _random_multiplier: float

    def __post_init__(self):
        self._log_proposal_multiplier = -(self._gamma + self._random_multiplier)

    @classmethod
    def from_h_max_and_gamma(cls, h_max, gamma, rng):
        random_multiplier = int(rng.integers(0, 3) - 1)
        return cls(h_max, gamma, random_multiplier)

    def step_size_log_likelihood(self, proposal_adapted_step_size):
        log_proposal_multiplier = proposal_adapted_step_size._log_proposal_multiplier
        uniform_deviate = log_proposal_multiplier + self._gamma
        if np.abs(uniform_deviate) <= 1.0:
            return 0.0
        print(f"uniform_deviate was {uniform_deviate}")
        print("Returning -np.inf")
        return -np.inf

    def proposal_step_size(self):
        return self._h_max * np.exp2(self._log_proposal_multiplier)

    def log_number_intermediate_steps(self):
        return int(self._gamma + self._random_multiplier)
