import traceback
from dataclasses import dataclass

import numpy as np

import NUTSOrbit as NUTSOrbit
import hmc as hmc

class StepAdaptNUTSMetro(hmc.HmcSamplerBase):
    '''
    Implements the adaptNUTS algorithm for step size adaptive NUTS with Metropolis correction.
    '''
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
        '''
        Evolves the Markov chain by one step using adaptNUTS.

        Returns:
            self._theta: np.array(): New position of the Markov chain.
            self._rho: np.array(): New momentum of the Markov chain.
        '''
        theta, rho = self._theta, self.refresh_velocity()
        bernoulli_sequence = self.refresh_bernoulli_sequence()
        adapted_step_size_from_initial, nuts_orbit = self.adapt_step_size_and_sample(theta, rho, bernoulli_sequence)
        self._adapted_step_sizes.append(adapted_step_size_from_initial._num_step_size_halvings)
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
        '''
        Returns: np.array(): Random velocity vector sampled from a standard normal distribution.
        '''
        return self._rng.normal(size=self._model.param_unc_num())

    def refresh_bernoulli_sequence(self):
        '''
        Returns: tuple: 0s and 1s of length self._max_nuts_depth selected
        uniformly at random
        '''
        return tuple(self._rng.binomial(1, 0.5, self._max_nuts_depth))

    def compute_acceptance_probability(self, adapted_step_size_from_initial, adapted_step_size_from_sample):
        '''
        Computes the acceptance probability for the Metropolis correction for the adapted step size.
        Args:
            adapted_step_size_from_initial: AdaptedStepSize: Object representing adapted step size from the initial state.
                    In this algorithm, plays no role in the acceptance probability calculation.
                    But it is included for completeness as it is relevant for other versions.

            adapted_step_size_from_sample:AdaptedStepSize: Adapted step size object from the sample state

        Returns:
            float: The acceptance probability for the Metropolis correction
        '''
        log_likelihood_from_initial = adapted_step_size_from_initial.step_size_log_likelihood(
            adapted_step_size_from_initial)
        log_likelihood_from_sample = adapted_step_size_from_sample.step_size_log_likelihood(
            adapted_step_size_from_initial)
        return np.exp2(log_likelihood_from_sample - log_likelihood_from_initial)

    def adapt_step_size_and_sample(self, theta, rho, bernoulli_sequence):
        '''
        Adapts the step size and samples from the NUTS orbit.
        Args:
            theta: np.array(): Initial position
            rho: np.array(): Initial momentum
            bernoulli_sequence: tuple: Sequence of 0s and 1s that determines the direction of the NUTS trajectory

        Returns:
            adapted_step_size: AdaptedStepSize: The adapted step size object representing the step size
                adaptatiion from the initial state. Used for Metropolis correction.

            orbit_with_adapted_step_size: NUTSOrbit: The NUTSOrbit object representing the sample
                and relevant information

        '''
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
                    adapted_step_size = AdaptedStepSize.from_h_max_and_halvings(self._max_stepsize,
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

        adapted_step_size = AdaptedStepSize.from_h_max_and_halvings(self._max_stepsize,
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
        '''
        Adapts the step size without sampling from the NUTS orbit. Used for the Metropolis correction from the
            sample.

        Args:
            theta: np.array(): Initial position (the sample position in this context)
            rho: np.array(): Initial momentum (the sample momentum in this context)
            bernoulli_sequence: tuple(int): Sequence of 0s and 1s that determines the direction of the NUTS trajectory
                (the transformed sequence B^* in this context)
        Returns:
            AdaptedStepSize: The adapted step size object representing the step size adaptation from the sample state.

        '''
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
                    adapted_step_size = AdaptedStepSize.from_h_max_and_halvings(self._max_stepsize,
                                                                                i,
                                                                                self._rng)
                    return adapted_step_size
            except Exception:
                traceback.print_exc()

        adapted_step_size = AdaptedStepSize.from_h_max_and_halvings(self._max_stepsize,
                                                                    self._max_stepsize_search_depth,
                                                                    self._rng)
        return adapted_step_size

    def acceptance_ratios(self):
        '''
        Returns: List of acceptance ratios for the Metropolis correction. Used for diagnostics.
        '''
        return self._acceptance_ratios


@dataclass
class AdaptedStepSize:
    '''
    Represents the adapted step size in the step size adaptation algorithm.

    Attributes:
        _h_max: float: The maximum step size
        _num_step_size_halvings: int: The number of halvings in the step size adaptation. This would be \texttt{step_reduction}
                in the companion paper.
        _log_random_multiplier: int: Integer in {-1, 0, 1} representing the random multiplier for the step size adaptation
        _log_proposal_multiplier: int: The combines the gamma and the random multiplier for the step size adaptation.
            The proposal step size is h_max * 2^log_proposal_multiplier
    '''
    _h_max: float
    _num_step_size_halvings: float
    _log_random_multiplier: float

    def __post_init__(self):
        '''
        Initializes the log proposal multiplier for the step size adaptation from the gamma and the random multiplier.
        Returns: None
        '''
        self._log_proposal_multiplier = -(self._num_step_size_halvings + self._log_random_multiplier)

    @classmethod
    def from_h_max_and_halvings(cls, h_max, num_halvings, rng):
        '''
        Initializes the class from the maximum step size, the number of halvings, and a random number generator.
        Args:
            h_max: float: Max step size
            num_halvings: int: Number of step size halvings
            rng: numpy rng object: Random number generator

        Returns: AdaptedStepSize object

        '''
        random_multiplier = int(rng.integers(0, 3) - 1)
        return cls(h_max, num_halvings, random_multiplier)

    def step_size_log_likelihood(self, proposal_adapted_step_size):
        '''
        Computes the log likelihood of a proposal step size given the current step size adaptation.
        Args:
            proposal_adapted_step_size: AdaptedStepSize: Proposal step size AdaptedStepSize object.
            We compute the log likelihood of this proposal given the current step size adaptation.

        Returns:
        '''
        log_proposal_multiplier = proposal_adapted_step_size._log_proposal_multiplier
        uniform_deviate = log_proposal_multiplier + self._num_step_size_halvings
        if np.abs(uniform_deviate) <= 1.0:
            return 0.0
        print(f"uniform_deviate was {uniform_deviate}")
        print("Returning -np.inf")
        return -np.inf

    def proposal_step_size(self):
        '''
        Computes the proposal step size from the adapted step size.
        Returns:
            float: The proposal step size

        '''
        return self._h_max * np.exp2(self._log_proposal_multiplier)

    def log_number_intermediate_steps(self):
        '''
        Computes the log number of intermediate steps in the adaptNUTS orbit from the adapted step size.
        Returns:
            int: The log number of intermediate steps
        '''
        return int(self._num_step_size_halvings + self._log_random_multiplier)
