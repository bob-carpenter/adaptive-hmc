import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import NUTSOrbit as NUTSOrbit
import hmc as hmc


class FixedStepSizeNUTS(hmc.HmcSamplerBase):
    '''
    Implements the No-U-Turn Sampler with a fixed step size.

    The implementation here does NOT bias samples towards the later parts of the trajectory.
    Rather, it samples according to the Boltzmann weights restricted to the NUTS trajectory ("Multinomial NUTS").
    '''
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
        self._max_nuts_depth = max_nuts_depth
        self.observed_heights = []
    def draw(self):
        '''
        Evolves the Markov chain by one step using NUTS with a fixed step size.

        Returns:
            self._theta: np.array(float64): The new position
            self._rho: np.array(float64): The new momentum (not used)
        '''
        self._rho = self._rng.normal(size=self._model.param_unc_num())
        theta, rho = self._theta, self._rho
        bernoulli_sequence = self.refresh_bernoulli_sequence()
        nuts_orbit = self.NUTS(theta, rho, bernoulli_sequence)
        theta_prime, rho_prime = nuts_orbit.sample_coordinates()
        self._theta = theta_prime
        self._rho = rho_prime
        self.observed_heights.append(nuts_orbit._orbit_root._height)
        return self._theta, self._rho

    def NUTS(self, theta, rho, bernoulli_sequence):
        '''
        Draws a sample from NUTS using the fixed step size, theta, rho, and max_height

        Args:
            theta: np.array: Initial position
            rho: np.array: Initial momentum
            bernoulli_sequence: tuple(int): A sequence of 0s and 1s that determines the direction of the
                                NUTS trajectory

        Returns:
            NUTSOrbit sample_orbit: A NUTSOrbit object representing the sample and relevant information
            about the NUTS trajectory
        '''

        sample_orbit = NUTSOrbit.NUTSOrbit(self,
                                           self._rng,
                                           theta,
                                           rho,
                                           self._stepsize,
                                           1,
                                           bernoulli_sequence)
        return sample_orbit

    def refresh_velocity(self):
        '''
        Returns: np.array(float64): momentum vector drawn from a standard normal distribution
        '''
        return self._rng.normal(size=self._model.param_unc_num())

    def refresh_bernoulli_sequence(self):
        '''
        Returns: tuple: 0s and 1s of length self._max_nuts_depth selected
        uniformly at random
        '''

        return tuple(self._rng.binomial(1, 0.5, self._max_nuts_depth))

