import numpy as np
import hmc


class AAPSMetropolizedAdapt(hmc.HmcSamplerBase):
    def __init__(self,
                 model,
                 rng,
                 theta,
                 min_accept_prob,
                 max_step_size,
                 max_step_size_search_depth,
                 apogee_factor):

        super.__init__(self, model, 0, rng)
        self._rng = rng
        self._theta = theta
        self._min_accept_prob = min_accept_prob
        self._max_step_size = max_step_size
        self._max_step_size_search_depth = max_step_size_search_depth
        self._number_fine_grid_leapfrog_steps = 1
        self._apogee_factor = apogee_factor
        self._rejects = 0
        self._no_return_rejects = 0

    def draw(self):
        self._stepsize = self._max_step_size
        self._number_fine_grid_leapfrog_steps = 1
        #Reset/reinitialize the step size and number of fine grid leapfrog steps

        rho = self._rng.normal(size=self._model.param_unc_num())
        theta = self._theta
        number_apogees = self._rng.geometric(self._apogee_factor)
        shift = self._rng.integers(0, number_apogees)
        #Resample the momentum, generate random number of apogees, and generate
        # a random shift

        (number_left_initial,
         number_right_initial,
         iterate_numbers_marking_segments_initial) = self.obtain_interval(
            theta, rho, number_apogees, shift
        )

        log_damping_factor_initial, generated_step_size_initial = self.adapt_step_size(
            theta, rho, number_left_initial, number_right_initial
        )

        (theta_prop,
         rho_prop,
         index,
         coarse_grid_segment_number,
         normalization_factor_initial) = self.generate_proposal(
            theta, rho, generated_step_size_initial,
            number_left_initial, number_right_initial, iterate_numbers_marking_segments_initial
        )

        # Generates the proposal, finds the interval on the coarse grid,
        # adapts the step size, and generates the proposal
        (number_left_prop,
         number_right_prop,
         iterate_numbers_marking_segments_proposal)  = self.obtain_interval(
            theta_prop, rho_prop, number_apogees, shift - coarse_grid_segment_number
        )

        log_damping_factor_proposal, generated_step_size_proposal = self.adapt_step_size(
            theta_prop, rho_prop, number_left_prop, number_right_prop
        )
        normalization_factor_proposal = self.compute_normalization_factor(
            theta_prop, rho_prop, generated_step_size_proposal, number_left_prop, number_right_prop
        )

        stepsize_accept_ratio = self.stepsize_accept_ratio(log_damping_factor_initial, log_damping_factor_proposal)
        # Compute the step size acceptance ratio
        indicator_index_in_reverse_interval = int((index < = number_right_prop) and (index >= number_left_prop))
        index_selection_accept_ratio = ((normalization_factor_proposal / normalization_factor_initial)*
                                        indicator_index_in_reverse_interval)

        # The acceptance criterion for the index is the ratio of the normalizing factors times the
        # indicator that the index is within the interval for the reverse direction
        # I've encoded this indictor into int( condition)
        # If condition is True, this expression evaluates to 1
        # if it is False, then the expression evaluates to 0.
        self._no_return_rejects += (1 - indicator_index_in_reverse_interval)
        acceptance_probability = stepsize_accept_ratio*index_selection_accept_ratio

        if self._rng.uniform() < acceptance_probability:
            theta_prop = theta_prop
            rho = rho_prop
        else:
            self._rejects += 1

        self._theta = theta
        return self._theta, rho

    def obtain_interval(self, theta, rho, number_apogees, shift):
        # ::array -> array -> int -> int -> (int, int, [int])
        #This function implements finding the endpoint in the Apogee to
        #Apogee interval. In principle this should not differ too much for other
        #similar conditions

    def adapt_step_size(self, theta, rho, number_left, number_right):
    def generate_proposal(self, theta, rho, h_star, number_left, number_right):
    def compute_normalization_factor(self, theta, rho, number_left, number_right):
    def stepsize_accept_ratio(self, log_damping_factor_initial, log_dampling_factor_prop):