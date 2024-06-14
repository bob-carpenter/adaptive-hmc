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
        self._log_min_accept_prob = np.log(min_accept_prob)
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
        self._stepsize  = self._max_step_size
        # ::array -> array -> int -> int -> (int, int, [int])
        #This function implements finding the endpoint in the Apogee to
        #Apogee interval. In principle this should not differ too much for other
        #similar conditions
        number_steps_forward = 0
        number_steps_backward = 0
        iterate_numbers_marking_segments = [0 for _ in range(number_apogees+1)]
        current_theta = theta
        current_rho = rho
        _, current_grad_u = self._model.log_density_gradient(current_theta)
        current_up_down_status = np.dot(current_rho, current_grad_u)
        number_segments_to_left = shift

        while(number_segments_to_left < number_apogees):
            number_steps_forward += 1
            next_theta, next_rho = self.leapfrog_step(current_theta, current_rho)
            _, next_grad_u = self._model.log_density_gradient(next_theta)
            next_up_down_status = np.dot(next_rho, next_grad_u)
            if (current_up_down_status * next_up_down_status) < 0:
                number_segments_to_left += 1
                iterate_numbers_marking_segments[number_segments_to_left] = number_steps_forward
            current_theta = next_theta
            current_rho = next_rho
            current_up_down_status = next_up_down_status

        current_theta = theta
        current_rho = rho
        _, current_grad_u = self._model.log_density_gradient(current_theta)
        current_up_down_status = np.dot(current_rho, current_grad_u)
        number_segments_to_left = shift

        #I dont like this code duplication, so I will come back to this later
        #to change it. Right now, I just want to get the code running

        while(number_segments_to_left => 0):
            number_steps_backward -= 1
            next_theta, next_rho = self.leapfrog_step(current_theta, -current_rho)
            next_rho = -next_rho
            _, next_grad_u = self._model.log_density_gradient(next_theta)
            next_up_down_status = np.dot(next_rho, next_grad_u)
            if (current_up_down_status * next_up_down_status) < 0:
                iterate_numbers_marking_segments[number_segments_to_left] = number_steps_backward + 1
                number_segments_to_left -= 1
            current_theta = next_theta
            current_rho = next_rho
            current_up_down_status = next_up_down_status

        return (number_steps_backward, number_steps_forward, iterate_numbers_marking_segments)

    def adapt_step_size(self, theta, rho, number_left, number_right):
        # ::array -> array -> int -> int -> (float, float)
        random_multiplier = 2**self._rng.integers(-1,2)

        for i in range(self._max_step_size_search_depth):
            energy_max_over_fine_grid, energy_min_over_fine_grid = self.compute_energy_range(theta,
                                                                                         rho,
                                                                                         number_left,
                                                                                         number_right)
            if energy_max_over_fine_grid - energy_min_over_fine_grid < -self._log_min_accept_prob:
                return i, self._stepsize*random_multiplier
            self._stepsize = self._stepsize/2
            self._number_fine_grid_leapfrog_steps *= 2
        return i, self._stepsize*random_multiplier
    def compute_energy_range(self, theta, rho, number_left, number_right):
        max_energy = min_energy = -self.log_joint(theta, rho)
        theta_current = theta
        rho_current = rho

        for i in range(number_right-1):
            (theta_current,
             rho_current,
             max_over_fine,
             min_over_fine) = self.iterated_leapfrog_with_energy_max_min(theta_current, rho_current)
            max_energy = max(max_energy, max_over_fine)
            min_energy = min(min_energy, min_over_fine)

        theta_current = theta
        rho_current = rho

        for i in range(abs(number_left)):
            (theta_current,
             rho_current,
             max_over_fine,
             min_over_fine) = self.iterated_leapfrog_with_energy_max_min(theta_current, -rho_current)
            rho_current = -rho_current
            max_energy = max(max_energy, max_over_fine)
            min_energy = min(min_energy, min_over_fine)
        return max_energy, min_energy

    def generate_proposal(self, theta, rho, step_size, number_intermediate_steps, number_left, number_right, iterate_numbers_marking_segments):
        current_theta = theta
        current_rho = rho
        current_weight = current_energy = -self.log_joint(current_theta, current_rho)

    def resample_proposal(self, theta_current, rho_current, theta_new, rho_new, weight_current, weight_new):
        # ::array -> array -> array -> array -> float -> float -> (array, array, float)
        if self._rng.uniform() < weight_new/(weight_current + weight_new):
            return theta_new, rho_new
        return theta_current, rho_current
    def compute_normalization_factor(self, theta, rho, number_left, number_right):
    def stepsize_accept_ratio(self, log_damping_factor_initial, log_dampling_factor_prop):

    def iterated_leapfrog_with_energy_max_min(self, theta, rho):
        # Compute \Phi^{2^{number of stepsize_halvings}} while simultaneously computing the energy
        # and min along the extension
        max_energy = min_energy = -self.log_joint(theta, rho)
        theta_current = theta
        rho_current = rho

        for i in range(self._number_fine_grid_leapfrog_steps):
            theta_current, rho_current = self.leapfrog_step(theta_current, rho_current)
            current_energy = -self.log_joint(theta_current, rho_current)
            max_energy = max(max_energy, current_energy)
            min_energy = min(min_energy, current_energy)

        return theta_current, rho_current, max_energy, min_energy
