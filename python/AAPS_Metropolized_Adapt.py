import numpy as np
import hmc


# Current status: Changing the adapt step size to also return the number of steps
# used over the fine grid AFTER the random multiplier.

class AAPSMetropolizedAdapt(hmc.HmcSamplerBase):
    def __init__(self,
                 model,
                 rng,
                 theta,
                 min_accept_prob,
                 max_step_size,
                 max_step_size_search_depth,
                 apogee_selection_parameter):

        super().__init__(model, 0, rng)
        self._rng = rng
        self._theta = theta
        self._log_min_accept_prob = np.log(min_accept_prob)
        self._max_step_size = max_step_size
        self._max_step_size_search_depth = max_step_size_search_depth
        self._apogee_selection_parameter = apogee_selection_parameter
        self._rejects = 0
        self._no_return_rejects = 0

    def set_stepsize(self, stepsize):
        self._stepsize = stepsize
    def stepsize_reset_original(self):
        self._stepsize = self._max_step_size
    def draw(self):

        coarse_level_interval_current_state = CoarseLevelInterval(self, self._theta, self._rho, self._stepsize)
        #fine_step_size_interval = self.adapt_step_size(coarse_level_interval)
        #proposal_theta, proposal_rho = fine_step_size_interval.proposal()
        #coarse_level_interval_proposal = CoarseLevelInterval(self, proposal_theta, proposal_rho, self._stepsize)
        #fine_step_size_interval_proposal = self.adapt_step_size(coarse_level_interval_proposal)
        #acceptance_probability = self.compute_acceptance_probability(fine_step_size_interval, fine_step_size_interval_proposal)
        self._stepsize = self._max_step_size
        self._number_fine_grid_leapfrog_steps = 1
        # Reset/reinitialize the step size and number of fine grid leapfrog steps

        rho = self._rng.normal(size=self._model.param_unc_num())
        theta = self._theta
        number_apogees = self._rng.geometric(self._apogee_selection_parameter)
        shift = self._rng.integers(0, number_apogees)
        # Resample the momentum, generate random number of apogees, and generate
        # a random shift

        (number_left_initial,
         number_right_initial,
         iterate_numbers_marking_segments_initial) = self.obtain_interval(
            theta, rho, number_apogees, shift
        )

        (log_damping_factor_initial,
         generated_step_size_initial,
         number_intermediate_leapfrog_steps_initial) = self.adapt_step_size(
            theta, rho, number_left_initial, number_right_initial
        )

        (theta_prop,
         rho_prop,
         index,
         coarse_grid_segment_number,
         normalization_factor_initial) = self.generate_proposal(
            theta, rho, generated_step_size_initial, number_intermediate_leapfrog_steps_initial,
            number_left_initial, number_right_initial, iterate_numbers_marking_segments_initial,
            number_apogees, shift
        )

        # Generates the proposal, finds the interval on the coarse grid,
        # adapts the step size, and generates the proposal
        (number_left_prop,
         number_right_prop,
         iterate_numbers_marking_segments_proposal) = self.obtain_interval(
            theta_prop, rho_prop, number_apogees, shift - coarse_grid_segment_number
        )

        (log_damping_factor_proposal,
         generated_step_size_proposal,
         number_intermediate_leapfrog_steps_proposal) = self.adapt_step_size(
            theta_prop, rho_prop, number_left_prop, number_right_prop
        )
        normalization_factor_proposal = self.compute_normalization_factor(
            theta_prop, rho_prop, generated_step_size_proposal, number_intermediate_leapfrog_steps_proposal,
            number_left_prop, number_right_prop
        )

        maximal_step_size_proposal = generated_step_size_proposal/(2 ** log_damping_factor_proposal)
        stepsize_accept_ratio = self.compute_stepsize_accept_ratio(generated_step_size_initial,
                                                                   maximal_step_size_proposal)
        # Compute the step size acceptance ratio
        indicator_index_in_reverse_interval = int((index < number_right_prop) and (index >= number_left_prop))
        index_selection_accept_ratio = ((normalization_factor_proposal / normalization_factor_initial) *
                                        indicator_index_in_reverse_interval)

        # The acceptance criterion for the index is the ratio of the normalizing factors times the
        # indicator that the index is within the interval for the reverse direction
        # I've encoded this indictor into int( condition)
        # If condition is True, this expression evaluates to 1
        # if it is False, then the expression evaluates to 0.
        self._no_return_rejects += (1 - indicator_index_in_reverse_interval)
        acceptance_probability = stepsize_accept_ratio*index_selection_accept_ratio

        if self._rng.uniform() < acceptance_probability:
            theta = theta_prop
            rho = rho_prop
        else:
            self._rejects += 1

        self._theta = theta
        return self._theta, rho
    def obtain_interval(self, theta, rho, number_apogees, shift):

        # ::array -> array -> int -> int -> (int, int, [int])
        # This function implements finding the endpoint in the Apogee to
        # Apogee interval. In principle this should not differ too much for other
        # similar conditions
        self._stepsize = self._max_step_size
        number_steps_forward = 0
        number_steps_backward = 0
        iterate_numbers_marking_segments = [0 for _ in range(number_apogees+1)]
        current_theta = theta
        current_rho = rho
        _, current_grad_u = self._model.log_density_gradient(current_theta)
        current_up_down_status = np.dot(current_rho, current_grad_u)
        number_segments_to_left = shift

        while number_segments_to_left < number_apogees:
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

        # I dont like this code duplication, so I will come back to this later
        # to change it. Right now, I just want to get the code running

        while number_segments_to_left >= 0:
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

        return number_steps_backward, number_steps_forward, iterate_numbers_marking_segments
    def adapt_step_size(self, theta, rho, number_left, number_right):
        # ::array -> array -> int -> int -> (float, float)
        self._stepsize = self._max_step_size
        self._number_fine_grid_leapfrog_steps = 1
        # Reset the adaptation state

        random_multiplier_step_size = 2**self._rng.integers(-1, 2)
        for i in range(self._max_step_size_search_depth):
            energy_max_over_fine_grid, energy_min_over_fine_grid = self.compute_energy_range(theta,
                                                                                             rho,
                                                                                             number_left,
                                                                                             number_right)
            if energy_max_over_fine_grid - energy_min_over_fine_grid < -self._log_min_accept_prob:
                return (i, self._stepsize*random_multiplier_step_size,
                int(self._number_fine_grid_leapfrog_steps/random_multiplier_step_size))
            self._stepsize = self._stepsize/2
            self._number_fine_grid_leapfrog_steps *= 2
        return (self._max_step_size_search_depth - 1,
                self._stepsize*random_multiplier_step_size,
                int(self._number_fine_grid_leapfrog_steps/random_multiplier_step_size))
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
    def generate_proposal(self, theta, rho, step_size, number_intermediate_leapfrog_steps, number_left, number_right,
                          iterate_numbers_marking_segments, number_apogees, shift):
        # ::array -> array -> float -> int -> int -> [int] -> (array, array, int, int, float)
        self._stepsize = step_size
        self._number_fine_grid_leapfrog_steps = number_intermediate_leapfrog_steps
        current_theta = theta
        current_rho = rho
        sample_theta = theta
        sample_rho = rho
        current_energy = -self.log_joint(current_theta, current_rho)
        total_weight = np.exp(-current_energy)
        current_index = 0
        current_coarse_grid_segment_number = 0
        current_segment_left_index = iterate_numbers_marking_segments[shift]
        current_segment_right_index = iterate_numbers_marking_segments[shift+1]

        while current_index < number_right-1:
            # This is very unpythonic
            next_theta, next_rho, *_ = self.iterated_leapfrog_with_energy_max_min(current_theta, current_rho)
            next_energy = -self.log_joint(next_theta, next_rho)
            next_weight = np.exp(-next_energy)
            current_index += 1

            (sample_theta,
             sample_rho) = self.resample_proposal(sample_theta,
                                                  sample_rho,
                                                  current_theta,
                                                  current_rho,
                                                  total_weight,
                                                  next_weight)

            if current_index == current_segment_right_index:
                current_coarse_grid_segment_number += 1
                current_segment_left_index = current_segment_right_index
                current_segment_right_index = iterate_numbers_marking_segments[current_coarse_grid_segment_number + shift]

            current_theta = next_theta
            current_rho = next_rho
            current_energy = next_energy
            total_weight += next_weight

        current_theta = theta
        current_rho = rho
        current_energy = -self.log_joint(current_theta, current_rho)
        current_index = 0
        current_segment_left_index = iterate_numbers_marking_segments[shift]
        current_segment_right_index = iterate_numbers_marking_segments[shift+1]

        while current_index >= number_left:
            next_theta, next_rho, *_ = self.iterated_leapfrog_with_energy_max_min(current_theta, -current_rho)
            next_rho = -next_rho
            next_energy = -self.log_joint(next_theta, next_rho)
            next_weight = np.exp(-next_energy)
            current_index -= 1
            sample_theta, sample_rho = self.resample_proposal(sample_theta,
                                                              sample_rho,
                                                              current_theta,
                                                              current_rho,
                                                              total_weight,
                                                              next_weight)

            if current_index == current_segment_left_index - 1:
                current_coarse_grid_segment_number -= 1
                current_segment_right_index = current_segment_left_index
                current_segment_left_index = iterate_numbers_marking_segments[current_coarse_grid_segment_number + shift]

            current_theta = next_theta
            current_rho = next_rho
            current_energy = next_energy
            total_weight += next_weight

        return sample_theta, sample_rho, current_index, current_coarse_grid_segment_number, total_weight
    def compute_normalization_factor(self, theta,
                                     rho,
                                     generated_step_size_proposal,
                                     num_intermediate_leapfrog_steps,
                                     number_left,
                                     number_right):

        # ::array -> array -> int -> int -> float
        self._stepsize = generated_step_size_proposal
        self._number_fine_grid_leapfrog_steps = num_intermediate_leapfrog_steps
        current_theta = theta
        current_rho = rho
        total_weight = np.exp(-self.log_joint(current_theta, current_rho))

        for i in range(number_right-1):
            next_theta, next_rho, *_ = self.iterated_leapfrog_with_energy_max_min(current_theta, current_rho)
            next_weight = np.exp(-self.log_joint(next_theta, next_rho))
            total_weight += next_weight
            current_theta = next_theta
            current_rho = next_rho

        current_theta = theta
        current_rho = rho
        for i in range(abs(number_left)):
            next_theta, next_rho, *_ = self.iterated_leapfrog_with_energy_max_min(current_theta, -current_rho)
            next_rho = -next_rho
            next_weight = np.exp(-self.log_joint(next_theta, next_rho))
            total_weight += next_weight
            current_theta = next_theta
            current_rho = next_rho

        return total_weight
    def compute_stepsize_accept_ratio(self, step_size, step_size_from_reverse):
        # ::float -> float -> float
        return abs(np.log(step_size/step_size_from_reverse)) <= 1
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
class CoarseLevelIntervalAAPS:
    def __init__(self, sampler, rng, theta, rho, stepsize, apogee_factor):
        self._sampler = sampler
        self._rng = rng
        self._theta = theta
        self._rho = rho
        self._energy_max = -self._sampler.log_joint(self._theta, self._rho)
        self._energy_min = self._energy_max
        self._number_apogees = self._rng.geometric(apogee_factor)
        self._shift = self._rng.integers(0, self._number_apogees)
        self._stepsize = stepsize
        self._total_weight = np.exp(-self._sampler.log_joint(self._theta, self._rho))
        self._iterate_num_segments_left = []
        self._iterate_num_segments_right = []
        self._number_left = 0
        self._number_right = 0
        self._sample_theta = theta
        self._sample_rho = rho
        self._sample_segment_index = 0

        self._sampler.set_stepsize(self._stepsize)
        self.compute_interval()
        self._sampler.stepsize_reset_original()

    def compute_interval(self):
        # ::array -> array -> int -> int -> (int, int, [int])
        # This function implements finding the endpoint in the Apogee to
        # Apogee interval. In principle this should not differ too much for other
        # similar conditions
        self._iterate_num_segments_right = self.iterate_over_coarse_grid(1, self._number_apogees - self._shift)
        self._iterate_num_segments_left = self.iterate_over_coarse_grid(-1, self._shift + 1)
        self._number_left = self._iterate_num_segments_left[-1]
        self._number_right = self._iterate_num_segments_right[-1]
    def iterate_over_coarse_grid(self, direction, number_apogees):
        iterate_numbers_marking_segments = []
        current_theta = self._theta
        current_rho = self._rho
        current_iterate_index = 0
        number_changes_seen = 0
        _, current_grad = self._sampler._model.log_density_gradient(current_theta)
        current_sign = np.dot(current_rho, current_grad)
        while number_changes_seen < number_apogees:
            next_theta, next_rho = self._sampler.leapfrog_step(current_theta, direction*current_rho)
            next_rho = direction*next_rho
            #If the direction is -1, then we are going backwards
            _, next_grad = self._sampler._model.log_density_gradient(next_theta)
            #We're calcluating the gradient twice. In the next
            #version lets either cache the gradient or compute this during
            #the leapfrog step

            next_sign = np.dot(next_rho, next_grad)
            if current_sign * next_sign < 0:
                number_changes_seen += 1
                iterate_numbers_marking_segments.append(current_iterate_index)
            if number_changes_seen == number_apogees:
                break
                #In this case the iterate is the endpoint and should not be
                #considered to be resampled
            current_theta = next_theta
            current_rho = next_rho
            current_sign = next_sign
            current_iterate_index += 1
            current_weight = np.exp(-self._sampler.log_joint(current_theta, current_rho))
            current_energy = -self._sampler.log_joint(current_theta, current_rho)
            self._energy_max = max(self._energy_max, current_energy)
            self._energy_min = min(self._energy_min, current_energy)

            if current_weight/self._total_weight < self._rng.uniform():
                self._sample_theta = current_theta
                self._sample_rho = current_rho
                self._sample_segment_index = number_changes_seen

            self._total_weight += current_weight

        return iterate_numbers_marking_segments

class FineLevelIntervalAAPS:
    def __init__(self, sampler, rng, theta, rho, coarse_grid, depth):
        self._sampler = sampler
        self._rng = rng
        self._theta = theta
        self._rho = rho
        self._coarse_grid = coarse_grid
        self._depth = depth
        self._stepsize = self._coarse_grid._stepsize
        self._number_fine_grid_leapfrog_steps = 2**self._depth
        self._max_energy = -self._sampler.log_joint(self._theta, self._rho)
        self._min_energy = self._max_energy
        self._total_weight = np.exp(-self._sampler.log_joint(self._theta, self._rho))
        self._sample_theta = theta
        self._sample_rho = rho
        self._sample_coarse_grid_segment_index = 0
        self._iterate_numbers_marking_segments = []
    def iterate_over_fine_leapfrog_steps(self):
    def iterated_leapfrog_with_energy_max_min(self, theta, rho):
        # Compute \Phi^{2^{number of stepsize_halvings}} while simultaneously computing the energy
        # and min along the extension
        theta_current = theta
        rho_current = rho
        for i in range(self._number_fine_grid_leapfrog_steps):
            theta_current, rho_current = self._sampler.leapfrog_step(theta_current, rho_current)
            current_energy = -self._sampler.log_joint(theta_current, rho_current)
            self._max_energy = max(self._max_energy, current_energy)
            self._min_energy = min(self._min_energy, current_energy)
        return theta_current, rho_current




