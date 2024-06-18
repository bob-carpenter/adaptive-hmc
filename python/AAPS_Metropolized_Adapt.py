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
        self._step_size_non_overlap_rejects = 0

    def set_stepsize(self, stepsize):
        self._stepsize = stepsize
    def stepsize_reset_original(self):
        self._stepsize = self._max_step_size
    def draw(self):
        rho = self._rng.normal(size=self._model.param_unc_num())
        theta = self._theta
        number_apogees = self._rng.geometric(self._apogee_selection_parameter)
        shift = self._rng.integers(0, number_apogees)
        coarse_level_interval_current_state = CoarseLevelIntervalAAPS(self,
                                                                  self._rng,
                                                                  self._theta,
                                                                  self._rho,
                                                                  self._stepsize,
                                                                  apogee_factor= self._apogee_selection_parameter
                                                                      )
        fine_step_size_interval_initial = self.adapt_step_size(coarse_level_interval_current_state)
        proposal_theta, proposal_rho = fine_step_size_interval_initial.proposal_phase_space()
        coarse_level_interval_proposal = CoarseLevelIntervalAAPS(self,
                                                             proposal_theta,
                                                             proposal_rho,
                                                             self._stepsize,
                                                             number_apogees = coarse_level_interval_current_state.number_apogees(),

                                                             )
        fine_step_size_interval_proposal = self.adapt_step_size(coarse_level_interval_proposal)

        accept_step_size = fine_step_size_interval_initial.step_size_acceptance(fine_step_size_interval_proposal)
        self._step_size_non_overlap_rejects += (1 - accept_step_size)
        ratio_norm_fact = (fine_step_size_interval_initial.normalization_factor()
                           / fine_step_size_interval_proposal.normalization_factor())
        accept_index_selection = fine_step_size_interval_proposal.accept_index_selection(fine_step_size_interval_initial)

        self._no_return_rejects += (1 - accept_index_selection)
        acceptance_probability = accept_step_size * ratio_norm_fact * accept_index_selection
        if self._rng.uniform() < acceptance_probability:
            theta = proposal_theta
            rho = proposal_rho
        else:
            self._rejects += 1
        self._theta = theta
        return self._theta, rho

        #acceptance_probability = self.compute_acceptance_probability(fine_step_size_interval, fine_step_size_interval_proposal)

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
    def adapt_step_size(self, coarse_level_interval, randomize = True):
        #::CoarseLevelInterval -> (int, FineLevelInterval)
        last_interval = coarse_level_interval
        current_fine_level_interval = FineLevelIntervalAAPS(self,
                                                        self._rng,
                                                        self._theta,
                                                        self._rho,
                                                        coarse_level_interval,
                                                        1)
        energy_range = current_fine_level_interval.energy_range()
        while energy_range > -self._log_min_accept_prob and depth < self._max_step_size_search_depth:
            last_interval = current_fine_level_interval
            current_fine_level_interval = FineLevelIntervalAAPS(self,
                                                            self._rng,
                                                            self._theta,
                                                            self._rho,
                                                            coarse_level_interval,
                                                            depth+1)
            energy_range = current_fine_level_interval.energy_range()

        if randomize:
            return self.interval_selection_after_random_stepsize(depth, last_interval, current_fine_level_interval)
        return  current_fine_level_interval

    def compute_stepsize_accept_ratio(self, fine_grid_initial,depth_proposal):
        # ::float -> float -> float
        return abs(np.log(step_size/step_size_from_reverse)) <= 1
    def log_density_gradient_at_theta(self, theta):
        return self._model.log_density_gradient(theta)
    def interval_selection_after_random_stepsize(self, depth, coarsegrid, last_interval, current_interval):
        #::int -> CoarseLevelInterval -> CoarseLevelInterval -> CoarseLevelInterval
        log_random_step_size_multiplier = self._rng.integers(-1, 2)
        if log_random_step_size_multiplier == -1:
            return last_interval
        if log_random_step_size_multiplier == 0:
            return current_interval
        if log_random_step_size_multiplier == 1:
            return FineLevelIntervalAAPS(self, self._rng, self._theta, self._rho, coarsegrid, depth+1)


class CoarseLevelIntervalAAPS:
    def __init__(self, sampler, rng, theta, rho, stepsize, apogee_factor = None, num_apogees = None, shift = None):
        self._sampler = sampler
        self._rng = rng
        self._theta = theta
        self._rho = rho
        self._energy_max = -self._sampler.log_joint(self._theta, self._rho)
        self._energy_min = self._energy_max

        self._number_apogees = self._rng.geometric(apogee_factor)
        self._depth = 0
        self._shift = self._rng.integers(0, self._number_apogees)
        self._stepsize = stepsize
        self._total_weight = np.exp(-self._sampler.log_joint(self._theta, self._rho))
        self._iterate_num_segments_left = []
        self._iterate_num_segments_right = []
        self._number_left = 0
        self._number_right = 0
        self._sample_theta = theta
        self._sample_rho = rho
        self._sample_index = 0
        self._sample_coarse_grid_segment_index = 0

        if num_apogees is not None:
            self._number_apogees = num_apogees
        else:
            self._number_apogees = self._rng.geometric(apogee_factor)

        if shift is not None:
            self._shift = shift
        else:
            self._shift = self._rng.integers(0, self._number_apogees)

        self._sampler.set_stepsize(self._stepsize)
        self.iterate_over_both_sides_coarse()
        self._sampler.stepsize_reset_original()
    def iterate_over_both_sides_coarse(self):
        # ::None -> None
        # This function implements finding the endpoint in the Apogee to
        # Apogee interval. In principle this should not differ too much for other
        # similar conditions
        self._iterate_num_segments_right = self.iterate_over_single_side_coarse(1, self._number_apogees - self._shift)
        self._iterate_num_segments_left = self.iterate_over_single_side_coarse(-1, self._shift + 1)
        self._number_left = self._iterate_num_segments_left[-1]
        self._number_right = self._iterate_num_segments_right[-1]
    def iterate_over_single_side_coarse(self, direction, number_apogees):
        iterate_numbers_marking_segments = []
        current_theta = self._theta
        current_rho = self._rho
        current_iterate_index = 0
        number_changes_seen = 0
        _, current_grad = self._sampler.log_density_gradient_at_theta(current_theta)
        current_sign = np.dot(current_rho, current_grad)
        while number_changes_seen < number_apogees:
            next_theta, next_rho = self._sampler.leapfrog_step(current_theta, direction*current_rho)
            next_rho = direction*next_rho
            #If the direction is -1, then we are going backwards
            _, next_grad = self._sampler.log_density_gradient_at_theta(next_theta)
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
                self._sample_coarse_grid_segment_index = direction * number_changes_seen
                self._sample_index = current_iterate_index

            self._total_weight += current_weight

        return iterate_numbers_marking_segments
    def interval_markers_left(self):
        return self._iterate_num_segments_left
    def interval_markers_right(self):
        return self._iterate_num_segments_right
    def stepsize(self):
        return self._stepsize
    def energy_range(self):
        return self._energy_max, self._energy_min
    def proposal_phase_space(self):
        return self._sample_theta, self._sample_rho
    def number_apogees(self):
        return self._number_apogees
    def proposal_shift(self):
        return self._shift + self._sample_coarse_grid_segment_index
    def step_size_acceptance(self, intial_fine_grid):
        return int(abs(self._depth - intial_fine_grid._depth) <= 1)
    def normalization_factor(self):
        return self._total_weight

    def accept_index_selection(self, initial_fine_grid):
        proposal_index = -initial_fine_grid._sample_index
        number_left = -self._iterate_num_segments_left[-1]
        number_right = self._iterate_num_segments_right[-1]
        return int((proposal_index <= number_right) and (proposal_index >= number_left))

class FineLevelIntervalAAPS:
    def __init__(self, sampler, rng, theta, rho, coarse_grid, depth):
        self._sampler = sampler
        self._rng = rng
        self._theta = theta
        self._rho = rho
        self._coarse_grid = coarse_grid
        self._depth = depth
        self._stepsize = self._coarse_grid.stepsize()/(2**self._depth)
        self._number_fine_grid_leapfrog_steps = 2**self._depth
        self._max_energy = -self._sampler.log_joint(self._theta, self._rho)
        self._min_energy = self._max_energy
        self._total_weight = np.exp(-self._sampler.log_joint(self._theta, self._rho))
        self._sample_theta = theta
        self._sample_rho = rho
        self._sample_coarse_grid_segment_index = 0
        self._sample_index = 0
        self._iterate_num_segments_left = self._coarse_grid.interval_markers_left()
        self._iterate_num_segments_right = self._coarse_grid.interval_markers_right()

        self._sampler.set_stepsize(self._stepsize)
        self.iterate_over_both_sides_fine()
        self._sampler.stepsize_reset_original()
    def iterate_over_both_sides_fine(self):
        self.iterate_over_single_side_fines(1, self._iterate_num_segments_right)
        self.iterate_over_single_side_fine(-1, self._iterate_num_segments_left)
    def iterate_over_single_side_fine(self, direction, iterate_num_marking_segments):
        current_theta = self._theta
        current_rho = self._rho
        current_iterate_index = 0
        current_energy = -self._sampler.log_joint(current_theta, current_rho)
        current_weight = np.exp(-current_energy)
        self._max_energy = max(self._max_energy, current_energy)
        self._min_energy = min(self._min_energy, current_energy)

        for iterate_index, segment_marker in enumerate(iterate_num_marking_segments):
            for index in range(current_iterate_index, segment_marker):
                current_theta, current_rho = self.iterated_leapfrog_with_energy_max_min(current_theta,current_rho)
                current_energy = -self._sampler.log_joint(current_theta, current_rho)
                current_weight = np.exp(-current_energy)
                if current_weight/self._total_weight < self._rng.uniform():
                    self._sample_theta = current_theta
                    self._sample_rho = current_rho
                    self._sample_coarse_grid_segment_index = direction*iterate_index
                    self._sample_index = index + 1

                self._total_weight += current_weight
                self._max_energy = max(self._max_energy, current_energy)
                self._min_energy = min(self._min_energy, current_energy)
        return None
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
    def energy_range(self):
        return self._max_energy, self._min_energy
    def proposal_phase_space(self):
        return self._sample_theta, self._sample_rho
    def proposal_shift(self):
        return self._coarse_grid._shift + self._sample_coarse_grid_segment_index
            #Accessing the private variable isn't ideal, but it really isn't that big of a deal
    def step_size_acceptance(self, intial_fine_grid):
        return int(abs(self._depth - intial_fine_grid._depth) <= 1)
    def normalization_factor(self):
        return self._total_weight
    def accept_index_selection(self, initial_fine_grid):
        proposal_index = -initial_fine_grid._sample_index
        number_left = -self._iterate_num_segments_left[-1]
        number_right = self._iterate_num_segments_right[-1]
        return int((proposal_index <= number_right) and (proposal_index >= number_left))