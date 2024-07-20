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

        coarse_interval_current = CoarseLevelIntervalAAPS(self,
                                                          self._rng,
                                                          self._theta,
                                                          self._rho,
                                                          self._max_step_size,
                                                          )
        coarse_interval_current.generate_num_apogees_and_shift(self._apogee_selection_parameter)
        # print(f"Initial shift: {coarse_interval_current.get_shift()}")
        coarse_interval_current.populate_interval()
        # This is the logic for generating the coarse level interval

        fine_interval_current = self.adapt_step_size(coarse_interval_current)
        proposal_theta, proposal_rho = fine_interval_current.proposal_phase_space()
        # This is the logic for adapting the step size and generating a sample

        coarse_interval_proposal = CoarseLevelIntervalAAPS(self,
                                                           self._rng,
                                                           proposal_theta,
                                                           proposal_rho,
                                                           self._max_step_size,
                                                           )
        coarse_interval_proposal.set_num_apogees_and_shift(coarse_interval_current.number_apogees(),
                                                           fine_interval_current.proposal_shift())

        coarse_interval_proposal.populate_interval()
        fine_interval_proposal = self.adapt_step_size(coarse_interval_proposal, randomize=False)
        # This the logic for simulating from the proposal. We do not randomize the step size
        # because we want to compare the acceptance probability with the current step size
        accept_step_size = fine_interval_current.step_size_acceptance(fine_interval_proposal)
        self._step_size_non_overlap_rejects += (1 - accept_step_size)
        ratio_norm_fact = (fine_interval_current.normalization_factor()
                           / fine_interval_proposal.normalization_factor())
        boltzmann_ratio = np.exp(-fine_interval_proposal.initial_energy() + fine_interval_current.initial_energy())
        accept_index_selection = fine_interval_proposal.accept_index_selection(fine_interval_current)
        # print(f"Current normalization factor: {fine_interval_current.normalization_factor()}")
        # print(f"Proposal normalization factor: {fine_interval_proposal.normalization_factor()}")
        # print(f"Ratio of normalization factors: {ratio_norm_fact}")
        # print(f"Boltzmann ratio: {boltzmann_ratio}")
        # print(f"Acceptance index selection: {accept_index_selection}")
        # print(f"Accept step size: {accept_step_size}")
        # print(f"Acceptance probability: {accept_step_size * ratio_norm_fact * accept_index_selection * boltzmann_ratio}")
        # We should have the normalization factor for the current interval on top
        # since it is in the basement of the Boltzmann probability.

        self._no_return_rejects += (1 - accept_index_selection)
        acceptance_probability = accept_step_size * ratio_norm_fact * accept_index_selection * boltzmann_ratio
        # print(f"Number of apogees proposal: {coarse_interval_proposal.get_number_apogees()}")
        # print(f"Shift proposal: {coarse_interval_proposal.get_shift()}")
        if self._rng.uniform() < acceptance_probability:
            theta = proposal_theta
            rho = proposal_rho
        else:
            self._rejects += 1
        self._theta = theta
        return self._theta, rho

    def adapt_step_size(self, coarse_level_interval, randomize=True):
        #::CoarseLevelInterval -> (int, FineLevelInterval)
        last_interval = coarse_level_interval
        current_fine_level_interval = FineLevelIntervalAAPS(self,
                                                            self._rng,
                                                            self._theta,
                                                            self._rho,
                                                            coarse_level_interval,
                                                            1)
        energy_range = current_fine_level_interval.energy_range()
        depth = 1
        while energy_range > -self._log_min_accept_prob and depth < self._max_step_size_search_depth:
            last_interval = current_fine_level_interval
            current_fine_level_interval = FineLevelIntervalAAPS(self,
                                                                self._rng,
                                                                self._theta,
                                                                self._rho,
                                                                coarse_level_interval,
                                                                depth + 1)
            depth += 1
            energy_range = current_fine_level_interval.energy_range()

        if randomize:
            return self.interval_selection_after_random_stepsize(depth, coarse_level_interval, last_interval,
                                                                 current_fine_level_interval)
        return current_fine_level_interval

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
            return FineLevelIntervalAAPS(self, self._rng, self._theta, self._rho, coarsegrid, depth + 1)


class CoarseLevelIntervalAAPS:
    def __init__(self, sampler, rng, theta, rho, stepsize):
        self._sampler = sampler
        self._rng = rng
        self._theta = theta
        self._rho = rho
        self._energy_max = -self._sampler.log_joint(self._theta, self._rho)
        self._energy_min = self._energy_max
        self._initial_energy = self._energy_max
        self._total_weight = 1

        self._number_apogees = None
        self._shift = None
        # print("New Coarse Level Interval")
        # print(f"Number of apogees: {self._number_apogees}")
        self._depth = 0
        self._stepsize = stepsize

        self._iterate_num_segments_left = []
        self._iterate_num_segments_right = []
        self._number_left = 0
        self._number_right = 0
        self._sample_theta = theta
        self._sample_rho = rho
        self._sample_index = 0
        self._sample_coarse_grid_segment_index = 0

    def generate_num_apogees_and_shift(self, apogee_factor):
        self._number_apogees = 2 + self._rng.geometric(apogee_factor)
        # We want to make sure that the number of apogees is at least 2
        print(f"Number of apogees: {self._number_apogees}")
        self._shift = self._rng.integers(0, self._number_apogees)

    def set_num_apogees_and_shift(self, number_apogees, shift):
        self._number_apogees = number_apogees
        self._shift = shift

    def populate_interval(self):
        self._sampler.set_stepsize(self._stepsize)
        self.iterate_over_both_sides_coarse()
        self._sampler.stepsize_reset_original()

    def get_number_apogees(self):
        return self._number_apogees

    def get_shift(self):
        return self._shift

    def iterate_over_both_sides_coarse(self):
        # ::None -> None
        # This function implements finding the endpoint in the Apogee to
        # Apogee interval. In principle this should not differ too much for other
        # similar conditions
        if self._number_apogees is None or self._shift is None:
            raise ValueError("The number of apogees or the shift has not been set at the coarse level")
        self._iterate_num_segments_right = self.iterate_over_single_side_coarse(1, self._number_apogees - self._shift)
        # print("\n \n \n Done with right side \n \n \n")
        self._iterate_num_segments_left = self.iterate_over_single_side_coarse(-1, self._shift + 1)
        self._number_left = self._iterate_num_segments_left[-1]
        self._number_right = self._iterate_num_segments_right[-1]

    def iterate_over_single_side_coarse(self, direction, num_segments):
        iterate_numbers_marking_segments = []
        current_theta = self._theta
        current_rho = self._rho
        current_iterate_index = 0
        number_changes_seen = 0
        _, current_grad = self._sampler.log_density_gradient_at_theta(current_theta)
        current_sign = np.dot(current_rho, current_grad)
        while number_changes_seen < num_segments:
            next_theta, next_rho = self._sampler.leapfrog_step(current_theta, direction * current_rho)
            next_rho = direction * next_rho
            # If the direction is -1, then we are going backwards
            _, next_grad = self._sampler.log_density_gradient_at_theta(next_theta)
            # We're calcluating the gradient twice. In the next
            # version lets either cache the gradient or compute this during
            # the leapfrog step

            next_sign = np.dot(next_rho, next_grad)
            if current_sign * next_sign < 0:
                # print("Hit an apogee")
                number_changes_seen += 1
                iterate_numbers_marking_segments.append(current_iterate_index)
            # else:
            # print("Still looking for a sign change")
            # print("Current sign: ", current_sign)
            # print("Current position: ", current_theta)
            # print("Stepsize: ", self._stepsize)
            if number_changes_seen == num_segments:
                # print("Hit the endpoint")
                break
                # In this case the iterate is the endpoint and should not be
                # considered to be resampled
            current_theta = next_theta
            current_rho = next_rho
            current_sign = next_sign
            current_iterate_index += 1
            current_energy = -self._sampler.log_joint(current_theta, current_rho)
            current_weight = np.exp(-(current_energy - self._initial_energy))
            self._energy_max = max(self._energy_max, current_energy)
            self._energy_min = min(self._energy_min, current_energy)

            if self._rng.uniform() < current_weight / self._total_weight:
                self._sample_theta = current_theta
                self._sample_rho = current_rho
                self._sample_coarse_grid_segment_index = direction * number_changes_seen
                self._sample_index = direction * current_iterate_index

            self._total_weight += current_weight

        return iterate_numbers_marking_segments

    def interval_markers_left(self):
        return self._iterate_num_segments_left

    def interval_markers_right(self):
        return self._iterate_num_segments_right

    def stepsize(self):
        return self._stepsize

    def energy_range(self):
        return self._energy_max - self._energy_min

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

    def sample_index(self):
        return self._sample_index

    def accept_index_selection(self, initial_fine_grid):
        proposal_index = -initial_fine_grid._sample_index
        number_left = -self._iterate_num_segments_left[-1]
        number_right = self._iterate_num_segments_right[-1]
        return int((proposal_index <= number_right) and (proposal_index >= number_left))

    def initial_energy(self):
        return self._initial_energy


class FineLevelIntervalAAPS:
    def __init__(self, sampler, rng, theta, rho, coarse_grid, depth):
        self._sampler = sampler
        self._rng = rng
        self._theta = theta
        self._rho = rho
        self._coarse_grid = coarse_grid
        self._depth = depth
        self._stepsize = self._coarse_grid.stepsize() / (2 ** self._depth)
        self._number_fine_grid_leapfrog_steps = 2 ** self._depth
        self._max_energy = -self._sampler.log_joint(self._theta, self._rho)
        self._min_energy = self._max_energy
        self._initial_energy = self._max_energy
        self._total_weight = 1

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
        self.iterate_over_single_side_fine(1, self._iterate_num_segments_right)
        # print("\n \n \n Done with right side \n \n \n")
        self.iterate_over_single_side_fine(-1, self._iterate_num_segments_left)

    def iterate_over_single_side_fine(self, direction, iterate_num_marking_segments):
        current_theta = self._theta
        current_rho = self._rho
        current_iterate_index = 0
        current_energy = -self._sampler.log_joint(current_theta, current_rho)
        current_weight = np.exp(-(current_energy - self._initial_energy))
        # This is some wasted computation, but I'm keeping it for
        # conceptual clarity

        self._max_energy = max(self._max_energy, current_energy)
        self._min_energy = min(self._min_energy, current_energy)
        # print(f"Direction: {direction}")

        for segment_index, segment_marker in enumerate(iterate_num_marking_segments):
            for index in range(current_iterate_index, segment_marker):
                current_theta, current_rho = self.iterated_leapfrog_with_energy_max_min(current_theta,
                                                                                        direction * current_rho)
                current_rho = direction * current_rho
                current_energy = -self._sampler.log_joint(current_theta, current_rho)
                current_weight = np.exp(-(current_energy - self._initial_energy))
                current_iterate_index += 1
                # print(f"Current weight: {current_weight}")
                # print(f"Total weight: {self._total_weight}")
                # print(f"Current index: {index}")
                # print(f"Current segment index: {segment_index}")
                if self._rng.uniform() < current_weight / self._total_weight:
                    self._sample_theta = current_theta
                    self._sample_rho = current_rho
                    self._sample_coarse_grid_segment_index = direction * segment_index
                    # print(f"Current proposal index {self._sample_coarse_grid_segment_index}")
                    self._sample_index = direction * (index + 1)

                # print(f"Current weight: {current_weight}")

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
        return self._max_energy - self._min_energy

    def proposal_phase_space(self):
        return self._sample_theta, self._sample_rho

    def proposal_shift(self):
        return self._coarse_grid._shift + self._sample_coarse_grid_segment_index
        # Accessing the private variable isn't ideal, but it really isn't that big of a deal

    def step_size_acceptance(self, intial_fine_grid):
        return int(abs(self._depth - intial_fine_grid._depth) <= 1)

    def normalization_factor(self):
        return self._total_weight

    def sample_index(self):
        return self._sample_index

    def accept_index_selection(self, initial_fine_grid):
        proposal_index = -initial_fine_grid.sample_index()
        number_left = -self._iterate_num_segments_left[-1]
        number_right = self._iterate_num_segments_right[-1]
        # print(f"Proposal index: {proposal_index}")
        # print(f"Left side inital fine grid: {initial_fine_grid._iterate_num_segments_left}")
        # print(f"Right side initial fine grid: {initial_fine_grid._iterate_num_segments_right}")
        # print(f"Proposal segment index: {initial_fine_grid._sample_coarse_grid_segment_index}")
        # print(f"Number left: {number_left}")
        # print(f"Number right: {number_right}")

        # We're testing whether or not the original point is reachable from the proposal
        return int((proposal_index <= number_right) and (proposal_index >= number_left))

    def initial_energy(self):
        return self._initial_energy
