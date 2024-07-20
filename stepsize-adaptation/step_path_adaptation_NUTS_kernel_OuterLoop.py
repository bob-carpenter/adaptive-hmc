import numpy as np

import hmc

class StepAdapt_NUTS_Sampler(hmc.HmcSamplerBase):
    def __init__(self,
                 model,
                 rng,
                 theta,
                 rho,
                 min_accept_prob,
                 max_stepsize,
                 max_step_size_search_depth,
                 max_nuts_depth):

        super().__init__(model, 0.0, rng)
        self._theta = theta
        self._rho = rho
        self._log_min_accept_prob = np.log(min_accept_prob)
        self._max_stepsize = max_stepsize
        self._max_step_size_search_depth = max_step_size_search_depth
        self._max_nuts_search_depth = max_nuts_depth

    def draw(self):
        self._stepsize = self._max_stepsize
        self._rho = self._rng.normal(size=self._model.param_unc_num())
        theta, rho = self._theta, self._rho
        for i in range(self._max_step_size_search_depth):
            theta_prime, rho_prime, energy_max, energy_min = self.NUTS(theta, rho, self._max_nuts_search_depth)
            if -(energy_max - energy_min) > self._log_min_accept_prob:
                self._theta = theta_prime
                return theta_prime, rho_prime
            self._stepsize = self._stepsize / 2
        self._theta = theta_prime
        return self._theta, self._rho

    def NUTS(self, theta, rho, max_height):
        lower_theta = theta
        lower_rho = rho
        upper_theta = theta
        upper_rho = rho
        height = 0
        sample_theta = theta
        sample_rho = rho
        weight_current = np.exp(self.log_joint(theta, rho))
        energy_max = energy_min = -self.log_joint(theta, rho)

        weight_new = 0

        for i in range(max_height):
            forward_or_backward_choice = self._rng.integers(0, 2)
            if forward_or_backward_choice == 0:
                extension_theta, extension_rho = self.leapfrog_step(lower_theta, -lower_rho)
                # This line handles considering the new extension

                (lower_theta,
                 lower_rho,
                 new_sample_theta,
                 new_sample_rho,
                 weight_new,
                 sub_u_turn,
                 new_energy_max,
                 new_energy_min) = self.evaluate_proposed_subtree(extension_theta,
                                                                  extension_rho,
                                                                  height)
                lower_rho = -lower_rho

            else:
                extension_theta, extension_rho = self.leapfrog_step(upper_theta, upper_rho)
                (upper_theta,
                 upper_rho,
                 new_sample_theta,
                 new_sample_rho,
                 weight_new,
                 sub_u_turn,
                 new_energy_max,
                 new_energy_min) = self.evaluate_proposed_subtree(extension_theta,
                                                                  extension_rho,
                                                                  height)
            if sub_u_turn:
                return sample_theta, sample_rho, energy_max, energy_min

            sample_theta, sample_rho = self.resample_top(sample_theta,
                                                         sample_rho,
                                                         new_sample_theta,
                                                         new_sample_rho,
                                                         weight_current,
                                                         weight_new)
            height += 1
            weight_current += weight_new
            energy_max = max(energy_max, new_energy_max)
            energy_min = min(energy_min, new_energy_min)

            if self.nuts_style_u_turn(lower_theta, lower_rho, upper_theta, upper_rho):
                return sample_theta, sample_rho, energy_max, energy_min

        return sample_theta, sample_rho, energy_max, energy_min

    def evaluate_proposed_subtree(self, theta, rho, height):
        if height == 0:
            return theta, rho, theta, rho, np.exp(self.log_joint(theta, rho)), False, -self.log_joint(theta,
                                                                                                      rho), -self.log_joint(
                theta, rho)

        (theta_subtree_left,
         rho_subtree_left,
         sample_theta_subtree_left,
         sample_rho_subtree_left,
         weight_subtree_left,
         sub_u_turn_subtree_left,
         new_energy_max_subtree_left,
         new_energy_min_subtree_left) = self.evaluate_proposed_subtree(theta, rho, height - 1)

        sub_u_turn = sub_u_turn_subtree_left

        if sub_u_turn:
            return (theta, rho, theta, rho, 0, True, 0, 0)
            # Immediately return if the subtree has a u-turn

        left_theta_subtree_right, left_rho_subtree_right = self.leapfrog_step(theta_subtree_left, rho_subtree_left)
        (theta_subtree_right,
         rho_subtree_right,
         sample_theta_subtree_right,
         sample_rho_subtree_right,
         weight_subtree_right,
         sub_u_turn_subtree_right,
         new_energy_max_subtree_right,
         new_energy_min_subtree_right) = self.evaluate_proposed_subtree(left_theta_subtree_right,
                                                                        left_rho_subtree_right, height - 1)

        sub_u_turn = (sub_u_turn) or (sub_u_turn_subtree_right) or self.nuts_style_u_turn(theta,
                                                                                          rho,
                                                                                          theta_subtree_right,
                                                                                          rho_subtree_right)

        if sub_u_turn:
            # Immediately return if the subtree has a u-turn

            return (theta, rho, theta, rho, 0, True, 0, 0)

        sample_theta, sample_rho = self.resample_sub_tree(sample_theta_subtree_left,
                                                          sample_rho_subtree_left,
                                                          sample_theta_subtree_right,
                                                          sample_rho_subtree_right,
                                                          weight_subtree_left,
                                                          weight_subtree_right)

        weight_subtree = weight_subtree_left + weight_subtree_right
        new_energy_max = max(new_energy_max_subtree_left, new_energy_max_subtree_right)
        new_energy_min = min(new_energy_min_subtree_left, new_energy_min_subtree_right)
        return (theta_subtree_right,
                rho_subtree_right,
                sample_theta,
                sample_rho,
                weight_subtree,
                sub_u_turn,
                new_energy_max,
                new_energy_min)

    def resample_top(self, sample_theta, sample_rho, new_sample_theta, new_sample_rho, weight_current, weight_new):
        if self._rng.uniform(0, 1) < weight_new / (weight_current + weight_new):
            return new_sample_theta, new_sample_rho
        else:
            return sample_theta, sample_rho

    def resample_sub_tree(self, sample_theta, sample_rho, new_sample_theta, new_sample_rho, weight_current, weight_new):
        if self._rng.uniform(0, 1) < weight_new / (weight_current + weight_new):
            return new_sample_theta, new_sample_rho
        else:
            return sample_theta, sample_rho

    def nuts_style_u_turn(self, lower_theta, lower_rho, upper_theta, upper_rho):
        delta_theta = upper_theta - lower_theta
        return (np.dot(delta_theta, lower_rho) < 0) or (np.dot(delta_theta, upper_rho) < 0)
