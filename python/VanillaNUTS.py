import hmc
import numpy as np


class VanillaNUTS(hmc.HmcSamplerBase):
    def __init__(self,
                 model,
                 rng,
                 theta,
                 rho,
                 step_size,
                 max_nuts_depth):
        super().__init__(model, step_size, rng)
        self._theta = theta
        self._rho = rho
        self._max_nuts_search_depth = max_nuts_depth

    def draw(self):
        self._rho = self._rng.normal(size=self._model.param_unc_num())
        theta, rho = self._theta, self._rho
        theta_prime, rho_prime, energy_max, energy_min = self.NUTS(theta, rho, self._max_nuts_search_depth)
        self._theta = theta_prime
        self._rho = rho_prime
        return theta_prime, rho_prime

    def NUTS(self, theta, rho, max_height):
        bernoulli_sequence = self._rng.binomial(1, 0.5, max_height)
        sample_orbit = NUTS_Orbit(self,
                                  self._rng,
                                  theta,
                                  rho,
                                  self._stepsize,
                                  1,
                                  max_height,
                                  bernoulli_sequence)
        return sample_orbit._sample_theta, sample_orbit._sample_rho, sample_orbit._energy_max, sample_orbit._energy_min


class NUTSOrbit:
    def __init__(self, sampler, rng, theta, rho, step_size, number_fine_grid_leapfrog_steps, max_height, bernoulli_sequence):
        self._sampler = sampler
        self._rng = rng
        self._theta = theta
        self._rho = rho
        self._step_size = step_size
        self._number_fine_grid_leapfrog_steps = number_fine_grid_leapfrog_steps
        self._max_height = max_height
        self._bernoulli_sequence = bernoulli_sequence

        self._sampler._step_size = step_size
        sample_theta, sample_rho, energy_max, energy_min = self.Coarse_Fine_NUTS()
        self._sample_theta = sample_theta
        self._sample_rho = sample_rho
        self._energy_max = energy_max
        self._energy_min = energy_min

    def Coarse_Fine_NUTS(self):
        theta, rho = self._theta, self._rho
        left_theta = theta
        left_rho = rho
        right_theta = theta
        right_rho = rho
        height = 0
        sample_theta = theta
        sample_rho = rho
        log_weight_current = self._sampler.log_joint(theta, rho)
        log_weight_new = 0
        energy_max = energy_min = -self._sampler.log_joint(theta, rho)

        for forward in self._bernoulli_sequence:
            if forward:
                (extension_theta,
                 extension_rho,
                 energy_max_interior_fine_grid,
                 energy_min_interior_fine_grid) = self.iterated_leapfrog_with_energy_max_min(right_theta,
                                                                                             right_rho)

                (right_theta,
                 right_rho,
                 extension_sample_theta,
                 extension_sample_rho,
                 log_weight_new,
                 sub_u_turn,
                 extension_energy_max,
                 extension_energy_min) = self.evaluate_proposed_subtree(extension_theta,
                                                                        extension_rho,
                                                                        height)
            else:
                (extension_theta,
                 extension_rho,
                 energy_max_interior_fine_grid,
                 energy_min_interior_fine_grid) = self.iterated_leapfrog_with_energy_max_min(left_theta,
                                                                                             -left_rho)
                (left_theta,
                 left_rho,
                 extension_sample_theta,
                 extension_sample_rho,
                 log_weight_new,
                 sub_u_turn,
                 extension_energy_max,
                 extension_energy_min) = self.evaluate_proposed_subtree(extension_theta,
                                                                        extension_rho,
                                                                        height)

                left_rho = -left_rho

            if sub_u_turn:
                return sample_theta, sample_rho, energy_max, energy_min

            sample_theta, sample_rho = self.resample_top(sample_theta,
                                                         sample_rho,
                                                         extension_sample_theta,
                                                         extension_sample_rho,
                                                         log_weight_current,
                                                         log_weight_new)
            height += 1
            log_weight_current =  np.logaddexp(log_weight_current, log_weight_new)
            energy_max = max(energy_max, energy_max_interior_fine_grid, extension_energy_max)
            energy_min = min(energy_min, energy_min_interior_fine_grid, extension_energy_min)

            if self.nuts_style_u_turn(left_theta, left_rho, right_theta, right_rho):
                return sample_theta, sample_rho, energy_max, energy_min

        return sample_theta, sample_rho, energy_max, energy_min

    def evaluate_proposed_subtree(self, theta, rho, height):
        if height == 0:
            return (theta,
                    rho,
                    theta,
                    rho,
                    self._sampler.log_joint(theta, rho),
                    False,
                    -self._sampler.log_joint(theta, rho),
                    -self._sampler.log_joint(theta, rho))

        (theta_right_subtree_left,
         rho_right_subtree_left,
         sample_theta_subtree_left,
         sample_rho_subtree_left,
         log_weight_subtree_left,
         sub_u_turn_subtree_left,
         energy_max_fine_grid_left,
         energy_min_fine_grid_left) = self.evaluate_proposed_subtree(theta, rho, height - 1)

        if sub_u_turn_subtree_left:
            return (theta, rho, theta, rho, 0, True, 0, 0)

        (theta_left_subtree_right,
         rho_left_subtree_right,
         energy_max_interior_fine_grid,
         energy_min_interior_fine_grid) = self.iterated_leapfrog_with_energy_max_min(theta_right_subtree_left,
                                                                                     rho_right_subtree_left)

        (theta_right_subtree_right,
         rho_right_subtree_right,
         sample_theta_subtree_right,
         sample_rho_subtree_right,
         log_weight_subtree_right,
         sub_u_turn_subtree_right,
         energy_max_fine_grid_right,
         energy_min_fine_grid_right) = self.evaluate_proposed_subtree(theta_left_subtree_right,
                                                                      rho_left_subtree_right,
                                                                      height - 1)

        sub_u_turn = (sub_u_turn_subtree_left) or (sub_u_turn_subtree_right) or self.nuts_style_u_turn(theta,
                                                                                                       rho,
                                                                                                       theta_right_subtree_right,
                                                                                                       rho_right_subtree_right)

        if sub_u_turn:
            return (theta, rho, theta, rho, 0, True, 0, 0)

        sample_theta, sample_rho = self.resample_sub_tree(sample_theta_subtree_left,
                                                          sample_rho_subtree_left,
                                                          sample_theta_subtree_right,
                                                          sample_rho_subtree_right,
                                                          log_weight_subtree_left,
                                                          log_weight_subtree_right)

        log_weight_subtree = np.logaddexp(log_weight_subtree_left,  log_weight_subtree_right)
        energy_max = max(energy_max_fine_grid_left, energy_max_interior_fine_grid, energy_max_fine_grid_right)
        energy_min = min(energy_min_fine_grid_left, energy_min_interior_fine_grid, energy_min_fine_grid_right)

        return (theta_right_subtree_right,
                rho_right_subtree_right,
                sample_theta,
                sample_rho,
                log_weight_subtree,
                sub_u_turn,
                energy_max,
                energy_min)

    def resample_top(self, sample_theta, sample_rho, new_sample_theta, new_sample_rho, log_weight_current, log_weight_new):
        if np.log(self._rng.uniform(0, 1)) < log_weight_new - np.logaddexp(log_weight_current, log_weight_new):
            return new_sample_theta, new_sample_rho
        else:
            return sample_theta, sample_rho

    def resample_sub_tree(self, sample_theta, sample_rho, new_sample_theta, new_sample_rho, log_weight_current, log_weight_new):
        if np.log(self._rng.uniform(0, 1)) < log_weight_new - np.logaddexp(log_weight_current,  log_weight_new):
            return new_sample_theta, new_sample_rho
        else:
            return sample_theta, sample_rho

    def nuts_style_u_turn(self, left_theta, left_rho, right_theta, right_rho):
        delta_theta = right_theta - left_theta
        return (np.dot(delta_theta, left_rho) < 0) and (np.dot(delta_theta, right_rho) < 0)

    def iterated_leapfrog_with_energy_max_min(self, theta, rho):
        max_energy = min_energy = -self._sampler.log_joint(theta, rho)
        theta_current = theta
        rho_current = rho

        for i in range(self._number_fine_grid_leapfrog_steps):
            theta_current, rho_current = self._sampler.leapfrog_step(theta_current, rho_current)
            current_energy = -self._sampler.log_joint(theta_current, rho_current)
            max_energy = max(max_energy, current_energy)
            min_energy = min(min_energy, current_energy)

        return theta_current, rho_current, max_energy, min_energy

class NutsTreeNode:

class NutsSample:
    def __init__(self, theta, rho, l):
