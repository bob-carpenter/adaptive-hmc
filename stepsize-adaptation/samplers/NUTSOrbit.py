from dataclasses import dataclass
import numpy as np

@dataclass
class NUTSSample:
    _theta: float
    _rho: float
    _bernoulli_sequence: tuple

    def reverse_time(self):
        self._bernoulli_sequence = tuple(1 - x for x in self._bernoulli_sequence)
        self._rho = -self._rho


@dataclass
class NUTSTreeNode:
    _left_theta: float
    _left_rho: float
    _right_theta: float
    _right_rho: float
    _sub_u_turn: bool
    _u_turn: bool
    _height: int
    _log_weight: float
    _energy_max: float
    _energy_min: float
    _sample: NUTSSample

    @classmethod
    def initialize_leaf(cls,
                        parent_orbit,
                        theta,
                        rho,
                        energy_max,
                        energy_min):
        return cls(theta,
                   rho,
                   theta,
                   rho,
                   False,
                   False,
                   0,
                   parent_orbit._sampler.log_joint(theta, rho),
                   energy_max,
                   energy_min,
                   NUTSSample(theta, rho, ())
                   )

    def time_reverse(self):
        self._left_theta, self._right_theta = self._right_theta, self._left_theta
        self._left_rho = -self._right_rho
        self._right_rho = -self._left_rho
        self._sample.reverse_time()
        return self

    @classmethod
    def nuts_style_u_turn(cls, left_theta, left_rho, right_theta, right_rho):
        delta_theta = right_theta - left_theta
        return (np.dot(delta_theta, left_rho) < 0) and (np.dot(delta_theta, right_rho) < 0)

    @classmethod
    def combine_lower_level_trees(cls, left, right, sample):
        new_node = cls.combine_top_level_trees(left, right, sample)
        new_node._sub_u_turn = new_node._sub_u_turn or new_node._u_turn
        return new_node

    @classmethod
    def combine_top_level_trees(cls, left, right, sample):
        sub_u_turn_lower = left._sub_u_turn or right._sub_u_turn
        u_turn = cls.nuts_style_u_turn(left._left_theta, left._left_rho, right._right_theta, right._right_rho)
        log_weight = np.logaddexp(left._log_weight, right._log_weight)
        energy_max = max(left._energy_max, right._energy_max)
        energy_min = min(left._energy_min, right._energy_min)
        return cls(left._left_theta,
                   left._left_rho,
                   right._right_theta,
                   right._right_rho,
                   sub_u_turn_lower,
                   u_turn,
                   left._height + 1,
                   log_weight,
                   energy_max,
                   energy_min,
                   sample)


class NUTSOrbit:
    def __init__(self,
                 sampler,
                 rng,
                 theta,
                 rho,
                 stepsize,
                 number_fine_grid_leapfrog_steps,
                 bernoulli_sequence,
                 tree_node_class=NUTSTreeNode):

        self._sampler = sampler
        self._rng = rng
        self._theta = theta
        self._rho = rho
        self._stepsize = stepsize
        self._number_fine_grid_leapfrog_steps = number_fine_grid_leapfrog_steps
        self._bernoulli_sequence = bernoulli_sequence
        self._sampler._stepsize = stepsize
        self._tree_node_class = tree_node_class
        self._orbit_root = self.coarse_fine_nuts()
        self.set_reverse_bernoulli_sequence_for_sample()

    def coarse_fine_nuts(self):
        energy_max = energy_min = -self._sampler.log_joint(self._theta, self._rho)
        nuts_root_node = self._tree_node_class.initialize_leaf(self,
                                                               self._theta,
                                                               self._rho,
                                                               energy_max,
                                                               energy_min)

        for forward in self._bernoulli_sequence:
            if forward:
                proposal_root_node = self.evaluate_forward(nuts_root_node._right_theta,
                                                           nuts_root_node._right_rho,
                                                           nuts_root_node._height)
                root_for_merged = self.combine_top_level_trees(nuts_root_node, proposal_root_node)
            else:
                proposal_root_node = self.evaluate_backward(nuts_root_node._left_theta,
                                                            nuts_root_node._left_rho,
                                                            nuts_root_node._height)
                root_for_merged = self.combine_top_level_trees(proposal_root_node, nuts_root_node)

            if proposal_root_node._sub_u_turn:
                return nuts_root_node

            if root_for_merged._u_turn:
                return root_for_merged

            nuts_root_node = root_for_merged
        return nuts_root_node

    def evaluate_forward(self, right_theta_existing_tree, right_rho_existing_tree, height):
        if height == 0:
            (left_theta_forward_subtree,
             left_rho_forward_subtree,
             energy_max_interior,
             energy_min_interior) = self.iterated_leapfrog_with_energy_max_min(right_theta_existing_tree,
                                                                               right_rho_existing_tree)

            return self._tree_node_class.initialize_leaf(self,
                                                         left_theta_forward_subtree,
                                                         left_rho_forward_subtree,
                                                         energy_max_interior,
                                                         energy_min_interior)

        root_left_subtree = self.evaluate_forward(right_theta_existing_tree,
                                                  right_rho_existing_tree,
                                                  height - 1)

        if root_left_subtree._sub_u_turn:
            return root_left_subtree

        root_right_subtree = self.evaluate_forward(root_left_subtree._right_theta,
                                                   root_left_subtree._right_rho,
                                                   height - 1)
        if root_right_subtree._sub_u_turn:
            return root_right_subtree

        return self.combine_lower_level_trees(root_left_subtree, root_right_subtree)

    def evaluate_backward(self, left_theta_existing_tree, left_rho_existing_tree, height):
        time_reversed_node = self.evaluate_forward(left_theta_existing_tree, -left_rho_existing_tree, height)
        forward_time_node = time_reversed_node.time_reverse()
        return forward_time_node

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

    def combine_lower_level_trees(self, left_root_node, right_root_node):
        new_sample = self.resample_sub_tree(left_root_node, right_root_node)
        new_root_node = self._tree_node_class.combine_lower_level_trees(left_root_node, right_root_node, new_sample)
        return new_root_node

    def resample_sub_tree(self, left_root_node, right_root_node):
        if np.log(self._rng.uniform(0, 1)) < right_root_node._log_weight - np.logaddexp(left_root_node._log_weight,
                                                                                        right_root_node._log_weight):

            return NUTSSample(right_root_node._sample._theta, right_root_node._sample._rho,
                              right_root_node._sample._bernoulli_sequence + (0,))
        else:

            return NUTSSample(left_root_node._sample._theta, left_root_node._sample._rho,
                              left_root_node._sample._bernoulli_sequence + (1,))

    def combine_top_level_trees(self, left_root_node, right_root_node):
        new_sample = self.resample_top_tree(left_root_node, right_root_node)
        new_root_node = self._tree_node_class.combine_top_level_trees(left_root_node, right_root_node, new_sample)
        return new_root_node

    def resample_top_tree(self, left_root_node, right_root_node):
        if np.log(self._rng.uniform(0, 1)) < right_root_node._log_weight - np.logaddexp(left_root_node._log_weight,
                                                                                        right_root_node._log_weight):

            return NUTSSample(right_root_node._sample._theta, right_root_node._sample._rho,
                              right_root_node._sample._bernoulli_sequence + (0,))
        else:

            return NUTSSample(left_root_node._sample._theta, left_root_node._sample._rho,
                              left_root_node._sample._bernoulli_sequence + (1,))

    def energy_gap(self):
        return self._orbit_root._energy_max - self._orbit_root._energy_min

    def set_reverse_bernoulli_sequence_for_sample(self):
        sample_sequence = self._orbit_root._sample._bernoulli_sequence
        orbit_sequence = self._bernoulli_sequence
        new_sequence = sample_sequence + orbit_sequence[len(sample_sequence):]
        self._orbit_root._sample._bernoulli_sequence = new_sequence

    def sample(self):
        return self._orbit_root._sample

    def sample_coordinates(self):
        return self._orbit_root._sample._theta, self._orbit_root._sample._rho

    def sample_bernoulli_sequence(self):
        return self._orbit_root._sample._bernoulli_sequence
