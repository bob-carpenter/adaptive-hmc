import NUTSOrbit
import step_size_adapt_NUTS_b_prime_transform


class NUTSTreeNodeDiagnostic(NUTSOrbit.NUTSTreeNode):
    def __init_(self, *args):
        super().__init__(*args)
        self._left_child = None
        self._right_child = None
        self._parent = None
        self._intermediate_grid_points = []

    @classmethod
    def initialize_leaf(cls, orbit,
                        left_theta,
                        left_rho,
                        energy_max,
                        energy_min,
                        intermediate_grid_points=None):
        leaf = super().initialize_leaf(orbit, left_theta, left_rho, energy_max, energy_min)
        leaf._intermediate_grid_points = intermediate_grid_points
        return leaf

    @classmethod
    def combine_top_level_trees(cls, left, right, sample):
        new_root = super().combine_top_level_trees(left, right, sample)
        new_root._left_child = left
        new_root._right_child = right
        left._parent = new_root
        right._parent = new_root
        return new_root

    @classmethod
    def combine_lower_level_trees(cls, left, right, sample):
        new_root = super().combine_lower_level_trees(left, right, sample)
        new_root._left_child = left
        new_root._right_child = right
        left._parent = new_root
        right._parent = new_root
        return new_root


class NUTSOrbitDiagnostic(NUTSOrbit.NUTSOrbit):
    orbits = []

    def __init__(self,
                 sampler,
                 rng,
                 theta,
                 rho,
                 stepsize,
                 number_fine_grid_leapfrog_steps,
                 bernoulli_sequence,
                 tree_node_class=NUTSTreeNodeDiagnostic):
        self.orbits.append(self)
        super().__init__(sampler,
                         rng,
                         theta,
                         rho,
                         stepsize,
                         number_fine_grid_leapfrog_steps,
                         bernoulli_sequence,
                         tree_node_class)

    def iterated_leapfrog_with_energy_max_min(self, theta, rho):
        max_energy = min_energy = -self._sampler.log_joint(theta, rho)
        theta_current = theta
        rho_current = rho
        intermediate_grid_points = []
        for i in range(self._number_fine_grid_leapfrog_steps):
            if i > 0:
                intermediate_grid_points.append((theta_current, rho_current))
            theta_current, rho_current = self._sampler.leapfrog_step(theta_current, rho_current)
            current_energy = -self._sampler.log_joint(theta_current, rho_current)
            max_energy = max(max_energy, current_energy)
            min_energy = min(min_energy, current_energy)
        return theta_current, rho_current, max_energy, min_energy, intermediate_grid_points

    def evaluate_forward(self, right_theta_existing_tree, right_rho_existing_tree, height):
        if height == 0:
            (left_theta_forward_subtree,
             left_rho_forward_subtree,
             energy_max_interior,
             energy_min_interior,
             intermediate_grid_points) = self.iterated_leapfrog_with_energy_max_min(right_theta_existing_tree,
                                                                                    right_rho_existing_tree)

            return self._tree_node_class.initialize_leaf(self,
                                                         left_theta_forward_subtree,
                                                         left_rho_forward_subtree,
                                                         energy_max_interior,
                                                         energy_min_interior,
                                                         intermediate_grid_points)

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

    @classmethod
    def clear_orbits(cls):
        cls.orbits = []

    @classmethod
    def get_orbits(cls):
        return cls.orbits


class NUTSBprimeTransformDiagnostic(step_size_adapt_NUTS_b_prime_transform.NUTSBprimeTransform):
    def __init__(self, *args):
        super().__init__(*args)

    def set_bernoulli_sequence(self, bernoulli_sequence):
        self._bernoulli_sequence = bernoulli_sequence

    def refresh_bernoulli_sequence(self):
        return self._bernoulli_sequence

    def refresh_velocity(self):
        return self._rho
