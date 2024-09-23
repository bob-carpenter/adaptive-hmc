from dataclasses import dataclass
import numpy as np

@dataclass
class NUTSSample:
    '''
    Class to store the sample of the NUTS algorithm

    Attributes:
        _theta: np.array(float64): The position value of the sample
        _rho: np.array(float64): The momentum value of the sample
        _bernoulli_sequence: tuple(int): The sequence of 0s and 1s that determine the
            direction of the NUTS trajectory that produced the sample when started FROM the sample.
            i.e. B^* in the companion paper
    '''
    _theta: float
    _rho: float
    _bernoulli_sequence: tuple

    def reverse_time(self):
        '''
        Reverses the time of the sample by flipping the sign of the momentum and the bernoulli sequence

        Returns: None
        '''
        self._bernoulli_sequence = tuple(1 - x for x in self._bernoulli_sequence)
        self._rho = -self._rho


@dataclass
class NUTSTreeNode:
    '''
    Class to store the tree nodes of the NUTS algorithm.

    Attributes:
        _left_theta: np.array(float64): The leftmost position value of the subtree rooted at the node
        _left_rho: np.array(float64): The leftmost momentum value of the subtree rooted at the node
        _right_theta: np.array(float64): The rightmost position value of the subtree rooted at the node
        _right_rho: np.array(float64): The rightmost momentum value of the subtree rooted at the node
        _sub_u_turn: bool: Whether any subtree of the node has a U-turn
        _u_turn: bool: Whether the node itself has a U-turn
        _height: int: The height of the node in the tree
        _log_weight: float: The log weight of the node
        _energy_max: float: The maximum energy over the tree rooted at the node (ie. over the leaves of the tree)
        _energy_min: float: The minimum energy over the tree rooted at the node (ie. over the leaves of the tree)
        _sample: NUTSSample: Sample point in the trajectory represented by the node. Contains the position, momentum,
            and transformed bernoulli sequence of the sample.
    '''
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
        '''
        Class method to initialize a leaf node of the NUTS tree
        Args:
            parent_orbit: NUTSOrbit: The parent orbit of the node
            theta: np.array(float64): position represented by the leaf node
            rho: np.array(float64): momentum represented by the leaf node
            energy_max: float: Maximum energy of leaf node. Takes max over intermediate leapfrog steps computed
                between positions. Hence, may be greater than the energy of the leaf node and different
                from the energy min.
            energy_min: float: Minimum energy of leaf node. Takes min over intermediate leapfrog steps computed
                between positions. Hence, may be lower than the energy of the leaf node and different
                from the energy max.
        Returns:
            NUTSTreeNode: The initialized leaf node
        '''
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
        '''
        Reverses the time of the node by flipping the sign of the momentum and the orientation of position values

        '''
        self._left_theta, self._right_theta = self._right_theta, self._left_theta
        self._left_rho = -self._right_rho
        self._right_rho = -self._left_rho
        self._sample.reverse_time()
        return self

    @classmethod
    def nuts_style_u_turn(cls, left_theta, left_rho, right_theta, right_rho):
        '''
        Determines if a U-turn has occurred between the left and right positions and momenta
        Args:
            left_theta: np.array(float64): The leftmost position value
            left_rho: np.array(float64): The leftmost momentum value
            right_theta: np.array(float64): The rightmost position value
            right_rho: np.array(float64): The rightmost momentum value

        Returns:
            Bool: True if a U-turn has occurred, False otherwise

        '''
        delta_theta = right_theta - left_theta
        return (np.dot(delta_theta, left_rho) < 0) or (np.dot(delta_theta, right_rho) < 0)

    @classmethod
    def combine_lower_level_trees(cls, left, right, sample):
        '''
        Combines two lower level trees into a single tree. This is implemented separately in
        anticipation of biased sampling in future implementations.
        Args:
            left: NUTSTreeNode: The root node of the left subtree
            right: NUTS TreeNode: The root node of the right subtree
            sample: Sample from the combined tree. This is sampling is
            done externally to the tree and passed in as an argument.

        Returns:
            NUTSTreeNode: The root node of the combined tree
        '''
        new_node = cls.combine_top_level_trees(left, right, sample)
        new_node._sub_u_turn = new_node._sub_u_turn or new_node._u_turn
        return new_node

    @classmethod
    def combine_top_level_trees(cls, left, right, sample):
        '''
        Combines two top level trees into a single tree. This is implemented separately in
        anticipation of biased sampling in future implementations.

        Args:
            left: NUTSTreeNode: The root node of the left tree
            right: NUTSTreeNode: The root node of the right tree
            sample: Sample from the combined tree. This is sampling is
            done externally to the tree and passed in as an argument.

        Returns:
            NUTSTreeNode: The root node of the combined tree

        '''
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
    '''
    Class to create an orbit of the NUTS algorithm. This class is nearly a wrapper around the NUTSTreeNode class
    and is used to create the orbit of the NUTS algorithm.

    Attributes:
        _sampler: The NUTS sampler object that is using the orbit
        _rng: np.random.Generator: The random number generator
        _theta: np.array(float64): The initial position value of the orbit
        _rho: np.array(float64): The initial momentum value of the orbit
        _stepsize: float: The step size of the NUTS algorithm
        _number_fine_grid_leapfrog_steps: int: The number of leapfrog steps to take on the fine grid
        _bernoulli_sequence: tuple(int): The sequence of 0s and 1s that determine the direction of the NUTS trajectory
        _tree_node_class: NUTSTreeNode: The class to use for the tree nodes. This is used for testing purposes
        _orbit_root: NUTSTreeNode: The root node of the orbit tree.
    '''
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
        '''
        Runs the NUTS algorithm to create the orbit tree. This is the main method of the class.
        Returns:
            NUTSTreeNode: The root node of the orbit tree

        '''
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
        '''
        Evaluates the trajectory of the NUTS algorithm in the forward in time direction. This is done recursively
        with a single pass over the leapfrog iterates and logarithmic memory usage.
        Args:
            right_theta_existing_tree: np.array(flaot64): Rightmost position value of the existing tree
            right_rho_existing_tree: np.array(float64): Rightmost momentum value of the existing tree
            height: int: Current height of the tree

        Returns:
            NUTSTreeNode: The root node of the subtree evaluated in the forward direction

        '''
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
        '''
        Evaluates the trajectory of the NUTS algorithm in the backward in time direction. This is done recursively.
        Due the symmetry of the NUTS algorithm and the leapfrog integrator, this is the same as evaluating the
        forward trajectory with the sign of the momentum reversed. We apply time reversal to the node evaluated
        in the end to get the correct orientation of the tree.

        Args:
            left_theta_existing_tree: np.array(float64): The leftmost position value of the existing tree
            left_rho_existing_tree: np.array(float64): The leftmost momentum value of the existing tree
            height: int: height of the tree

        Returns:
            NUTSTreeNode: The root node of the subtree evaluated in the backward direction

        '''
        time_reversed_node = self.evaluate_forward(left_theta_existing_tree, -left_rho_existing_tree, height)
        forward_time_node = time_reversed_node.time_reverse()
        return forward_time_node

    def iterated_leapfrog_with_energy_max_min(self, theta, rho):
        '''
        Runs the leapfrog integrator for a fixed number of "meta" steps and computes the maximum and minimum energy
        along the intermediate points in the trajectory.
        Args:
            theta: np.array(float64): The initial position
            rho: np.array(float64): The initial momentum

        Returns:
            theta_current: np.array(float64): The final position
            rho_current: np.array(float64): The final momentum
            max_energy: float: The maximum energy along the trajectory
            min_energy: float: The minimum energy along the trajectory

        '''
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
        '''
        Combines two lower level trees into a single tree. This is implemented separately in
        anticipation of biased sampling in future implementations. This method produces a sample from
        the combined tree then uses the class method to combine the trees.
        Args:
            left_root_node: NUTSTreeNode: The root node of the left subtree
            right_root_node: NUTSTreeNode: The root node of the right subtree

        Returns:
            NUTSTreeNode: The root node of the combined tree

        '''
        new_sample = self.resample_sub_tree(left_root_node, right_root_node)
        new_root_node = self._tree_node_class.combine_lower_level_trees(left_root_node, right_root_node, new_sample)
        return new_root_node

    def resample_sub_tree(self, left_root_node, right_root_node):
        '''
        Produces a sample from the combined tree using samples from the left and right trees and their weights:
        Args:
            left_root_node: NUTSTreeNode: The root node of the left subtree
            right_root_node: NUTSTreeNode: The root node of the right subtree

        Returns:
            NUTSSample: The sample from the combined tree

        '''
        if np.log(self._rng.uniform(0, 1)) < right_root_node._log_weight - np.logaddexp(left_root_node._log_weight,
                                                                                        right_root_node._log_weight):

            return NUTSSample(right_root_node._sample._theta, right_root_node._sample._rho,
                              right_root_node._sample._bernoulli_sequence + (0,))
        else:

            return NUTSSample(left_root_node._sample._theta, left_root_node._sample._rho,
                              left_root_node._sample._bernoulli_sequence + (1,))

    def combine_top_level_trees(self, left_root_node, right_root_node):
        '''
        Combines two top level trees into a single tree. This is implemented separately in
        anticipation of biased sampling in future implementations. This method produces a sample from
        the combined tree then uses the class method to combine the trees.

        Args:
            left_root_node:
            right_root_node:

        Returns:

        '''
        new_sample = self.resample_top_tree(left_root_node, right_root_node)
        new_root_node = self._tree_node_class.combine_top_level_trees(left_root_node, right_root_node, new_sample)
        return new_root_node

    def resample_top_tree(self, left_root_node, right_root_node):
        '''
        Produces a sample from the combined tree using samples from the left and right trees and their weights.
        This is the main method to change in order to implement biased sampling.

        Args:
            left_root_node: NUTSTreeNode: The root node of the left subtree
            right_root_node: NUTSTreeNode: The root node of the right subtree

        Returns:
            NUSSample: The sample from the combined tree

        '''
        if np.log(self._rng.uniform(0, 1)) < right_root_node._log_weight - np.logaddexp(left_root_node._log_weight,
                                                                                        right_root_node._log_weight):

            return NUTSSample(right_root_node._sample._theta, right_root_node._sample._rho,
                              right_root_node._sample._bernoulli_sequence + (0,))
        else:

            return NUTSSample(left_root_node._sample._theta, left_root_node._sample._rho,
                              left_root_node._sample._bernoulli_sequence + (1,))

    def energy_gap(self):
        '''
        Returns the energy gap of the orbit. This is the difference between the maximum and minimum energy
        along the trajectory.

        Returns:
            float: The energy gap of the orbit

        '''
        return self._orbit_root._energy_max - self._orbit_root._energy_min

    def set_reverse_bernoulli_sequence_for_sample(self):
        '''
        Sets the reverse of the Bernoulli sequence for the sample. This is used to re-run the step-size
        adaptation from the sample in order to implement the Metropolis-Hastings correction.

        Returns:
            None

        '''
        sample_sequence = self._orbit_root._sample._bernoulli_sequence
        orbit_sequence = self._bernoulli_sequence
        new_sequence = sample_sequence + orbit_sequence[len(sample_sequence):]
        self._orbit_root._sample._bernoulli_sequence = new_sequence

    def sample(self):
        '''
        Returns the sample of the orbit. This is the sample from the trajectory represented by the
            root node of the orbit tree.
        Returns:
            NUTSSample: The sample of the orbit

        '''
        return self._orbit_root._sample

    def sample_coordinates(self):
        '''
        Returns the position and momentum of the sample of the orbit.

        Returns:
            np.array(float64): The position of the sample
            np.array(float64): The momentum of the sample

        '''
        return self._orbit_root._sample._theta, self._orbit_root._sample._rho

    def sample_bernoulli_sequence(self):
        '''
        Returns the Bernoulli sequence of the sample. This is used to re-run the step-size adaptation from the sample
        in order to implement the Metropolis-Hastings correction.

        Returns:
            tuple(int): The Bernoulli sequence of the sample

        '''
        return self._orbit_root._sample._bernoulli_sequence
