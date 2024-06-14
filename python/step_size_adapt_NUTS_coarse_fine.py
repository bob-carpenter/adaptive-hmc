import hmc
import numpy as np
import scipy as sp
import traceback

#This file contains many comments at Nawaf's request
class StepadaptNutsCoarseFineSampler(hmc.HmcSamplerBase):
    def __init__(self,
                 model,
                 rng,
                 theta,
                 min_accept_prob,
                 max_step_size,
                 max_step_size_search_depth,
                 max_nuts_depth):

        super().__init__(model, 0.0, rng)
        self._theta = theta
        self._log_min_accept_prob = np.log(min_accept_prob)
        self._max_step_size= max_step_size
        self._max_step_size_search_depth = max_step_size_search_depth
        self._number_fine_grid_leapfrog_steps = 1
        self._max_nuts_search_depth = max_nuts_depth
        self._bernoulli_sequence = self._rng.integers(low = 0, high = 2, size = self._max_nuts_search_depth)

    def draw(self):
        self._stepsize = self._max_step_size
        self._number_fine_grid_leapfrog_steps = 1
        self._bernoulli_sequence = self._rng.integers(low=0, high=2, size=self._max_nuts_search_depth)
        #Reset the step_size, the number of leapfrog steps, and the entire sequence of Bernoulli draws

        self._rho = self._rng.normal(size=self._model.param_unc_num())
        theta, rho = self._theta, self._rho
        #Redraw the momentum, set theta, rho to the current state

        for i in range(self._max_step_size_search_depth):
            try:
                #Begin searching the step_size refinement process
                (theta_prime,
                 rho_prime,
                 energy_max,
                 energy_min) = self.Coarse_Fine_NUTS(theta, rho,self._max_nuts_search_depth)
                #The new iterates are drawn from the modified version of NUTS
                #The integrator gets more refined in each stage, but NUTS evaluates
                #the No-U-Turn condition only at iterates corresponding to the coarest grid points

                if -(energy_max - energy_min)> self._log_min_accept_prob:
                    self._theta = theta_prime
                    return theta_prime, rho_prime
                    #Return once the energy change meets the acceptance criterion

                self._stepsize = self._stepsize / 2
                self._number_fine_grid_leapfrog_steps  *= 2
            except Exception as e:
                #If we encounter an error, cut the step size in half
                #and double the number of intermediate leapfrog steps
                traceback.print_exc()
                self._stepsize = self._stepsize / 2
                self._number_fine_grid_leapfrog_steps *= 2
            #If we haven't met the acceptance criterion, cut the step size in half
            #and double the number of "intermediate" leapfrog steps between
            #corresponding points on the coarse grid

        self._theta = theta_prime
        return self._theta, self._rho
        #If we still haven't acheived acceptable energy error after the
        #maximum number of halvings just return the sample from the last state

    def Coarse_Fine_NUTS(self, theta, rho, max_height):
        #The interval from which NUTS samples will be specified by its left and right endpoints
        #and log of the number of points in this interval - which is the same as the number of
        #doublings and the height of the corresponding binary tree.

        #These right and left endpoints are updated dynamically, along with a sample
        #from the corresponding interval

        left_theta = theta
        left_rho = rho
        #Leapfrog iterate corresponding to the left/earlier in time
        #endpoint of the NUTS interval
        right_theta = theta
        right_rho = rho
        #Leapfrog iterate corresponding to the right/later in timer
        #endpoint of the NUTS interval
        height = 0
        #Height of the current NUTS tree = log(size of interval) = number of "doublings"
        sample_theta = theta
        sample_rho = rho
        #Dynamically updated sample
        weight_current = np.exp(self.log_joint(theta, rho))
        weight_new = 0
        #Weight of the current interval and a dummy for the weight of the prposed interval
        #Used for dynamically updated sample
        energy_max = energy_min = -self.log_joint(theta, rho)
        #Inital energy maximum and minimum over the FINE gridpoints


        for i in range(max_height):
            #forward_or_backward_choice = self._rng.integers(0, 2)
            forward_or_backward_choice = self._bernoulli_sequence[i]
            #Select extension forward or backward
            if forward_or_backward_choice == 0:
                #Backward extension
                # We evaluate the subtree corresponding to an extension assuming it was FORWARD in time
                # To get the "backward" in time version, just flip velocity and flip back at the end
                (extension_theta,
                 extension_rho,
                 energy_max_interior_fine_grid,
                 energy_min_interior_fine_grid) = self.iterated_leapfrog_with_energy_max_min(left_theta,
                                                                                             -left_rho)
                #Get the left endpoint of the proposed extension and return the maximum and minimum of the
                #energy over any intermediate fine gridpoints between the left endpoint of the current
                #interval and the right endpoint of the proposed extension

                (left_theta,
                 left_rho,
                 extension_sample_theta,
                 extension_sample_rho,
                 weight_new,
                 sub_u_turn,
                 extension_energy_max,
                 extension_energy_min) = self.evaluate_proposed_subtree(extension_theta,
                                                                        extension_rho,
                                                                        height)
                #Evaluates tree corresponding to the proposed extension. Since this is backward in time,
                #this returns the left endpoint of the proposed extension interval with its momentum flipped,
                #a sample from the proposed extension, the total boltzmann weight at the coarse gridpoints of
                #the proposed extension, whether the extension contains a sub-u-turn, and the energy max and min
                #at the fine gridpoints

                left_rho = -left_rho
                #Flip the momentum back

            else:
                (extension_theta,
                 extension_rho,
                 energy_max_interior_fine_grid,
                 energy_min_interior_fine_grid) = self.iterated_leapfrog_with_energy_max_min(right_theta, right_rho)
                #Same as the above without the momentum flip - since in this case we're already
                #integrating forward in time

                (right_theta,
                 right_rho,
                 extension_sample_theta,
                 extension_sample_rho,
                 weight_new,
                 sub_u_turn,
                 extension_energy_max,
                 extension_energy_min) = self.evaluate_proposed_subtree(extension_theta,
                                                                        extension_rho,
                                                                        height)
                #Same return signature as above
            if sub_u_turn:
                #If the proposal had a sub-u-turn just throw it away and return the current state
                return sample_theta, sample_rho, energy_max, energy_min

            sample_theta, sample_rho = self.resample_top(sample_theta,
                                                     sample_rho,
                                                     extension_sample_theta,
                                                     extension_sample_rho,
                                                     weight_current,
                                                     weight_new)
            height += 1
            weight_current += weight_new
            energy_max = max(energy_max, energy_max_interior_fine_grid, extension_energy_max)
            energy_min = min(energy_min, energy_min_interior_fine_grid, extension_energy_min)
            #If the proposal had no sub-u-turn the above updates the state to include the extension

            if self.nuts_style_u_turn(left_theta, left_rho, right_theta, right_rho):
            #If the current interval has a u-turn then return the state
                return sample_theta, sample_rho, energy_max, energy_min

        return sample_theta, sample_rho, energy_max, energy_min
        #Return the state if we've hit the maximum size -regardless of everything else


    def evaluate_proposed_subtree(self, theta, rho, height):
        # -> (right_theta, right_rho, sample_theta, sample_rho, weight, sub_u_turn, energy_max, energy_min)
        #This evaluates the proposed sub-tree recursively.
        #Give the left endpoint of the interval defining this tree as well as the height
        #Return the right endpoint of this interval, a sample from the interval
        #the total weight of the interval, the sub-u-turn indicator,
        #and the energy max and min over fine grid points

        if height == 0:
            #For a tree of height zero, the right endpoint is the same as the left
            #the sample is just the current point, the weight is the Boltzmann weight
            #of the current point, it has no sub-u-turns, and the max and min at fine
            #grid points is just the current energy
            return (theta,
                    rho,
                    theta,
                    rho,
                    np.exp(self.log_joint(theta, rho)),
                    False,
                    -self.log_joint(theta, rho),
                    -self.log_joint(theta, rho))

        (theta_right_subtree_left,
         rho_right_subtree_left,
         sample_theta_subtree_left,
         sample_rho_subtree_left,
         weight_subtree_left,
         sub_u_turn_subtree_left,
         energy_max_fine_grid_left,
         energy_min_fine_grid_left) = self.evaluate_proposed_subtree(theta, rho, height - 1)
        #Evaluate all the relevant quantities over the left subtree. This has the same left endpoint
        #as the current tree, but with height one less

        if sub_u_turn_subtree_left:
            return (theta, rho, theta, rho, 0, True,0, 0)
            #Immediately return if the subtree has a u-turn

        (theta_left_subtree_right,
         rho_left_subtree_right,
         energy_max_interior_fine_grid,
         energy_min_interior_fine_grid) = self.iterated_leapfrog_with_energy_max_min(theta_right_subtree_left,
                                                                                     rho_right_subtree_left)
        #Get the left endpoint of the right subtree. This is \Phi^{2^{number of stepsize_halvings}}
        #applied to the right endpoint of the left subtree. We also get the energy max and min
        #over the fine gridpoints.
        #As implemented, coarse gridpoints will be included twice in the energy max and min
        #computation, but this has no effect on the result.

        (theta_right_subtree_right,
         rho_right_subtree_right,
         sample_theta_subtree_right,
         sample_rho_subtree_right,
         weight_subtree_right,
         sub_u_turn_subtree_right,
         energy_max_fine_grid_right,
         energy_min_fine_grid_right) = self.evaluate_proposed_subtree(theta_left_subtree_right,
                                                                      rho_left_subtree_right,
                                                                      height - 1)
        #Obtain all the corresponding quantities over the right subtree

        sub_u_turn = (sub_u_turn_subtree_left) or (sub_u_turn_subtree_right) or self.nuts_style_u_turn(theta_right_subtree_left,
                                                                                          rho_right_subtree_left,
                                                                                          theta_right_subtree_right,
                                                                                          rho_right_subtree_right)
        #Evaluate subintervals and the current interval for sub-u-turns

        if sub_u_turn:
            #Immediately return if the subtree has a u-turn
            return (theta, rho, theta, rho, 0, True,0, 0)

        sample_theta, sample_rho = self.resample_sub_tree(sample_theta_subtree_left,
                                                          sample_rho_subtree_left,
                                                          sample_theta_subtree_right,
                                                          sample_rho_subtree_right,
                                                          weight_subtree_left,
                                                          weight_subtree_right)
        #Produce a new sample from those in the left and right subtrees

        weight_subtree = weight_subtree_left + weight_subtree_right
        energy_max = max(energy_max_fine_grid_left, energy_max_interior_fine_grid, energy_max_fine_grid_right)
        energy_min = min(energy_min_fine_grid_left, energy_min_interior_fine_grid, energy_min_fine_grid_right)
        #Combine the results from the left and right subtrees

        return (theta_right_subtree_right,
                rho_right_subtree_right,
                sample_theta,
                sample_rho,
                weight_subtree,
                sub_u_turn,
                energy_max,
                energy_min)

    #For "biased progressive sampling" the below functions would be different
    #for this implementation they are the same
    def resample_top(self, sample_theta, sample_rho, new_sample_theta, new_sample_rho, weight_current, weight_new):
        #Update the dynamic sample in the NUTS trajectory
        if self._rng.uniform(0, 1) < weight_new / (weight_current + weight_new):
            return new_sample_theta, new_sample_rho
        else:
            return sample_theta, sample_rho

    def resample_sub_tree(self, sample_theta, sample_rho, new_sample_theta, new_sample_rho, weight_current, weight_new):
        #Update the sample during the recursive sub-tree evaluation for the NUTS extension.
        if self._rng.uniform(0, 1) < weight_new / (weight_current + weight_new):
            return new_sample_theta, new_sample_rho
        else:
            return sample_theta, sample_rho

    def nuts_style_u_turn(self, left_theta, left_rho, right_theta, right_rho):
        #Return whether or not the interval defined by the given endpoints
        #has a u-turn
        delta_theta = right_theta - left_theta
        return (np.dot(delta_theta, left_rho) < 0) and (np.dot(delta_theta, right_rho) < 0)

    def iterated_leapfrog_with_energy_max_min(self, theta, rho):
        #Compute \Phi^{2^{number of stepsize_halvings}} while simultaneously computing the energy
        #and min along the extension
        max_energy = min_energy = -self.log_joint(theta, rho)
        theta_current = theta
        rho_current = rho

        for i in range(self._number_fine_grid_leapfrog_steps):
            theta_current, rho_current = self.leapfrog_step(theta_current, rho_current)
            current_energy = -self.log_joint(theta_current, rho_current)
            max_energy = max(max_energy, current_energy)
            min_energy = min(min_energy, current_energy)

        return theta_current, rho_current, max_energy, min_energy




