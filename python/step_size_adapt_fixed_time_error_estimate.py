import VanillaNUTS as vn
import numpy as np
import traceback

#This is the most basic way to do the adaptation. I have another idea
#based on AAPS-style shifted intervals
class NUTS_local_adapt(vn.VanillaNUTS):
    def __init__(self,
                 model,
                 rng,
                 theta,
                 rho,
                 min_acceptance_probability,
                 energy_estimate_steps,
                 max_stepsize,
                 max_stepsize_search_depth,
                 max_nuts_depth
                 ):
        super().__init__(model, rng, theta, rho, 0, max_nuts_depth)
        self._energy_gap_bound = -np.log(min_acceptance_probability)
        self._energy_estimate_steps = energy_estimate_steps

        self._max_stepsize_search_depth = max_stepsize_search_depth
        self._max_stepsize = max_stepsize
        self._acceptance_ratios = []

    def draw(self):
        self._rho = self._rng.normal(size=self._model.param_unc_num())
        theta, rho = self._theta, self._rho

        current_position_adapted_step_size = self.adapt_step_size(theta, rho)
        self._stepsize = current_position_adapted_step_size.proposal_step_size()
        nuts_orbit = self.NUTS(theta, rho, self._max_nuts_search_depth)
        theta_prime, rho_prime = nuts_orbit.sample_coordinates()
        proposal_position_adapted_step_size = self.adapt_step_size(theta_prime, rho_prime)
        acceptance_probability = self.compute_acceptance_probability(current_position_adapted_step_size,
                                                                     proposal_position_adapted_step_size)
        self._acceptance_ratios.append(min(1,acceptance_probability))
        if self._rng.uniform() < acceptance_probability:
            self._theta = theta_prime
            self._rho = rho_prime
        return self._theta, self._rho

    def adapt_step_size(self, theta, rho):
        return AdaptedStepSize(self, self._rng, theta, rho)

    def energy_estimate_steps(self):
        return self._energy_estimate_steps

    def max_step_size(self):
        return self._max_stepsize
    def max_step_size_search_depth(self):
        return self._energy_estimate_steps

    def energy_gap_bound(self):
        return self._energy_gap_bound

    def compute_acceptance_probability(self, current_position_adapted_step_size,
                                       proposal_position_adapted_step_size):

        # I think this is the last bit I need to figure out.
        log_proposed_step_size = current_position_adapted_step_size.log_proposal_step_size()
        log_likelihood_from_proposal = proposal_position_adapted_step_size.step_size_log_likelihood(log_proposed_step_size)
        log_likelihood_from_current = current_position_adapted_step_size.step_size_log_likelihood(log_proposed_step_size)
        return np.exp(log_likelihood_from_proposal - log_likelihood_from_current)

    def acceptance_ratios(self):
        return self._acceptance_ratios

class AdaptedStepSize:
    def __init__(self, sampler, rng, theta, rho):
        self._std = np.log(2)/2
        self._sampler = sampler
        self._rng = rng
        self._log_h_star = self.log_h_star(theta, rho)
        self._log_proposal_step_size = self.compute_log_proposal_step_size(self._log_h_star)
            # This is constant for now but I might play with it later

    def leapfrog_energy_gap_dynamic_step_size(self, theta, rho, step_size, number_steps):
        original_step_size = self._sampler._stepsize
        self._sampler._stepsize = step_size
        #Save the original step_size and modify the step_size to what we want
        energy_max = energy_min = -self._sampler.log_joint(theta, rho)
        for i in range(number_steps):
            try:
                theta, rho = self._sampler.leapfrog_step(theta, rho)
                energy = -self._sampler.log_joint(theta, rho)
                energy_max = max(energy, energy_max)
                energy_min = min(energy, energy_min)
            except:
                traceback.print_exc()
        self._sampler._stepsize = original_step_size
        return energy_max, energy_min

    def log_h_star(self, theta, rho):
        number_steps = self._sampler.energy_estimate_steps()
        max_step_size = self._sampler.max_step_size()
        for i in range(self._sampler.max_step_size_search_depth()):
            step_size = max_step_size * 2 ** (-i)
            (energy_max_forward,
             energy_min_forward) = self.leapfrog_energy_gap_dynamic_step_size(theta,
                                                                              rho,
                                                                              step_size,
                                                                              number_steps*(2 ** i))
            (energy_max_backward,
                energy_min_backward) = self.leapfrog_energy_gap_dynamic_step_size(theta,
                                                                                -rho,
                                                                                step_size,
                                                                                number_steps*(2 ** i))
            energy_max = max(energy_max_forward, energy_max_backward)
            energy_min = min(energy_min_forward, energy_min_backward)
            if energy_max - energy_min < self._sampler.energy_gap_bound():
                return np.log(max_step_size) - (i+1) * np.log(2)

        return np.log(max_step_size) - self._sampler.max_step_size_search_depth()* np.log(2)

    def compute_log_proposal_step_size(self, log_h_star):
        return log_h_star + self._rng.normal(0, self._std)

    def log_proposal_step_size(self):
        return self._log_proposal_step_size

    def proposal_step_size(self):
        return np.exp(self._log_proposal_step_size)

    def step_size_log_likelihood(self, log_proposal):
        normal_deviate = (log_proposal - self._log_h_star)/self._std
        log_likelihood = -(normal_deviate**2)/2
        return log_likelihood

