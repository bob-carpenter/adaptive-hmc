import AAPS_Metropolized_Adapt as AAPSAdapt
import hmc


class PureAAPS(hmc.HmcSamplerBase):
    def __init__(self,
                 model,
                 rng,
                 theta,
                 step_size,
                 number_apogees):
        super().__init__(model, step_size, rng)
        self._max_step_size = step_size
        self._theta = theta
        self._number_apogees = number_apogees

    def draw(self):
        theta = self._theta
        rho = self._rng.normal(size=self._model.param_unc_num())
        coarse_interval_current = AAPSAdapt.CoarseLevelIntervalAAPS(self,
                                                                    self._rng,
                                                                    self._theta,
                                                                    self._rho,
                                                                    self._stepsize
                                                                    )
        shift = self._rng.integers(0, self._number_apogees)
        coarse_interval_current.set_num_apogees_and_shift(self._number_apogees, shift)
        coarse_interval_current.populate_interval()

        return coarse_interval_current.proposal_phase_space()

    def set_stepsize(self, stepsize):
        self._stepsize = stepsize

    def stepsize_reset_original(self):
        self._stepsize = self._max_step_size

    def log_density_gradient_at_theta(self, theta):
        return self._model.log_density_gradient(theta)
