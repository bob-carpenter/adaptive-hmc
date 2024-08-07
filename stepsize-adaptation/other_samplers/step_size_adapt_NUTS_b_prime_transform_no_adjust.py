import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import samplers.step_size_adapt_NUTS_b_prime_transform as nuts_b_prime_transform


class NUTSBPrimeNoAdjust(nuts_b_prime_transform.StepAdaptNUTSMetro):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._name = "NUTS_b_prime_transform_no_adjust"
        self._acceptance_ratios = []

    def compute_acceptance_probability(self, adapted_step_size_from_initial, adapted_step_size_from_sample):
        return 1
