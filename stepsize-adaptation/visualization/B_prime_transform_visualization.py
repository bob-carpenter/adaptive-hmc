import numpy as np

import Fixed_step_size_NUTS_simulation as fv
import NUTSOrbitDiagnostic as nod
import step_size_adapt_NUTS_b_prime_transform

seed = 12909067
model = fv.create_model_stan_and_json("funnel", "funnel")
rng = np.random.default_rng(seed)

theta_0 = np.zeros(model.param_unc_num())
sampler = step_size_adapt_NUTS_b_prime_transform.StepAdaptNUTSMetro(model,
                                                                    rng,
                                                                    theta_0,
                                                                    np.zeros(model.param_unc_num()),
                                                                    0.7,
                                                                    1 / 4,
                                                                    10,
                                                                    10,
                                                                    nod.NUTSOrbitDiagnostic)
sampler.draw()

print(f"First orbit = {nod.NUTSOrbitDiagnostic.get_orbits()[0]._coarse_grid_points}")
