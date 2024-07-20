import Fixed_step_size_NUTS_simulation as fv
import numpy as np
import NUTSOrbit as nuts
import VanillaNUTS as vn

model = fv.create_model_stan_and_json("funnel", "funnel")
D = model.param_unc_num()
seed = int("              ".encode("utf-8").hex(), 16)
print(f"Seed = {seed}")
number_fine_grid_steps = 2048*16
step_size = 1/number_fine_grid_steps
#seed = 12909067
#seed = 12734
rng = np.random.default_rng(seed)
theta0 = np.zeros(D)
theta0[0] = rng.normal(0, 3)
theta0[1:D] = rng.normal(scale=np.exp(theta0[0] / 2), size=(D - 1))
rho0 = rng.normal(size=D)
sampler = vn.VanillaNUTS(model,
                 rng,
                 theta0,
                 rho0,
                 0,
                 10)

bernoulli_sequence = tuple(rng.binomial(1, 0.5, 10))
#bernoulli_sequence = (0,0,0,0,0)
print("Computing NUTS orbit from initial point and from sample \n")
nuts_orbit = nuts.NUTSOrbit(sampler, rng, theta0, rho0, step_size, number_fine_grid_steps, bernoulli_sequence)
sample = nuts_orbit._orbit_root._sample
print(" \nComputing NUTS orbit from sample \n")
nuts_orbit_from_sample = nuts.NUTSOrbit(sampler, rng, sample._theta, sample._rho, step_size, number_fine_grid_steps, sample._bernoulli_sequence)

print(f"Left endpoint and momentum as computed from initial point: theta = {nuts_orbit._orbit_root._left_theta}, rho = {nuts_orbit._orbit_root._left_rho}\n")
print(f"Left endpoint as computed from sample: theta = {nuts_orbit_from_sample._orbit_root._left_theta}, rho = {nuts_orbit_from_sample._orbit_root._left_rho}\n")
print(f"Right endpoint and momentum as computed from initial point: theta = {nuts_orbit._orbit_root._right_theta}, rho = {nuts_orbit._orbit_root._right_rho}\n")
print(f"Right endpoint as computed from sample: theta = {nuts_orbit_from_sample._orbit_root._right_theta}, rho = {nuts_orbit_from_sample._orbit_root._right_rho}\n")

print(f"Left endpoints equal = {np.allclose(nuts_orbit._orbit_root._left_theta, nuts_orbit_from_sample._orbit_root._left_theta)}")
print(f"Right endpoints equal = {np.allclose(nuts_orbit._orbit_root._right_theta, nuts_orbit_from_sample._orbit_root._right_theta)}")
print(f"Left momenta equal = {np.allclose(nuts_orbit._orbit_root._left_rho, nuts_orbit_from_sample._orbit_root._left_rho)}")
print(f"Right momenta equal = {np.allclose(nuts_orbit._orbit_root._right_rho, nuts_orbit_from_sample._orbit_root._right_rho)}")

print(f"Initial point: theta = {theta0}, rho = {rho0}")
print(f"Sample point: theta = {sample._theta}, rho = {sample._rho}")
print(f"Orginal sequence = {bernoulli_sequence}")
print(f"Sample sequence = {sample._bernoulli_sequence}")

print(f"Energy gap from initial = {nuts_orbit.energy_gap()}")
print(f"Energy max from initial = {nuts_orbit._orbit_root._energy_max}")
print(f"Energy min from initial = {nuts_orbit._orbit_root._energy_min}")


print(f"Energy gap from sample = {nuts_orbit_from_sample.energy_gap()}")
print(f"Energy max from sample = {nuts_orbit_from_sample._orbit_root._energy_max}")
print(f"Energy min from sample = {nuts_orbit_from_sample._orbit_root._energy_min}")

print(f"Energy gap the same = {np.allclose(nuts_orbit.energy_gap(), nuts_orbit_from_sample.energy_gap())}")
print(f"Energy max the same = {np.allclose(nuts_orbit._orbit_root._energy_max, nuts_orbit_from_sample._orbit_root._energy_max)}")
print(f"Energy min the same = {np.allclose(nuts_orbit._orbit_root._energy_min, nuts_orbit_from_sample._orbit_root._energy_min)}")

print(f"Final depth from initial = {nuts_orbit._orbit_root._height}")
print(f"Final depth from sample = {nuts_orbit_from_sample._orbit_root._height}")

