import numpy as np
import matplotlib.pyplot as plt
import step_size_adapt_NUTS_metropolized as adaptNUTS
import Fixed_step_size_NUTS_log_sigma_histogram as fn
import time
import os

current_directory = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(f"{current_directory}/results/HighDimensionalGaussian"):
    os.makedirs(f"{current_directory}/results/HighDimensionalGaussian")

results_directory = f"{current_directory}/results/HighDimensionalGaussian"

print(f"Step size adaptation for high-dimensional Gaussian")
num_chains = 800
num_samples = 100
h_max = 0.5
num_halvings = np.empty((num_chains, num_samples-1))

print("Initializing model")
seed = 12909067
rng = np.random.default_rng(seed)
model = fn.create_model_stan_and_json("high-dim-normal", 'high-dim-normal')

start_time = time.time()
for chain in range(num_chains):
    print(f'Chain {chain}')
    theta0 = np.zeros(model.param_unc_num())
    sampler = adaptNUTS.StepAdaptNUTSMetro(model,
                                           rng,
                                           theta0,
                                           np.zeros(model.param_unc_num()),
                                           0.7,
                                           h_max,
                                           10,
                                           10)

    samples = sampler.sample_constrained(num_samples)
    num_halvings[chain] = np.array(sampler._adapted_step_sizes)

end_time = time.time()
print(f"Time taken: {end_time - start_time}")

step_sizes = h_max*np.exp2(-num_halvings)
average_step_sizes = np.mean(step_sizes, axis=0)
plt.plot(average_step_sizes)
plt.xlabel("Iteration")
plt.ylabel("Average Step Size")
plt.grid(True)
#plt.title("Average Step Size Over Iterations")
plt.savefig(f"{results_directory}/average_step_size_over_iterations_{num_chains}_chains_{num_samples}_samples.pdf",
            dpi=300)
plt.show()



