import cmdstanpy as csp
import numpy as np
import util as ut
import logging
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings( "ignore", module = "plotnine\..*" )
import cmdstanpy as csp
csp.utils.get_logger().setLevel(logging.ERROR)

model = csp.CmdStanModel(stan_file = "../stan/normal.stan")

# IID
def iid_normal():
    D = 5        # dims
    N = 10     # runs
    M = 10_000   # draws/run
    msq_jumps = np.empty(N)
    sq_err_Y = np.empty((N, D))
    sq_err_Ysq = np.empty((N, D))
    for n in range(N):
        draws_y = np.random.normal(size=(M, D))
        msq_jumps[n] = ut.mean_sq_jump_distance(draws_y)
        sq_err_Y[n, :] = np.mean(draws_y, axis=0)**2
        sq_err_Ysq[n, :] = (np.mean(draws_y**2, axis=0) - 1)**2
    print("I.I.D.")
    print(f"   Y std err: {np.sqrt(sq_err_Y.reshape(N * D).sum() / (N * D))}")        
    print(f"Y**2 std err: {np.sqrt(sq_err_Ysq.reshape(N * D).sum() / (N * D))}")        
    print(f"mean sq jump: {np.mean(msq_jumps):5.1f}  std-dev msq jump: {np.std(msq_jumps):4.2f}")

# NUTS
def nuts_normal():
    D = 5        # dims
    N = 10    # runs
    M = 10_000   # draws/run
    msq_jumps = np.empty(N)
    sq_err_Y = np.empty((N, D))
    sq_err_Ysq = np.empty((N, D))
    for n in range(N):
        sample = model.sample(data = { 'D': 5 }, chains = 4, iter_sampling = M // 4,
                                show_console=False, show_progress=False)
        draws_y = sample.stan_variable('y')
        msq_jumps[n] = ut.mean_sq_jump_distance(draws_y)
        sq_err_Y[n, :] = np.mean(draws_y, axis=0)**2
        sq_err_Ysq[n, :] = (np.mean(draws_y**2, axis=0) - 1)**2
    print("NUTS")
    print(f"   Y std err: {np.sqrt(sq_err_Y.reshape(N * D).sum() / (N * D))}")        
    print(f"Y**2 std err: {np.sqrt(sq_err_Ysq.reshape(N * D).sum() / (N * D))}")        
    print(f"mean sq jump: {np.mean(msq_jumps):5.1f}  std-dev msq jump: {np.std(msq_jumps):4.2f}")
    print("\n")

# NUTS 8 schools
def nuts_eight_schools():
    D = 10        # dims
    N = 10     # runs
    M = 10   # draws/run
    model = csp.CmdStanModel(stan_file = "../stan/eight-schools.stan")
    msq_jumps = np.empty(N)
    sq_err_Y = np.empty((N, D))
    sq_err_Ysq = np.empty((N, D))
    for n in range(N):
        sample = model.sample(data = "../stan/eight-schools.json", chains = 4, iter_sampling = M // 4,
                                show_console=False, show_progress=False)
        draws_y = sample.draws(concat_chains = True)
        msq_jumps[n] = ut.mean_sq_jump_distance(draws_y)
        sq_err_Y[n, :] = np.mean(draws_y, axis=0)**2
        sq_err_Ysq[n, :] = (np.mean(draws_y**2, axis=0) - 1)**2
    print("NUTS")
    print(f"   Y std err: {np.sqrt(sq_err_Y.reshape(N * D).sum() / (N * D))}")        
    print(f"Y**2 std err: {np.sqrt(sq_err_Ysq.reshape(N * D).sum() / (N * D))}")        
    print(f"mean sq jump: {np.mean(msq_jumps):5.1f}  std-dev msq jump: {np.std(msq_jumps):4.2f}")
    print("\n")


iid_normal()
nuts_normal()
# nuts_eight_schools()
