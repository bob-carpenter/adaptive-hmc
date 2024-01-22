import cmdstanpy as csp
import numpy as np

def mean_sq_jump_distance(sample):
    sq_jump = []
    M = 50
    for m in range(M - 1):
        jump = sample[m + 1, :] - sample[m, :]
        sq_jump.append(jump.dot(jump))
    return np.mean(sq_jump)

model = csp.CmdStanModel(stan_file = "normal.stan")

D = 5
N = 50
msq_jumps = np.empty(N)
sq_err_X = np.empty((N, D))
sq_err_Xsq = np.empty((N, D))
for n in range(N):
    sample = model.sample(data = { 'D': 5 }, iter_sampling=2500)
    draws_y = sample.stan_variable('y')
    msq_jumps[n] = mean_sq_jump_distance(draws_y)
    sq_err_X[n, :] = np.mean(draws_y, axis=0)**2
    sq_err_Xsq[n, :] = (np.mean(draws_y**2, axis=0) - 1)**2

print(f"   X std err: {np.sqrt(sq_err_X.reshape(N * D).sum() / (N * D))}")        
print(f"X**2 std err: {np.sqrt(sq_err_Xsq.reshape(N * D).sum() / (N * D))}")        
print(f"mean sq jump: {np.mean(msq_jumps):5.1f}  std-dev msq jump: {np.std(msq_jumps):4.2f}")
