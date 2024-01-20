import numpy as np
from scipy.stats import norm
import plotnine as pn
import pandas as pd

def joint_logp(theta, rho, model):
    return model.log_density(theta) - 0.5 * np.dot(rho, rho)

def leapfrog_step(theta, rho, stepsize, model):
    _, grad = model.log_density_gradient(theta)
    rho += 0.5 * stepsize * grad
    theta += stepsize * rho
    _, grad = model.log_density_gradient(theta)
    rho += 0.5 * stepsize * grad
    return theta, rho

def leapfrog(theta, rho, stepsize, L, model):
    for _ in range(L):
        theta, rho = leapfrog_step(theta, rho, stepsize, model)
    return theta, rho

def dGdt(theta, rho):
    return rho.dot(rho) - theta.dot(theta)

def stop_criteria(theta, rho, stepsize, model, H0):
    theta_next = theta.copy()
    rho_next = rho.copy()
    # integration time strongly depends on delta
    # delta smaller => more steps, e.g. 0.1
    # delta larger => fewer steps, e.g. 0.5
    delta = 0.3
    M = 0
    e = 0.0
    P = 0.0
    px = []
    while True:
        M += 1
        theta_next, rho_next = leapfrog_step(theta_next, rho_next, stepsize, model)
        v = dGdt(theta_next, rho_next)
        H = joint_logp(theta_next, rho_next, model)
        p = np.exp(H0 - H)
        # px.append(p)
        px.append(np.sum((theta - theta_next) ** 2))
        P += p
        e += (v * p / P - e) / M
        if np.abs(e) < delta:
            return M, theta_next, rho_next, px

def draw(theta, rng, stepsize, model):
    rho = rng.normal(size = model.dims())
    logp = joint_logp(theta, rho, model)
    direction = rng.choice([-1, 1])
    stepsize *= direction
    N, theta_N, rho_N, px = stop_criteria(theta.copy(), rho.copy(), stepsize, model, logp)
    num_leapfrogs = N
    x = np.arange(1, N + 1)
    px /= np.sum(px)
    # p = 1.0
    # px = x ** p / np.sum(x ** p)
    L = rng.choice(x, p = px)
    # L = rng.integers(1, N + 1)
    theta_prop, rho_prop = leapfrog(theta.copy(), rho.copy(), stepsize, L, model)
    num_leapfrogs += L
    logp_prop = joint_logp(theta_prop, rho_prop, model)
    if rng.uniform() < np.minimum(1, np.exp(logp_prop - logp)):
        return theta_prop, num_leapfrogs, True, N
    return theta, num_leapfrogs, False, N

def sample(M, theta0, rng, stepsize, model):
    theta = np.empty((M, model.dims()))
    theta[0, :] = theta0
    pr = 0.0
    Nm = 0.0
    for m in range(1, M):
        theta[m, :], num_leapfrogs, p, N = draw(theta[m - 1, :], rng, stepsize, model)
        pr += (p - pr) / m
        Nm += (N - Nm) / m
    return theta, num_leapfrogs, pr, Nm

class StdNormal:
    def __init__(self, dims = 1):
        self._dims = dims

    def log_density(self, x):
        return -0.5 * np.dot(x, x)

    def log_density_gradient(self, x):
        return self.log_density(x), -x

    def dims(self):
        return self._dims

# run one
# M = 1000
# theta0 = np.array([0.2])
# rng = np.random.default_rng()
# stepsize = 0.99
# model = StdNormal()
# sample = sample(M, theta0, rng, stepsize, L, model)
# print(f"   mean: {np.mean(sample, axis=0)[0]:6.2f}")
# print(f"std dev: {np.std(sample, axis=0)[0]:6.2f}")

# run ESS
M = 1_000
K = 50
model = StdNormal(10)
theta0 = np.ones(model.dims()) * 0.2
stepsize = 0.099

means = np.zeros(K)
stds = np.zeros(K)
mse_y = 0.0
mse_y_sq = 0.0
all_draws = np.zeros(shape = (K, M))
n_leapfrogs = np.zeros(K)
p = 0.0
N = 0.0

RNG0 = np.random.default_rng(204)
seeds = RNG0.integers(0, np.iinfo(np.int64).max, size = K)

for k in range(K):
    rng = np.random.default_rng(seeds[k])
    draws, num_leapfrogs, prob, Nm = sample(M, theta0, rng, stepsize, model)
    all_draws[k, :] = draws[:, 0]
    means[k] = np.mean(draws[:, 0])
    stds[k] = np.std(draws[:, 0], ddof = 1)
    mse = (means[k] - 0.0) ** 2
    mse_y += (mse - mse_y) / (k + 1)
    mse2 = (np.mean(draws[:, 0] ** 2.0) - 1.0) ** 2
    mse_y_sq += (mse2 - mse_y_sq) / (k + 1)
    n_leapfrogs[k] = num_leapfrogs
    p += (prob - p) / (k + 1)
    N += (Nm - N) / (k + 1)

mn_n_leapfrog = np.mean(n_leapfrogs)
ess_y = 1.0 / mse_y
ess_y_ave_L = ess_y / mn_n_leapfrog
ess_y_sq = 2.0 / mse_y_sq
ess_y_sq_ave_L = ess_y_sq / mn_n_leapfrog

print(f"           mean(y): {np.mean(means):.4f}")
print(f"            std(y): {np.mean(stds):.4f}")
print(f"      ave Leapfrog: {mn_n_leapfrog:.2f}")
print(f"            ESS(y): {ess_y:.2f}")
print(f"          ESS(y^2): {ess_y_sq:.2f}")
print(f"     ESS(y)/ ave L: {ess_y_ave_L:.2f}")
print(f" ESS(y ^ 2)/ ave L: {ess_y_sq_ave_L:.2f}")
print(f"    ave stop steps: {N:.2f}")
print(f"        acceptance: {p:.2f}")

df = pd.DataFrame({'x': all_draws.ravel(), "g": np.repeat(np.arange(1, K+1), M) })
df["g"] = df["g"].astype(str)

plot = ( pn.ggplot(df, pn.aes(x = 'x'))
         + pn.geom_density(pn.aes(group = "g"), color = "blue", alpha = 0.1)
         + pn.scale_color_discrete(guide = False)
         + pn.stat_function(fun=norm.pdf,
                            args={'loc': 0, 'scale': 1},
                            color='red', size=1)
        )
print(plot)
