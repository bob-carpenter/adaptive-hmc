import numpy as np

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

def uturn(theta, rho, stepsize, model):
    theta_next = theta.copy()
    rho_next = rho.copy()
    last_dist_sq = 0
    L = 0
    while True:
        theta_next, rho_next = leapfrog_step(theta_next, rho_next, stepsize, model)
        diff = theta_next - theta
        dist_sq = np.sum(diff**2)
        if dist_sq < last_dist_sq:
            return L
        last_dist_sq = dist_sq
        L += 1

def draw(theta, rng, stepsize, L, model):
    rho = rng.normal(size = model.dims())
    logp = joint_logp(theta, rho, model)
    N = uturn(theta.copy(), rho.copy(), stepsize, model)
    L = rng.integers(1, N + 1)
    theta_prop, rho_prop = leapfrog(theta.copy(), rho.copy(), stepsize, L, model)
    rho_prop = -rho_prop
    Nb = uturn(theta_prop.copy(), rho_prop.copy(), stepsize, model)
    # print(f"{N = }  {Nb = }")
    logp_prop = joint_logp(theta_prop, rho_prop, model)
    # vs. < np.minimum(1, np.exp(logp_prop - logp) * np.exp(np.log(N) - np.log(Nb)))
    if rng.uniform() < np.minimum(1, np.exp(logp_prop - logp)) * np.minimum(1, np.exp(np.log(N) - np.log(Nb))):
        return theta_prop
    return theta

def sample(M, theta0, rng, stepsize, L, model):
    theta = np.empty((M, model.dims()))
    theta[0, :] = theta0
    for m in range(1, M):
        theta[m, :] = draw(theta[m - 1, :], rng, stepsize, L, model)
    return theta    

class StdNormal:
    def __init__(self):
        ""

    def log_density(self, x):
        return -0.5 * np.dot(x, x)

    def log_density_gradient(self, x):
        return self.log_density(x), -x

    def dims(self):
        return 1

M = 1000
theta0 = np.array([0.2])
rng = np.random.default_rng()
stepsize = 0.99
L = 15
model = StdNormal()
sample = sample(M, theta0, rng, stepsize, L, model)
print(f"   mean: {np.mean(sample, axis=0)[0]:6.2f}")
print(f"std dev: {np.std(sample, axis=0)[0]:6.2f}")


