import numpy as np
import bridgestan as bs


class StdNormal:
    def __init__(self, dims=1):
        self._dims = dims

    def log_density(self, x):
        return -0.5 * np.dot(x, x)

    def log_density_gradient(self, x):
        return self.log_density(x), -x

    def dims(self):
        return self._dims


class StanModel:
    def __init__(self, file, data = None):
        self._model = bs.StanModel(model_lib = file, data = data,
                                       capture_stan_prints = False)
        
    def dims(self):
        return self._model.param_num()

    def log_density(self, theta):
        return self._model.log_density(theta)

    def log_density_gradient(self, theta):
        return self._model.log_density_gradient(theta)
