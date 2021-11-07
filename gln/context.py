import numpy as np

class Context:
    def __init__(self, dim, func):
        self.dim = dim
        self.f = func
    def __call__(self, z):
        return self.f(z)

def half_space(n, b):
    return lambda x: 1 if np.dot(x,n)-b > 0 else 0
