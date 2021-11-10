import numpy as np

class Context:
    def __init__(self, in_dim, cdim, func):
        self.in_dim = in_dim
        self.dim = cdim #dimension of context space
        self.f = func
    def __call__(self, z):
        return self.f(z)

class CompositeContext(Context):
    def __init__(self, *ctxts):
        assert all([c.in_dim == ctxts[0].in_dim for c in ctxts])

        self.fs = [c.f for c in ctxts]
        self.dims = [c.dim for c in ctxts]
        p = 1
        self._prods = []
        for c in ctxts:
            self._prods.append(p)
            p *= c.dim
        self._prods = reversed(self._prods)

        composite_func = lambda x: [f(x) for f in self.fs]
        super(CompositeContext, self).__init__(ctxts[0].in_dim, p, composite_func)

    def __call__(self, z):
        return sum([p*f(z) for p, f in zip(self._prods, self.fs)])

def half_space(n, b):
    return lambda x: 1 if np.dot(x,n)-b > 0 else 0

def rand_halfspace(in_dim, planes=1, bias=1, var=1):
    bias = np.random.normal(scale=bias)
    if planes == 1:
        return Context(in_dim, 2,
                half_space( np.random.normal(scale=var, size=(in_dim)), bias))

    ctxts = []
    for _ in range(planes):
        ctxts.append(
            Context(in_dim, 2, half_space(np.random.normal(scale=var,size=(in_dim)),bias))
        )
    return CompositeContext(*ctxts)
