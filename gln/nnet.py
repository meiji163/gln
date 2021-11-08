import numpy as np

# ith layer: weight matrix (w_kc)
#      k = number of neurons
#      c = c_k(z) is the context row

class Layer:
    def __init__(self, in_dim, out_dim, ctxt, **kwargs):
        self.ctxt = ctxt

        if len(ctxt) != out_dim:
            raise ValueError("Number of context functions must be equal to output dimension")
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.cdim = max([c.dim for c in self.ctxt])
        size = (self.out_dim, self.in_dim, self.cdim)

        weight_init = kwargs.get("init","zero")
        if weight_init == "zero":
            self.weights = np.zeros(size)
        elif weight_init == "rand":
            self.weights =  np.random.normal(scale=0.1,size=size)
        elif weight_init == "equal":
            self.weights = np.ones(size)/self.out_dim
        else:
            raise ValueError('"init" argument must be "zero","rand", or "equal"')

        self._cs = None

    def __call__(self, p, x):
        self._cs = [c(x) for c in self.ctxt]
        W = self.weights[np.arange(self.cdim), self._cs]
        return np.dot(W,p)
