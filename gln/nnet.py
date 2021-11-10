import numpy as np
from .util import sigmoid

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

    def step(self, v, lr):
        self.weights[np.arange(self.out_dim),:,self._cs] -= lr*v

    def __call__(self, lgts, x):
        self._cs = [c(x) for c in self.ctxt]
        W = self.weights[np.arange(self.out_dim),:,self._cs]
        return np.dot(W,lgts)

class GLN:
    def __init__(self, layers):
        self.layers = layers
        self.depth = len(layers)
        
        for i in range(self.depth-1):
            assert(layers[i].out_dim == layers[i+1].in_dim)
        
        self.outputs = None
    
    def step(self, targ, lr=0.01):
        for i, lyr in enumerate(self.layers):
            inp, out = self.outputs[i], self.outputs[i+1]
            grad = np.outer( (sigmoid(out)-targ), inp )
            lyr.step(grad,lr)
    
    def __call__(self, x):
        data = x.copy()
        self.outputs = [data]
        for lyr in self.layers:
            x = lyr(x, data)
            self.outputs.append(x)
        return self.outputs
