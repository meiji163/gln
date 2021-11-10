import numpy as np

def logit(p):
    return np.log(p/(1.0-p))

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# probability of x, GEO(x; p,w)
def geo_mix(w, p, x=1):
    p1 = sigmoid( np.dot(w, logit(p)) )
    return p1 if x==1 else 1.-p1
    
def binary_entropy(targ, probs=None, logits=None):
    if probs is None and logits is None:
        raise ValueError('"probs" or "logits" required')
    if logits is not None:
        probs = sigmoid(logits) 

    if targ == 1:
        return -np.log(probs)
    elif targ == 0:
        return -np.log(1.-probs)
    else:
        raise ValueError('"targ" must be 0 or 1')

# gradient of loss wrt w
def grad_loss(p, w, x):
    return (geo_mix(w,p)-x) * logit(p)
