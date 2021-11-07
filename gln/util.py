import numpy as np

def logit(p):
    return np.log(p/(1.0-p))

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# probability of x, GEO(x; p,w)
def geo_mix(w, p, x=1):
    p1 = sigmoid( np.dot(w, logit(p)) )
    return p1 if x==1 else 1.-p1
    
def loss(w, p, x):
    return -np.log(geo_mix(w,p,x))

# gradient of loss wrt w
def grad_loss(p, w, x):
    return (geo_mix(w,p)-x) * logit(p)
