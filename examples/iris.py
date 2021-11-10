import numpy as np
from sklearn import datasets
from gln.context import rand_halfspace
from gln.nnet import Layer, GLN
from gln.util import binary_entropy 

if __name__ == "__main__":
    iris = datasets.load_iris()

    # binary classifier for the second class (iris-setosa)
    targ = np.where( iris["target"] == 1, 1, 0)
    data = iris["data"]

    dims = [4,25,25,1]
    layers = []
    for i in range(3):
        ctxts = [rand_halfspace(4, var=10, bias=4) for _ in range(dims[i+1])]
        layers.append( Layer(dims[i], dims[i+1], ctxts) )
    model = GLN(layers)

    # training loop
    lr = 0.01
    losses = []
    for i in np.random.permutation(len(data)):
        x, y = data[i], targ[i]
        out = model(x)
        model.step(y, lr)
        loss = binary_entropy(y, logits=out[-1]) # negative log likelihood of the prediction
        losses.append(loss)

        if len(losses) >= 5: 
            print(f"avg loss ----------- {np.mean(losses):.3f}")
            losses = []



