import numpy as np
from NNet.NNet import NNet

epoch = 500
tol = 1e-2

arch = [2, 2, 2]
input_data = [0.05, 0.1]
labels = [0.01, 0.99]
eta = 0.5

nnet = NNet(arch)
weights = nnet.init_random_weights()

for i in range( epoch ):
    error, weights = nnet.train(input_data, labels, weights, eta)
    output = nnet.predict(input_data, weights)
    print( i, error, output )
    if error <= tol: break
