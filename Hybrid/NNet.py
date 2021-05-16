import numpy as np

from Functions import *

class NNet:
    def __init__(self, arch):
        self.arch = arch
        self.n_layers = len(arch)-1

    def predict(self, data, weights):
        for layer in range( self.n_layers ):
            data = np.append(data, 1)
            w_data = np.matmul(data, weights[layer])
            data = sigmoid(w_data)
        return data

    def test_prediction(self, data, weights):
        inter_data = []
        for layer in range( self.n_layers ):
            inter_data.append(data)
            data = np.append(data, 1)
            w_data = np.matmul(data, weights[layer])
            data = sigmoid(w_data)
        inter_data.append(data)
        return data, inter_data

    def train(self, data, label, weights, eta):
        updates = [[]] * self.n_layers
        y_bar, inter_data = self.test_prediction(data, weights)
        error = mse(label, y_bar)
        d_error = d_mse(label, y_bar)
        for b_layer in range(self.n_layers,0,-1):
            d_out = d_sigmoid(inter_data[b_layer])
            d_in = add_bias(inter_data[b_layer-1])
            d_w = np.matmul(d_in.T, d_error*d_out)
            updates[b_layer-1] = d_w
            d_error = np.matmul(weights[b_layer-1][:-1], (d_error*d_out).T).T
        new_weights = [ weight - update*eta for weight, update in zip( weights, updates ) ]
        return error, new_weights

    def init_random_weights(self, low=0, high=1):
        return [ np.random.uniform( low, high, size=(self.arch[i]+1, self.arch[i+1]) ) for i in range(self.n_layers) ]

def train( epoch, tol, arch, eta, x, y, batch_size=100 ):
    print("arch", arch)
    print("eta", eta)
    print("batch", batch_size)

    nnet = NNet(arch)
    weights = nnet.init_random_weights()

    errors = np.zeros( x.shape[0] )

    for i in range( epoch ):
        batch_x, batch_y = get_batch( x, y, batch_size )
        for idx, (input_data, label) in enumerate(zip(batch_x, batch_y)): 
            errors[idx], weights = nnet.train(input_data, label, weights, eta)
        mean_error = errors.mean()
        if mean_error < tol: break
        print(i, "%.8f" % mean_error, end="\r" )
    print()

    return nnet, weights
