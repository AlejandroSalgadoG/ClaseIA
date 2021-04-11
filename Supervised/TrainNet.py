import numpy as np
import pandas as pd
from NNet.NNet import NNet
from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

#n = 100
#x = np.random.randint(0, 2, size=(n,2) )
#y = (x[:,0] & x[:,1]).reshape(-1,1)
#y = np.hstack( [y, ~y.astype(bool)] )

dataset = load_breast_cancer( as_frame=True )

x = dataset["data"].values
x = (x - x.mean()) / x.std()

y = dataset["target"]
y_inv = ~y.astype(bool)
y = pd.concat([y, y_inv.astype(int)], axis=1).values

x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3, shuffle=True )

epoch = 2000
tol = 1e-3

arch = [30, 20, 10, 2]
eta = 0.2
batch_size = 100

nnet = NNet(arch)
weights = nnet.init_random_weights()

def get_batch(x, y, n):
    idx = np.random.choice( np.arange(n), size=batch_size, replace=False )
    return x[idx], y[idx]

for i in range( epoch ):
    batch_x, batch_y = get_batch( x_train, y_train, batch_size )
    errors = np.zeros(batch_size)
    for idx, (input_data, label) in enumerate(zip(batch_x, batch_y)):
        error, weights = nnet.train(input_data, label, weights, eta)
        errors[idx] = error
    print(i, errors.mean(), end="\r" )
print()

y_hat = np.array( [ nnet.predict(input_data, weights) for input_data in x_test ] )

zero = y_hat[:,0] < 0.5

y_hat[zero, 0] = 0
y_hat[~zero, 0] = 1

conf_matrix = confusion_matrix( y_test[:,0], y_hat[:,0] )
print( conf_matrix )
