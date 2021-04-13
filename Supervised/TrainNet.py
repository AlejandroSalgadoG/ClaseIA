import numpy as np

from NNet.NNet import NNet

from sklearn.metrics import confusion_matrix
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

def normalize( x ):
    return (x - x.mean()) / x.std()

def classes2binary( y ):
    classes = np.unique(y)
    bin_classes = np.zeros( shape=( y.size, classes.size ) )
    for i, c in enumerate(y): bin_classes[i,c] = 1
    return bin_classes

def binary2class( y ):
    n,_ = y.shape
    classes = np.zeros( n )
    for i, c in enumerate(y): classes[i] = np.argmax(c)
    return classes

def get_batch(x, y, batch_size):
    idx = np.random.choice( np.arange(batch_size), size=batch_size, replace=False )
    return x[idx], y[idx]

def hard_classification( y_hat, threshold=0.5 ):
    y_hard = np.zeros( y_hat.shape )
    y_hard[ y_hat > threshold ] = 1
    return y_hard

dataset = load_breast_cancer( as_frame=True )

x,y = dataset["data"], dataset["target"]
x,y = normalize( x.values ), classes2binary( y.values )
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3, shuffle=True )

epoch = 500
tol = 1e-2

arch = [30, 15, 2]
eta = 0.2
batch_size = 100

nnet = NNet(arch)
weights = nnet.init_random_weights()

errors = np.zeros(batch_size)
for i in range( epoch ):
    batch_x, batch_y = get_batch( x_train, y_train, batch_size )
    for idx, (input_data, label) in enumerate(zip(batch_x, batch_y)):
        errors[idx], weights = nnet.train(input_data, label, weights, eta)
    error = errors.mean()
    if error < tol: break
    print(i, "%.8f" % error, end="\r" )
    errors[:] = 0
print()

y_hat = np.array( [ nnet.predict(input_data, weights) for input_data in x_test ] )
y_hard = hard_classification( y_hat, threshold=0.5 )

conf_matrix = confusion_matrix( binary2class( y_test ), binary2class( y_hard ) )
print( conf_matrix )
