import numpy as np

def add_bias(x):
    ans = np.append(x,1)
    return np.expand_dims(ans, 0)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def d_sigmoid(x):
    ans = x*(1-x)
    return np.expand_dims(ans, 0)  

def mse(y, y_bar):
    return np.sum((y-y_bar)**2/2)

def d_mse(y, y_bar):
    ans = y_bar-y
    return np.expand_dims(ans, 0)

def standardize( x ):
    return (x - x.mean(axis=0)) / x.std(axis=0)

def normalize( x ):
    return (x - x.min(axis=0)) / (x.max(axis=0) - x.min(axis=0))

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

def hard_classification( y_hat, threshold=0.5 ):
    y_hard = np.zeros( y_hat.shape )
    y_hard[ y_hat > threshold ] = 1
    return y_hard

def predict( nnet, weights, x ):
    return np.array( [ nnet.predict(input_data, weights) for input_data in x ] )

def get_batch(x, y, batch_size):
    idx = np.random.choice( np.arange(batch_size), size=batch_size, replace=False )
    return x[idx], y[idx]

def accuracy( tn, fp, fn, tp ):
    return (tn+tp) / (tn+fp+fn+tp)

def sensitivity( fn, tp ):
    return tp / (tp+fn)

def specificity( tn, fp ):
    return tn / (tn+fp)
