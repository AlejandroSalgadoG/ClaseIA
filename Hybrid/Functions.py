import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score

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

def encode( nnet, weights, x ):
    return np.array( [ nnet.encode(input_data, weights) for input_data in x ] )

def get_batch(x, y, batch_size):
    idx = np.random.choice( np.arange(batch_size), size=batch_size, replace=False )
    return x[idx], y[idx]

def accuracy( tn, fp, fn, tp ):
    return (tn+tp) / (tn+fp+fn+tp)

def sensitivity( fn, tp ):
    return tp / (tp+fn)

def specificity( tn, fp ):
    return tn / (tn+fp)

def get_autoencoder_results( y, y_pred, threshold=50 ):
  n,_ = y.shape
  diff = np.abs( (y - y_pred)/y ) * 100
  results = []
  for i, d_var in enumerate(diff.T):
      d_clean = d_var[ d_var < threshold ]
      results.append( [i, d_clean.size, d_clean.size/n * 100, d_clean.mean(), d_clean.std() ] )
  return np.array( results )

def get_net_results( y, y_hat ):
  y_true = binary2class( y )
  y_pred = binary2class( hard_classification( y_hat ) )
  
  conf_matrix = confusion_matrix( y_true, y_pred )
  tn, fp, fn, tp = conf_matrix.ravel()
  
  accu = accuracy( tn, fp, fn, tp )
  sens = sensitivity( fn, tp )
  spec = specificity( tn, fp )
  roc_auc = roc_auc_score( y_true, y_pred )
  
  return accu, sens, spec, roc_auc
