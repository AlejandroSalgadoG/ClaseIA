import numpy as np
import pandas as pd

np.random.seed(1234567)
np.set_printoptions(precision=2)
np.set_printoptions(suppress=True)

from NNet import *
from Functions import *

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

dataset = load_breast_cancer( as_frame=True )

x = dataset["data"].iloc[:,:]
x = normalize( x.values ) + 0.005

x_train, x_val, y_train, y_val = train_test_split( x, x, test_size=0.2, shuffle=True )
x_test, y_test = x_train[:40], y_train[:40]  

autoencoder = Autoencoder(arch=[30,3,30])
weights = train( autoencoder, epoch=500, tol=1e-3, eta=0.9, x=x_train, y=x_train, batch_size=100 )

results_train = get_autoencoder_results( y_train, predict( autoencoder, weights, x_train ) )
results_test  = get_autoencoder_results( y_test , predict( autoencoder, weights, x_test  ) )
results_val   = get_autoencoder_results( y_val  , predict( autoencoder, weights, x_val   ) )

print( "train, n=", x_train.shape[0] )
print( results_train )
print( results_train.mean(axis=0) )

print( "test, n=", x_test.shape[0] )
print( results_test )
print( results_test.mean(axis=0) )

print( "validation, n=", x_val.shape[0] )
print( results_val )
print( results_val.mean(axis=0) )

x = encode( autoencoder, weights, x )
y = classes2binary( dataset["target"].values )

x_train, x_val, y_train, y_val = train_test_split( x, y, test_size=0.2, shuffle=True )
x_test, y_test = x_train[:40], y_train[:40]  

nnet = NNet(arch=[3,3,2,2])
weights = train( nnet, epoch=500, tol=1e-3, eta=0.9, x=x_train, y=y_train )

train_accu, train_sens, train_spec, train_roc_auc = get_net_results( y_train, predict( nnet, weights, x_train ) )
test_accu , test_sens , test_spec , test_roc_auc  = get_net_results( y_test , predict( nnet, weights, x_test  ) )
val_accu  , val_sens  , val_spec  , val_roc_auc   = get_net_results( y_val  , predict( nnet, weights, x_val   ) )

print( "train" )
print( "accuracy", "%.2f" % train_accu )
print( "sensitivity", "%.2f" % train_sens )
print( "specificity", "%.2f" % train_spec )
print( "roc_auc", "%.2f" % train_roc_auc )

print( "test" )
print( "accuracy", "%.2f" % test_accu )
print( "sensitivity", "%.2f" % test_sens )
print( "specificity", "%.2f" % test_spec )
print( "roc_auc", "%.2f" % test_roc_auc )

print( "validation" )
print( "accuracy", "%.2f" % val_accu )
print( "sensitivity", "%.2f" % val_sens )
print( "specificity", "%.2f" % val_spec )
print( "roc_auc", "%.2f" % val_roc_auc )
