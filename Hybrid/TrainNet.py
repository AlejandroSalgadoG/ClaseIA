import numpy as np
np.random.seed(1234567)

from NNet import train
from Functions import *

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

dataset = load_breast_cancer( as_frame=True )
x,y = dataset["data"], dataset["target"]
x,y = normalize( x.values ), classes2binary( y.values )
x_train, x_test, y_train, y_test = train_test_split( x, y, test_size=0.3, shuffle=True )
nnet, weights = train( epoch=500, tol=1e-2, arch=[30, 20, 2], eta=0.5, x=x_train, y=y_train )

y_hat = predict( nnet, weights, x_test )
y_pred = binary2class( hard_classification( y_hat ) )
y_true = binary2class( y_test )

conf_matrix = confusion_matrix( y_true, y_pred )
tn, fp, fn, tp = conf_matrix.ravel()

accu = accuracy( tn, fp, fn, tp )
sens = sensitivity( fn, tp )
spec = specificity( tn, fp )
roc_auc = roc_auc_score( y_true, y_pred )

print( "accuracy", "%.6f" % accu )
print( "sensitivity", "%.6f" % sens )
print( "specificity", "%.6f" % spec )
print( "roc_auc", "%.6f" % roc_auc )
