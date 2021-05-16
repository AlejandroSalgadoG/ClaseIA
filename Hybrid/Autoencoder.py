import numpy as np
import pandas as pd

np.random.seed(1234567)
np.set_printoptions(suppress=True)

from NNet import train
from Functions import *

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

dataset = load_breast_cancer( as_frame=True )

x = dataset["data"].iloc[:,:]
x = normalize( x.values ) + 0.005

#x_train, x_test, y_train, y_test = train_test_split( x, x, test_size=0.3, shuffle=True )

nnet, weights = train( epoch=1000, tol=1e-10, arch=[30, 3, 30], eta=0.9, x=x, y=x, batch_size=100 )
x_pred = predict( nnet, weights, x )

diff = (x - x_pred)/x
diff = np.abs(diff) * 100

print( diff.shape )
for i in range(30):
    d = diff[:,i]
    d_clean = d[ d < 50]
    print(i, d_clean.shape, "%.2f" % d_clean.mean(), "%.2f" % d_clean.std() )
