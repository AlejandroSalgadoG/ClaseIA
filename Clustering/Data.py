import numpy as np
import matplotlib.image as matimg
from sklearn.manifold import TSNE

from PlotResults import plot_data2d, plot_data3d

img = matimg.imread("images/MNM_small.jpg").astype(int)
img_n, img_m,_ = img.shape
X = img.reshape( img_n*img_m, 3 )

#X = np.vstack( [ X.T, X.mean(axis=1), X.std(axis=1) ] ).T
#Xmax, Xmin = X.max(axis=0), X.min(axis=0)
#X = (X - Xmin ) / (Xmax - Xmin)
#X_embedded = TSNE(n_components=3).fit_transform(X)

#plot_data3d( X )
#plot_data3d( X_embedded )
