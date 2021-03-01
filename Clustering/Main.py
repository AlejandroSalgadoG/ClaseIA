from Data import X

from Kmeans import kmeans
from FuzzyKMeans import fuzzy_kmeans
from Mountain import mountain
from Substractive import substract
from Agglomerative import agglomerative

from PlotClusters import *
from PlotClusters3d import *

C, M = kmeans( X, iters=10 )
plot_kmeans( X, C, M )
#plot_kmeans3d( X, C, M )

#C, U = fuzzy_kmeans( X, iters=10 )
#plot_fuzzy_kmeans( X, C, U )
#plot_fuzzy_kmeans3d( X, C, U )

#C, M = mountain( X , num_divisions=5)
#plot_mountain( X, C, M )
#plot_mountain3d( X, C, M )

#C, M = substract( X , num_divisions=5)
#plot_substract( X, C, M )
#plot_substract3d( X, C, M )

#R = agglomerative(X)
#plot_agglomerative( R )
#plot_agglomerative3d( R )
