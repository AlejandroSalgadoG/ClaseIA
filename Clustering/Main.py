from Data import X, img_n, img_m

from Kmeans import kmeans
from FuzzyKMeans import fuzzy_kmeans
from Mountain import mountain
from Substractive import substract
from Agglomerative import agglomerative

from Utils import *
from PlotResults import *

C, M = kmeans( X, num_c=7, iters=10 )
plot_result2d( X, M, C )
plot_result3d( X, M, C )
plot_as_img( img_n, img_m, X, M )

#C, U = fuzzy_kmeans( X, num_c=7, iters=10 )
#M, Umax = fuzzy_to_membership( U )
#plot_result2d( X, M, C, Umax )
#plot_result3d( X, M, C, Umax )
#plot_as_img( img_n, img_m, X, M, Umax )

#C = mountain( X , num_c=7, num_div=5)
#M = calculate_membership( X, C )
#plot_result2d( X, M, C)
#plot_result3d( X, M, C)
#plot_as_img( img_n, img_m, X, M )

#C = substract( X, num_clusters=7, num_divisions=5)
#M = calculate_membership( X, C )
#plot_result2d( X, M, C )
#plot_result3d( X, M, C )
#plot_as_img( img_n, img_m, X, M )

#M = agglomerative(X, num_clusters=7 )
#plot_result2d( X, M )
#plot_result3d( X, M )
#plot_as_img( img_n, img_m, X, M )
