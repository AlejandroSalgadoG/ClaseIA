from Data import X
from Kmeans import kmeans
from FuzzyKMeans import fuzzy_kmeans
from Mountain import mountain
from Substractive import substract

#C, M = kmeans( X )
C, U = fuzzy_kmeans( X, iters=10 )
#mountain( X , num_divisions=5)
#substract( X , num_divisions=5)
