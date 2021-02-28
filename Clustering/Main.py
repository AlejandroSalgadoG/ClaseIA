from Data import X
from Kmeans import kmeans
from FuzzyKMeans import fuzzy_kmeans
from Mountain import mountain
from Substractive import substract
from Agglomerative import agglomerative
import numpy as np

#C, M = kmeans( X )
# C, U = fuzzy_kmeans( X, iters=10 )
#mountain( X , num_divisions=5)
D = []
for i in range(len(X)):
    norm = ((X[i] - X) ** 2).sum(axis=1)
    exp_i = np.exp(- norm /  (1 / 2) ** 2)
    D_i = exp_i.sum()
    D.append(D_i)


c1 = np.array([[2., 2.]])
Di1 = []
for i in range(len(X)):
    norm = ((X[i] - c1) ** 2).sum()
    exp = np.exp (- norm /  (1.5 / 2) ** 2)
    D_i1 = D[i] - np.max(D) * exp 
    Di1.append(D_i1)

c2 = np.array([[1., 1.]])
Di2 = []
for i in range(len(X)):
    norm = ((X[i] - c2) ** 2).sum()
    exp = np.exp (- norm /  (1.5 / 2) ** 2)
    D_i2 = Di1[i] - np.max(Di1) * exp 
    Di2.append(D_i2)

print(D)
print(Di1)
print(Di2)
print("Comienza programa")
clusters = substract( X , num_divisions=5, num_clusters=2)
print(clusters)
# clusters = agglomerative(X, num_clusters=2)
# print(clusters)