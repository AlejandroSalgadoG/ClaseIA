import numpy as np
from itertools import product

def distance(x, y):
    return np.linalg.norm(np.subtract(x, y))

def density_function( x, v, denominator ):
    return np.exp( - distance( x, v ) / denominator )

def substraction_function( X, v, denominator):
    return np.sum( [ density_function( x, v, denominator) for x in X ] )

def calculate_substraction_vector(X, ra):
    return [ substraction_function( X, x, sigma) for x in X ]

def select_first_center(M, V):
    return V[ np.argmax(M) ]

def update_substraction( X, V, M, c, rb):
    M_c = np.max(M)
    for i,_ in enumerate(M):
        M[i] -= M_c * density_function( V[i], c, beta )
    return M

def substract(X, num_clusters=2, num_divisions=1, ra=1.0, rb=None):
    if not rb:
        rb = 1.5 * ra
    M = calculate_substraction_vector( X, denominator=(ra/2)**2 )
    C = []
    for i in range(num_clusters):
        c = select_first_center(M, X)
        M = update_substraction(X, M, c, denominator=(ra/2)**2)
        C.append( c )
    return C
