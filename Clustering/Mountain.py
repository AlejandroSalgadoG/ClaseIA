import numpy as np
from itertools import product

def distance(x, y):
    return np.linalg.norm(x-y)

def density_function( x, v, param ):
    return np.exp( - distance( x, v ) / ( 2 * param**2 ) )

def mountain_function( X, v, sigma ):
    return np.sum( [ density_function( x, v, sigma ) for x in X ] )

def create_grid(X, num_divisions):
    n, m = X.shape
    maxs, mins = X.max(axis=0), X.min(axis=1)
    dimensions = [ np.linspace(mins[i], maxs[i], num_divisions) for i in range(m) ]
    generator = product( *dimensions )
    return list( generator )

def calculate_mountain(X, V, sigma):
    return [ mountain_function( X, v, sigma) for v in V ]

def select_first_center(M, V):
    return V[ np.argmax(M) ]

def update_mountain( X, V, M, c, sigma, beta ):
    M_c = np.max(M)
    for i,_ in enumerate(M):
        M[i] -= M_c * density_function( V[i], c, beta )
    return M

def mountain(X, num_clusters=2, num_divisions=1, sigma=0.1, beta=0.1):
    V = create_grid( X, num_divisions )
    M = calculate_mountain( X, V, sigma )
    C = []
    for i in range(num_clusters):
        c = select_first_center(M, V)
        M = update_mountain(X, V, M, c, sigma, beta)
        C.append( c )
    return C
