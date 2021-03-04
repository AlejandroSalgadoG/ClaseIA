import numpy as np

def distance(x, y):
    return np.linalg.norm( np.subtract( x, y ) )

def calculate_dist_matrix( X, C ):
    return np.array([ [ distance( x, c ) for c in C ] for x in X ])

def calculate_membership( X, C ):
    D = calculate_dist_matrix( X, C )
    return np.argmin( D, axis=1 )

def fuzzy_to_membership( U ):
    return np.argmax( U, axis=0 ), np.max( U, axis=0 )
