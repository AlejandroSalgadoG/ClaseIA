import numpy as np

def distance(x, y):
    return np.linalg.norm(x-y)

def select_centers( n, num_c ):
    return np.random.choice( n, size=num_c, replace=False )

def calc_membership(X, C):
    M = { i: [] for i,_ in enumerate(C) }
    for i,x in enumerate( X ):
        c_sel = np.argmin( [ distance( x, c ) for c in C ] )
        M[ c_sel ].append( i )
    return M

def cost_function(X, C, M):
    cost = 0
    for i,c in enumerate(C):
        for j in M[i]: 
            cost += distance( c, X[j] )
    return cost

def update_centers(X, C, M):
    for i,_ in enumerate(C):
        C[i] = np.sum( X[ M[i] ], axis=0 ) / len(M[i])
    return C

def kmeans( X, num_c=2, ite=10 ):
    n,m = X.shape
    C = X[ select_centers( n, num_c ) ]
    for i in range(ite):
        M = calc_membership( X, C )
        cost = cost_function( X, C, M )
        C = update_centers( X, C, M )
    return C, M
