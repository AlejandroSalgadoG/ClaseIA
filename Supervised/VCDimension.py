import math

def calc_samples_tree(e, d, m, k):
     c = math.log( 2 ) / (2*e**2) 
     n_c = (2**k - 1) * (1 + math.log2(m) ) + 1 + math.log( 1/d )
     return c * n_c

def calc_samples_svm(e, d, m, vc):
    t1 = 4/e * math.log( 2/d )
    t2 = 8*vc/e * math.log( 13/e )
    return max( t1, t2 )

d,e = 0.9, 0.8
m = 3

n_tree = calc_samples_tree( e, d, m, 8 )
n_linear = calc_samples_svm( e, d, m, m+1 )
n_poly = calc_samples_svm( e, d, m, math.comb( m+3, 3 ) )

print( n_tree, n_linear, n_poly )
