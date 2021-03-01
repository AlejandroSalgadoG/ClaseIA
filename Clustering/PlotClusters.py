import numpy as np
import matplotlib.pyplot as plt

from Colors import get_color

def extract_components( X ):
    return X[:,0], X[:,1]

def plot_kmeans( X, C, M ):
    X1, X2 = extract_components( X )
    C1, C2 = extract_components( C )

    c_colors = [ get_color(i) for i,_ in enumerate(C) ]
    x_colors = [ get_color(m) for m in M ]

    plt.scatter( C1, C2, c=c_colors, edgecolors='k', marker="X", s=100 )
    plt.scatter(X1, X2, c=x_colors, edgecolors='k')
    plt.show()


def plot_fuzzy_kmeans( X, C, U ):
    X1, X2 = extract_components( X )
    C1, C2 = extract_components( C )
    M = np.argmax( U, axis=0 )
    Umax = np.max( U, axis=0 )

    c_colors = [ get_color(i) for i,_ in enumerate(C) ]
    x_colors = [ get_color(m, alpha=u) for m,u in zip( M, Umax ) ]

    plt.scatter( C1, C2, c=c_colors, edgecolors='k', marker="X", s=100 )
    plt.scatter(X1, X2, c=x_colors, edgecolors='k')
    plt.show()

def plot_mountain( X, C, M ):
    plot_kmeans( X, C, M )

def plot_substract( X, C, M ):
    plot_kmeans( X, C, M )

def plot_agglomerative( R ):
    for i,X in enumerate(R):
        X1, X2 = extract_components( X )
        plt.scatter(X1, X2, color=get_color(i), edgecolors='k')
    plt.show()
