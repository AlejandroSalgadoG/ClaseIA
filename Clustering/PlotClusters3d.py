import numpy as np
import matplotlib.pyplot as plt

from Colors import get_color

def extract_components( X ):
    return X[:,0], X[:,1], X[:,2]

def plot_kmeans3d( X, C, M ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X1, X2, X3 = extract_components( X )
    C1, C2, C3 = extract_components( C )

    c_colors = [ get_color(i) for i,_ in enumerate(C) ]
    x_colors = [ get_color(m) for m in M ]

    ax.scatter( C1, C2, C3, c=c_colors, edgecolors='k', marker="X", s=100, depthshade=False )
    ax.scatter(X1, X2, X3, c=x_colors, edgecolors='k', depthshade=False)
    plt.show()

def plot_fuzzy_kmeans3d( X, C, U ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X1, X2, X3 = extract_components( X )
    C1, C2, C3 = extract_components( C )
    M = np.argmax( U, axis=0 )
    Umax = np.max( U, axis=0 )

    c_colors = [ get_color(i) for i,_ in enumerate(C) ]
    x_colors = [ get_color(m, alpha=u) for m,u in zip( M, Umax ) ]

    ax.scatter( C1, C2, C3, c=c_colors, edgecolors='k', marker="X", s=100, depthshade=False )
    ax.scatter(X1, X2, X3, c=x_colors, edgecolors='k', depthshade=False)
    plt.show()

def plot_mountain3d( X, C, M ):
    plot_kmeans3d( X, C, M )

def plot_substract3d( X, C, M ):
    plot_kmeans3d( X, C, M )

def plot_agglomerative3d( R ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i,X in enumerate(R):
        X1, X2, X3 = extract_components( X )
        ax.scatter(X1, X2, X3, color=get_color(i), edgecolors='k', depthshade=False)

    plt.show()
