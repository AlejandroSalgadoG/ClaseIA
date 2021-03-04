import numpy as np
import matplotlib.pyplot as plt

from Colors import get_plt_color, get_color

def extract_features( X, features ):
    return [ X[:, f ] for f in features ] 

def plot_data2d( X, features=[0,1] ):
    X1, X2 = extract_features( X, features )
    plt.scatter(X1, X2, edgecolors='k')
    plt.xlabel( "f1" )
    plt.xlabel( "f2" )
    plt.show()

def plot_data3d( X, features=[0,1,2] ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X1, X2, X3 = extract_features( X, features )
    ax.scatter(X1, X2, X3, edgecolors='k', depthshade=False)
    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")
    plt.show()

def plot_result2d( X, M, C=None, U=None, features=[0,1] ):
    X1, X2 = extract_features( X, features )
    if U is None: x_colors = [ get_plt_color(m) for m in M ]
    else: x_colors = [ get_plt_color(m, alpha=u) for m,u in zip( M, U ) ]
    plt.scatter(X1, X2, c=x_colors, edgecolors='k')

    if C is not None:
        C1, C2 = extract_features( C, features )
        c_colors = [ get_plt_color(i) for i,_ in enumerate(C) ]
        plt.scatter(C1, C2, c=c_colors, edgecolors='k', marker="X", s=100 )

    plt.xlabel( "f1" )
    plt.xlabel( "f2" )
    plt.show()

def plot_result3d( X, M, C=None, U=None, features=[0,1,2] ):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    X1, X2, X3 = extract_features( X, features )
    if U is None: x_colors = [ get_plt_color(m) for m in M ]
    else: x_colors = [ get_plt_color(m, alpha=u) for m,u in zip( M, U ) ]
    ax.scatter(X1, X2, X3, c=x_colors, edgecolors='k', depthshade=False)

    if C is not None:
        C1, C2, C3 = extract_features( C, features )
        c_colors = [ get_plt_color(i) for i,_ in enumerate(C) ]
        ax.scatter(C1, C2, C3, c=c_colors, edgecolors='k', marker="X", s=100, depthshade=False )

    ax.set_xlabel("f1")
    ax.set_ylabel("f2")
    ax.set_zlabel("f3")

    plt.show()

def plot_as_img( img_n, img_m, X, M, U=None ):
    origin_img = X.reshape( img_n, img_m, 3 )

    if U is None: result_img = np.array([ get_color(m) for m in M ]).reshape( img_n, img_m, 3 )
    else: result_img = np.array([ get_plt_color(m, alpha=u) for m,u in zip( M, U ) ]).reshape( img_n, img_m, 4 )

    fig, (ax1, ax2) = plt.subplots(2)
    ax1.imshow( origin_img )
    ax2.imshow( result_img )
    plt.show()
