import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# radius (mean of distances from center to points on the perimeter)
# texture (standard deviation of gray-scale values)
# perimeter
# area
# smoothness (local variation in radius lengths)
# compactness (perimeter^2 / area - 1.0)
# concavity (severity of concave portions of the contour)
# concave points (number of concave portions of the contour)
# symmetry
# fractal dimension ("coastline approximation" - 1)

# ===================================== ====== ======
#                                        Min    Max
# ===================================== ====== ======
# radius (mean):                        6.981  28.11
# texture (mean):                       9.71   39.28
# perimeter (mean):                     43.79  188.5
# area (mean):                          143.5  2501.0
# smoothness (mean):                    0.053  0.163
# compactness (mean):                   0.019  0.345
# concavity (mean):                     0.0    0.427
# concave points (mean):                0.0    0.201
# symmetry (mean):                      0.106  0.304
# fractal dimension (mean):             0.05   0.097
# radius (standard error):              0.112  2.873
# texture (standard error):             0.36   4.885
# perimeter (standard error):           0.757  21.98
# area (standard error):                6.802  542.2
# smoothness (standard error):          0.002  0.031
# compactness (standard error):         0.002  0.135
# concavity (standard error):           0.0    0.396
# concave points (standard error):      0.0    0.053
# symmetry (standard error):            0.008  0.079
# fractal dimension (standard error):   0.001  0.03
# radius (worst):                       7.93   36.04
# texture (worst):                      12.02  49.54
# perimeter (worst):                    50.41  251.2
# area (worst):                         185.2  4254.0
# smoothness (worst):                   0.071  0.223
# compactness (worst):                  0.027  1.058
# concavity (worst):                    0.0    1.252
# concave points (worst):               0.0    0.291
# symmetry (worst):                     0.156  0.664
# fractal dimension (worst):            0.055  0.208
# ===================================== ====== ======

dataset = load_breast_cancer( as_frame=True )

x = dataset["data"]
y = dataset["target"]

# corr = x.corr()
# fig, ax = plt.subplots()
# sns.heatmap(corr, vmax=1, center=0, linewidths=.5, square=True, cbar_kws={"shrink": .5} )

# ax.set_xticks( np.arange(30) + 0.5 )
# ax.set_yticks( np.arange(30) + 0.5 )
# ax.set_xticklabels( x.columns )
# ax.set_yticklabels( x.columns )

# plt.tight_layout()
# plt.show()

def plot_data3d(x, y, title=""):
    ax = plt.axes(projection='3d')
    ax.scatter3D(x[:, 0], x[:, 1], x[:, 2], c=y, cmap='viridis')
    if title:
        plt.title(title)
    plt.show()

x_pca = PCA(n_components=3).fit_transform(x)
plot_data3d(x_pca, y, "PCA")


x_embedded = TSNE(n_components=3, init="pca").fit_transform(x)
plot_data3d(x_embedded, y, "TSNE")

