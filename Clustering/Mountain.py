import numpy as np
from itertools import product


def distance(x, y):
    return np.linalg.norm(np.subtract(x, y))


def density_function(v, x, param):
    return np.exp(-distance(v, x) ** 2 / (2 * param ** 2))


def mountain_function(X, v, sigma):
    return np.sum([density_function(v, x, sigma) for x in X])


def create_grid(X, num_divisions):
    dimensions = np.linspace(X.max(axis=0), X.min(axis=0), num_divisions).T
    return list(product(*dimensions))


def calculate_mountain(X, V, sigma):
    return [mountain_function(X, v, sigma) for v in V]


def select_center(M, V):
    max_pos = np.argmax(M)
    return V[max_pos], M[max_pos]


def update_mountain(V, M, mc, c, sigma, beta):
    for i, _ in enumerate(M):
        M[i] -= mc * density_function(V[i], c, beta)
    return M


def calculate_dist_matrix(X, C):
    return np.array([[distance(x, c) for c in C] for x in X])


def calculate_membership(D):
    return np.argmin(D, axis=1)


def mountain(X, num_c=2, num_div=1, sigma=0.1, beta=0.1):
    V = create_grid(X, num_div)
    M = calculate_mountain(X, V, sigma)
    C = []
    for i in range(num_c):
        c, mc = select_center(M, V)
        M = update_mountain(V, M, mc, c, sigma, beta)
        C.append(c)
    return np.array(C)
