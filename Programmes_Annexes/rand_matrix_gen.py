import numpy as np


def rand_matrix_gen(): # Pour générer une matrice A de manière aléatoire
    (a, b) = (np.random.rand(), np.random.rand())
    return np.array([[a, 1-a], [b, 1-b]])