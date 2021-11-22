import numpy as np
import matplotlib.pyplot as plt

from Programmes_Annexes import set_params
from TP2.Partie2.main_functions_p2 import *


def tirage_classe2(p1, cl1, cl2): # Question 1
    a = np.random.rand()
    return cl1*(a <= p1) + cl2*(a > p1)


def genere_chaine2(n, params): # Question 2
    chain = [tirage_classe2(params['a'], params['cl1'], params['cl2'])]

    for i in range(1, n):
        chain.append(
            (chain[i-1] == params['cl1']) * tirage_classe2(params['A'][0][0], params['cl1'], params['cl2']) +
            (chain[i - 1] == params['cl2']) * tirage_classe2(params['A'][1][0], params['cl1'], params['cl2'])
        )
    return np.array(chain)






