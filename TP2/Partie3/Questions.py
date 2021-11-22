import numpy as np
import matplotlib.pyplot as plt

from Programmes_Annexes.set_params import *
from Programmes_Annexes.rand_matrix_gen import *

from TP2.Partie3.main_functions_p3 import *

all_params = [[120, 130, 1, 2],
              [127, 127, 1, 5],
              [127, 128, 1, 1],
              [127, 128, 0.1, 0.1],
              [127, 128, 2, 3],
              ]

all_probas = [
    [0.18, 0.82],
    [0.34, 0.66],
    [0.47, 0.53],
    [0.66, 0.44],
]


def question3(n):

    for i in range(len(all_probas)):
        params = setParams_without_X(all_params[0], 100, 200)
        params['a'] = all_probas[i]

        for j in range(3):
            params['A'] = rand_matrix_gen()

            fig = plt.figure(figsize=(13, 7))
            fig.suptitle("Réalisation d'une chaine de Markov cachée de longueur " + str(n) + ' \n  A :' + str(params['A']) + ', Probas :' + str(params['a']), fontsize=18)

            plt.plot(genere_chaine2(n, params), color='blue', label='Chaîne de Markov')
            plt.plot(simul2(n, params), color='orange', label='Empirique')

            plt.legend()
            plt.tight_layout()
            fig.show()
            fig.savefig('FiguresP3/Signal ' + str((i, j)) + '.png')


def question4():
    params = setParams_without_X(all_params[0], 100, 200)
    params['A'] = rand_matrix_gen()
    params['a'] = 0.5

    X = genere_chaine2(20, params)
    Y = bruit_gaussien(X, params)

    fig = plt.figure(figsize=(13, 7))
    fig.suptitle("Génération de signal en considérant une chaîne de Markov", fontsize=18)
    plt.plot(X, color='orange', label='Version Non Bruitée')
    plt.plot(Y, color='red', label='Version Bruitée')

    plt.legend()
    fig.show()

#question3()
#question4()

