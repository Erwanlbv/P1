import numpy as np

from Programmes_Annexes.set_params import *
from TP2.Partie3.main_functions_p3 import *


def comp_A_emp():

    all_A = [
        np.array([[0.05, 0.95], [0.25, 0.75]]),
        np.array([[0.34, 0.66], [0.48, 0.52]]),
        np.array([[0.64, 0.36], [0.17, 0.83]]),
        np.array([[0.88, 0.12], [0.23, 0.77]]),
        np.array([[0.15, 0.85], [0.13, 0.87]]),
        ]
    result = []

    params = setParams_without_X([120, 130, 1, 2], 100, 200)
    params['a'] = 0.6

    for A in all_A:
        L = []
        params['A'] = A
        print("A : " + str(A))
        avg_n_error = np.zeros((1, 3, 4))

        for i in range(200):
            if i % 50 == 0:
                print("Génération :" + str(i))
            full_chain = genere_chaine2(5000, params)

            for j in range(len([50, 250, 1000])):
                markov_chain = full_chain[:[50, 250, 5000][j]]
                emp_A = calc_transit_prio2(markov_chain)
                avg_n_error[0, j] += np.abs((emp_A-A).flatten())

        L.append(np.round((avg_n_error/200), 2).tolist())
        result.append(str(L))
    return result


print("\n".join(comp_A_emp()))