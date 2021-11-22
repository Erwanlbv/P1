import numpy as np
import matplotlib.pyplot as plt

from Programmes_Annexes.rand_matrix_gen import *
from TP3.Partie4.main_functions_p4 import *

all_params = [[120, 130, 1, 2],
              [127, 127, 1, 5],
              [127, 128, 1, 1],
              [127, 128, 0.1, 0.1],
              [127, 128, 2, 3],
              ]


def question5():
    params = setParams_without_X(all_params[0], 100, 200) #Numéro du bruit
    params['A'] = rand_matrix_gen()
    params['a'] = np.random.rand()
    markov_chain = genere_chaine2(50, params)
    noised_markov_chain = bruit_gaussien(markov_chain, params)

    mpm_seg = MPM_chaines2(gauss2(noised_markov_chain, params), 100, 200, params['A'], params['a'])

    fig, axs = plt.subplots(2, 1, figsize=(16, 9))
    fig.suptitle("Segmentation MPM2 \n Taux d'erreur : " + str(taux_erreur(markov_chain, mpm_seg)))
    axs[0].plot(markov_chain, label='Signal Généré')
    axs[1].plot(noised_markov_chain, "b:o", label='Signal Bruité', color='orange')
    axs[1].plot(mpm_seg, "r--", label='Signal Segmenté', color='red')

    for ax in axs.flat:
        ax.label_outer()
        ax.legend()

    fig.show()


def question6_avg_sign():
    results = []
    all_A = [
        np.array([[0.05, 0.95], [0.25, 0.75]]),
        np.array([[0.34, 0.66], [0.48, 0.52]]),
        np.array([[0.64, 0.36], [0.17, 0.83]]),
        np.array([[0.88, 0.12], [0.23, 0.77]]),
        np.array([[0.15, 0.85], [0.13, 0.87]]),
    ]

    all_a = [0.77, 0.46, 0.17, 0.8, 0.56]

    for i in range(5):
        print('signal' + str(i))

        for j in range(len(all_params)):
            error_map, error_mpm = 0, 0
            params = setParams_without_X(all_params[j], 100, 200)
            params['A'], params['a'] = all_A[i], all_a[i]

            for k in range(100):
                markov_chain = genere_chaine2(20, params)
                error_map += erreur_moyenne_map(200, markov_chain, params)
                error_mpm += erreur_moyenne_mpm(200, markov_chain, params)

            results.append([error_map / 100, error_mpm / 100])

    return results


def question8():  # Signaux non générés par genere_chaine2

    results = []
    for i in range(4):
        # Cette fois on part d'un signal déjà exitant dont on ne connait pas les priopriétés
        if i == 0:
            markov_chain = np.load('../../assets/signaux/signal.npy')
        else:
            markov_chain = np.load('assets/signaux/signal' + str(i) + '.npy')

        for j in range(len(all_params)):
            params = setParams(all_params[j], markov_chain)
            params['A'] = calc_transit_prio2(markov_chain)  # On estime A à partir du signal donné
            params['a'] = 0.5  # Probabilité de la racine.

            results.append(
                [erreur_moyenne_map(500, markov_chain, params), erreur_moyenne_mpm(500, markov_chain, params)])
            # On fait la moyenne sur 500 segmentations du taux d'erreur pour les 2 approches
    return results


def question12():
    results = []
    for i in range(6):
        if i == 0:
            markov_chain = np.load('../../assets/signaux/signal.npy')
        else:
            markov_chain = np.load('./assets/signaux/signal' + str(i) + '.npy')
        print('Signal' + str(i) + ' Taille : ' + str(len(markov_chain)))
        for j in range(len(all_params)):
            params = setParams(all_params[j], markov_chain)
            params['A'], params['a'] = calc_transit_prio2(markov_chain), np.random.rand()

            results.append(
                [erreur_moyenne_map(200, markov_chain, params), erreur_moyenne_mpm(200, markov_chain, params)])

    return results

