import numpy as np
import matplotlib.pyplot as plt

from main_functions_p2 import *


all_probas = [
    [0.18, 0.82],
    [0.34, 0.66],
    [0.47, 0.53],
    [0.66, 0.44],
    [0.87, 0.13]
]

all_params = [[120, 130, 1, 2],
              [127, 127, 1, 5],
              [127, 128, 1, 1],
              [127, 128, 0.1, 0.1],
              [127, 128, 2, 3],
              ]


def question4():

    results_class_gauss = []
    results_map_mpm = []

    for i in range(5):

        print('Nouveau signal')
        X = simul2(50, all_probas[i][0])

        for j in range(len(all_params)):

            print(j)
            params = setParams(all_params[j], X)
            params['a'] = calc_probaprio2(X)[0]

            avg_error_class_gauss = erreur_moyenne_classif_gauss(500, X, params)
            avg_error_map_mpm = erreur_moyenne_map(500, X, params)

            results_class_gauss.append(avg_error_class_gauss)
            results_map_mpm.append(avg_error_map_mpm)

    return [results_map_mpm, results_class_gauss]


def question2():

    for i in range(6):
        if i == 0:
            X = np.load('../../assets/signaux/signal.npy')
        else:
            X = np.load('assets/signaux/signal' + str(i) + '.npy')

        for j in range(len(all_params)):

            params = setParams(all_params[j], X)
            params['a'] = calc_probaprio2(X)[0]

            seg_map = MAP_MPM(bruit_gaussien(X, params), params)

            fig = plt.figure(figsize=(16, 9))
            fig.suptitle(
                'Bruit : m1= ' + str(params['m1']) + ', m2= ' + str(params['m2']) + ", sig1= " + str(params['sig1']) + ', sig2= ' + str(params['sig2']),
                fontsize=18,
            )

            plt.plot(X, label='Signal Initial', color='blue')
            plt.plot(bruit_gaussien(X, params), 'b:o', label='Bruit Gaussien', color='orange')
            plt.plot(seg_map, "b:o", label='Signal Segmenté MAP', color='red')

            plt.tight_layout()
            plt.legend()
            plt.show()

            fig.savefig('./Figuresp2q2/Signal' + str(i) +', Bruit : ' + str(j) + '.png')

