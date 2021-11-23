import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

from TP1.Partie1.main_functions_p1 import *


def erreur_moyenne_map(n, X, params):
    error = 0
    for i in range(n):
        error += taux_erreur(X, MAP_MPM(bruit_gaussien(X, params), params))
    return error/n


def erreur_moyenne_mpm(n, X, params):
    error = 0
    for i in range(n):
        error += taux_erreur(MAP_MPM_Markov(bruit_gaussien(X, params), calc_probas_prio(len(X), params['A'], params['a']), params), X)
    return error/n


def calc_probaprio2(X):
    return [np.unique(X, return_counts=True)[1][0]/len(X), np.unique(X, return_counts=True)[1][1]/len(X)]


def calc_transit_prio2(X):
    if len(np.unique(X, return_counts=True)[1]) == 1:
        return np.where(100 in X, np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]))

    cl1, cl2, count1, count2 = np.unique(X)[0], np.unique(X)[1], np.unique(X, return_counts=True)[1][0], np.unique(X, return_counts=True)[1][1]
    a, b = np.sum((X[:-1] == cl1) & (X[1:] == cl1))/(count1), np.sum((X[:-1] == cl2) & (X[1:] == cl2))/(count2)
    return np.array([[a, 1-a], [1-b, b]])


def calc_probas_prio(n, A, a): #Fonctions pour calcule les probabilités P(X_n = w_1) et P(X_n = w2) (de manière recursive) où n décrit l'ensemble des points du signal 
    probas = np.zeros((n, 2))
    probas[0] = np.array([[a, 1-a]]) #Probabilité de la racine

    for i in range(n):
        probas[i, 0] = probas[i-1, 0]*A[0][0] + probas[i-1, 1]*A[1][0]
        probas[i, 1] = 1 - probas[i, 0]

    return probas


def MAP_MPM(Y, params): #Loi a priori de X estimée de manière empirique
    return (
            (stats.norm.pdf(Y, params['m1'], params['sig1'])*params['a'] >= stats.norm.pdf(Y, params['m2'], params['sig2'])*(1-params['a']))*params['cl1'] +
            (stats.norm.pdf(Y, params['m1'], params['sig1'])*params['a'] < stats.norm.pdf(Y, params['m2'], params['sig2'])*(1-params['a']))*params['cl2']
            )


def MAP_MPM_Markov(Y, probas, params): #Loi a priori de X estimée à partir des propriétés des chaînes de Markov
    seg_signal = np.zeros((len(Y)))
    seg_signal[:] = np.where(
        (stats.norm.pdf(Y, params['m1'], params['sig1'])[:]*probas[:, 0] >= stats.norm.pdf(Y, params['m2'], params['sig2'])[:]*probas[:, 1]),
        params['cl1'],
        params['cl2']
    )
    return seg_signal


def simul2(n, params): # Question 3
    generated_signal = []
    for i in range(n):
        if np.random.rand() >= params['a']:
            generated_signal.append(params['cl1'])
        else:
            generated_signal.append(params['cl2'])
    return np.array(generated_signal)


#Etude supplémentaire, sur la qualité de l'estimateur empirique pour A

def ep_vs_cm_signal():
    result = []
    all_params = [
        [120, 130, 1, 2],
        [127, 127, 1, 5],
        [127, 128, 1, 1],
        [127, 128, 0.1, 0.1],
        [127, 128, 2, 3],
    ]

    for i in range(6):
        if i == 0:
            X = np.load('assets/signaux/signal.npy')
        else:
            X = np.load('assets/signaux/signal' + str(i) + '.npy')

        print("Signal : " + str(i))
        for j in range(len(all_params)):
            print("Paramètre : " + str(j))

            params = setParams(all_params[j], X)
            params['A'] = calc_transit_prio2(X)
            params['a'] = calc_probaprio2(X)[0]

            result.append([erreur_moyenne_mpm(250, X, params),
                           erreur_moyenne_map(250, X, params)])

    return result


#print(ep_vs_cm())














