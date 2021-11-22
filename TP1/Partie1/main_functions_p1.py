import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats


from Programmes_Annexes.set_params import * 
#SetParams permet de regrouper tous les paramètres utiles (cl1? cl2, m1, m2, sig1, sig2) dans une même variable pour éviter d'avoir des défintions de fonctions à rallonge


def bruit_gaussien(X, params):
    return ((X == params['cl1'])*np.random.normal(params['m1'], params['sig1'], len(X)) +
            (X == params['cl2'])*np.random.normal(params['m2'], params['sig2'], len(X))
            )


def classif_gauss2(Y, params):
    return (
        (stats.norm.pdf(Y, params['m1'], params['sig1']) >= stats.norm.pdf(Y, params['m2'], params['sig2']))*params['cl1'] +
        (stats.norm.pdf(Y, params['m1'], params['sig1']) < stats.norm.pdf(Y, params['m2'], params['sig2']))*params['cl2']
    )


def taux_erreur(A, B):
    return np.mean(A != B)


def erreur_moyenne_classif_gauss(T, X, params):
    error = 0
    for i in range(T):
        error += taux_erreur(X, classif_gauss2(bruit_gaussien(X, params), params))
    return error/T


def getResult(T, X, params):
    result = []
    for i in range(len(T)):
        result.append(erreur_moyenne_classif_gauss(int(T[i]), X, params))
    return result

