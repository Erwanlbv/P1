import numpy as np


def setParams(L, X):
    params = {'cl1': '', 'cl2': '', 'm1': '', 'sig1': '', 'm2': '', 'sig2': '', 'A': '', 'a': ''}
    params['cl1'] = np.unique(X)[0]
    params['cl2'] = np.unique(X)[1]
    params['m1'] = L[0]
    params['sig1'] = L[2]
    params['m2'] = L[1]
    params['sig2'] = L[3]
    return params

def setParams_without_X(L, cl1, cl2):
    params = {'cl1': '', 'cl2': '', 'm1': '', 'sig1': '', 'm2': '', 'sig2': '', 'A': '', 'a': ''}
    params['cl1'] = cl1
    params['cl2'] = cl2
    params['m1'] = L[0]
    params['sig1'] = L[2]
    params['m2'] = L[1]
    params['sig2'] = L[3]
    return params

