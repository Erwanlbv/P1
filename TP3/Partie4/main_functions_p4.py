import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

from TP2.Partie3.main_functions_p3 import *


def gauss2(Y, params):
    mat_f = np.zeros((len(Y), 2))

    mat_f[:, 0] = stats.norm.pdf(Y, params['m1'], params['sig1'])
    mat_f[:, 1] = stats.norm.pdf(Y, params['m2'], params['sig2'])
    return mat_f


def forward2(mat_f, A, p10):
    alpha = np.zeros((len(mat_f), 2))
    alpha[0] = mat_f[0]*[p10, 1-p10]

    for i in range(1, len(mat_f)):
        alpha[i][0] = alpha[i-1][0]*mat_f[i, 0]*A[0][0] + alpha[i-1][1]*mat_f[i, 0]*A[1][0]
        alpha[i][1] = alpha[i-1][0]*mat_f[i, 1]*A[0][1] + alpha[i-1][1]*mat_f[i, 1]*A[1][1]
    return alpha


def forward_with_rescaling(mat_f, A, p10):
    alpha = np.zeros((len(mat_f), 2))
    alpha_star = np.zeros((len(mat_f), 2))
    alpha[0] = mat_f[0]*[p10, 1-p10]
    alpha_star[0] = alpha[0]/np.sum(alpha[0])

    for i in range(1, len(mat_f)):
        alpha[i][0] = alpha_star[i-1][0]*mat_f[i, 0]*A[0][0] + alpha_star[i-1][1]*mat_f[i, 0]*A[1][0]
        alpha[i][1] = alpha_star[i-1][0]*mat_f[i, 1]*A[0][1] + alpha_star[i-1][1]*mat_f[i, 1]*A[1][1]

        alpha_star[i] = alpha[i]/np.sum(alpha[i])

    return alpha_star


def backward_with_rescaling(mat_f, A):
    beta = np.zeros((len(mat_f), 2))
    beta_star = np.zeros((len(mat_f), 2))
    beta[-1][0], beta[-1][1] = 1, 1
    beta_star[-1] = beta[-1]/np.sum(beta[-1:])

    for i in range(len(mat_f)-1):
        beta[-2-i][0] = beta_star[-i-1, 0]*mat_f[-i-1, 0]*A[0][0] + beta_star[-i-1, 1]*mat_f[-i-1, 1]*A[0][1]
        beta[-2-i][1] = beta_star[-i-1, 0]*mat_f[-i-1, 0]*A[1][0] + beta_star[-i-1, 1]*mat_f[-i-1, 1]*A[1][1]

        beta_star[-2-i] = beta[-2-i]/np.sum(beta[-2-i])

    return beta_star


def backward2(mat_f, A):
    beta = np.zeros((len(mat_f), 2))
    beta[-1] = np.array([[1, 1]])

    for i in range(len(mat_f)-1):
        beta[-2-i][0] = beta[-i-1, 0]*mat_f[-i-1, 0]*A[0][0] + beta[-i-1, 1]*mat_f[-i-1, 1]*A[0][1]
        beta[-2-i][1] = beta[-i-1, 0]*mat_f[-i-1, 0]*A[1][0] + beta[-i-1, 1]*mat_f[-i-1, 1]*A[1][1]

    return beta


def MPM_chaines2(mat_f, params):
    mpm_seg = np.zeros(len(mat_f))
    mpm_seg[:] = np.where(
        forward2(mat_f, params['A'], params['a'])[:, 0]*backward2(mat_f, params['A'])[:, 0] >=
        forward2(mat_f, params['A'], params['a'])[:, 1]*backward2(mat_f, params['A'])[:, 1],
        params['cl1'],
        params['cl2']
    )
    return mpm_seg


def MPM_chaines2_with_rescaling(mat_f, params):
    mpm_seg = np.zeros(len(mat_f))
    mpm_seg[:] = np.where(
        forward_with_rescaling(mat_f, params['A'], params['a'])[:, 0]*backward_with_rescaling(mat_f, params['A'])[:, 0] >=
        forward_with_rescaling(mat_f, params['A'], params['a'])[:, 1]*backward_with_rescaling(mat_f,  params['A'])[:, 1],
        params['cl1'],
        params['cl2']
    )
    return mpm_seg
























