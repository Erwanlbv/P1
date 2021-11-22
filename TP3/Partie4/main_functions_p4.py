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


def MPM_chaines2(mat_f, cl1, cl2, A, p10):
    mpm_seg = np.zeros(len(mat_f))
    mpm_seg[:] = np.where(
        forward2(mat_f, A, p10)[:, 0]*backward2(mat_f, A)[:, 0] >= forward2(mat_f, A, p10)[:, 1]*backward2(mat_f, A)[:, 1],
        cl1,
        cl2
    )
    return mpm_seg


def MPM_chaines2_with_rescaling(mat_f, cl1, cl2, A, p10):
    mpm_seg = np.zeros(len(mat_f))
    mpm_seg[:] = np.where(
        forward_with_rescaling(mat_f, A, p10)[:, 0]*backward_with_rescaling(mat_f, A)[:, 0] >= forward_with_rescaling(mat_f, A, p10)[:, 1]*backward_with_rescaling(mat_f, A)[:, 1],
        cl1,
        cl2
    )
    return mpm_seg


def calc_transit_prio2(X):
    if len(np.unique(X, return_counts=True)[1]) == 1:
        return np.where(100 in X, np.array([[1, 0], [0, 0]]), np.array([[0, 0], [0, 1]]))

    cl1, cl2, count1, count2 = np.unique(X)[0], np.unique(X)[1], np.unique(X, return_counts=True)[1][0], np.unique(X, return_counts=True)[1][1]
    a, b = np.sum((X[:-1] == cl1) & (X[1:] == cl1))/(count1), np.sum((X[:-1] == cl2) & (X[1:] == cl2))/(count2)
    return np.array([[a, 1-a], [1-b, b]])


"""" ---- Espace de fonctions d'affichage -----"""



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



"""
#question5()

results_one_sign = question6_one_sign()
resultats_one_sign, new_params_one_sig = results_one_sign[0], results_one_sign[1]
resultats_one_sign = np.round(resultats_one_sign, 2)
"""

#results_avg_sign = np.round(question6_avg_sign(), 3)
#array_to_latex(10, results_avg_sign.flatten())

#array_to_latex(10, np.round(question12(), 2))

"""l = np.array([[0.00048, 0.00032], [0.19816, 0.19919999999999982], [0.28679999999999994, 0.27912000000000015], [0.0, 0.0], [0.35999999999999954, 0.35999999999999954], [0.0003840000000000002, 0.00040000000000000024], [0.1775839999999999, 0.17792000000000005], [0.3062720000000002, 0.310528], [0.0, 0.0], [0.37985600000000025, 0.38353600000000015], [0.00039552, 0.00040583999999999987], [0.17639423999999976, 0.17638735999999997], [0.3085887200000001, 0.30858775999999993], [2.4000000000000003e-07, 4.800000000000001e-07], [0.3806219200000001, 0.38095279999999987], [0.00041696, 0.0004055199999999999], [0.1771876800000001, 0.17677160000000003], [0.3084373599999998, 0.3086020000000001], [2.0320000000000046e-05, 1.6e-07], [0.38173856000000017, 0.38192040000000005], [0.00041840000000000014, 0.00041200000000000015], [0.1768327999999999, 0.17670175999999996], [0.30867096, 0.30857352000000016], [2.0160000000000048e-05, 4.000000000000001e-07], [0.38155799999999973, 0.3816560800000001], [0.0004258399999999998, 0.00040775999999999996], [0.17641264000000006, 0.17623223999999996], [0.3084483999999999, 0.3083908], [2.0320000000000046e-05, 5.6e-07], [0.38086016000000017, 0.38062208]])

array_to_latex(5, np.round(l[:, 0], 3))
array_to_latex(5, np.round(l[:, 1], 3))"""

print("\n".join(comp_A_emp()))





















