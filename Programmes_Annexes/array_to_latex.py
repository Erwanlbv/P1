import numpy as np

def array_to_latex(n_col, tab):
    L = []
    L1 = []
    for i in range(len(tab)):
        if (i % n_col == 0) & (i != 0):
            L.append(L1)
            L1 = []
        L1.append(str(tab[i]))
    L.append(L1)
    print("Len L :" + str(len(L)))
    for l in L:
        print('Ligne ' + str(L.index(l) + 1) + ' & ' + ' & '.join(l) + "\\")