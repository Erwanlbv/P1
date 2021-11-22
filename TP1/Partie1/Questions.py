import matplotlib.pyplot as plt
import numpy as np

from main_functions_p1 import *

all_params = [[120, 130, 1, 2],
              [127, 127, 1, 5],
              [127, 128, 1, 1],
              [127, 128, 0.1, 0.1],
              [127, 128, 2, 3],
              ]

X = np.load("../../assets/signaux/signal.npy")
params = setParams(all_params[1], X)
Y = classif_gauss2(bruit_gaussien(X, params), params)

print('A')
fig, axs = plt.subplots(2)
fig.suptitle('Résultats')

axs[0].set_title('Initial Signal')
axs[0].plot(X, color='blue')

axs[1].set_title('Processed Signal')
axs[1].plot(Y, color='orange')

for ax in axs.flat: #Pour éviter que les titres et les abscisses se chevauchent
    ax.label_outer()

T = np.linspace(1, 500, 500)
result = getResult(T, X, params)
plt.plot(T, result, color='blue')
plt.show()
