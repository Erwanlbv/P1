import numpy as np
import scipy.integrate as integrate
import scipy.stats as stats
from numpy import trapz
import matplotlib.pyplot as plt
import matplotlib.image as image


def setParams(L):
    params = {'cl1': '', 'cl2': '', 'm1': '', 'sig1': '', 'm2': '', 'sig2': ''}
    params['cl1'] = 200
    params['cl2'] = 100
    params['m1'] = L[0]
    params['sig1'] = L[2]
    params['m2'] = L[1]
    params['sig2'] = L[3]
    return params


all_params = [[120, 130, 1, 2],
              [127, 127, 1, 5],
              [127, 128, 1, 1],
              [127, 128, 0.1, 0.1],
              [127, 128, 2, 3],
              ]

error_results = []

"""for params in all_params:

    params = setParams(params)

    T = np.linspace(100, 160, 1000)
    figure = plt.figure(figsize=(13, 7))
    plt.plot(T, stats.norm.pdf(T, params['m1'], params['sig1']))
    plt.plot(T, stats.norm.pdf(T, params['m2'], params['sig2']))
    plt.plot(T, np.minimum(stats.norm.pdf(T, params['m1'], params['sig1']), stats.norm.pdf(T, params['m2'], params['sig2'])))
    plt.tight_layout()
    plt.show()

    error_results.append(trapz(np.minimum(stats.norm.pdf(T, params['m1'], params['sig1']), stats.norm.pdf(T, params['m2'], params['sig2'])), dx=0.02))

print(error_results)"""

img = image.imread("./images_binaires/promenade2.bmp")
print(img.shape)
plt.imshow(img, cmap='gray')
plt.show()







