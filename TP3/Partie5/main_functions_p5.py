import numpy as np
import matplotlib.pyplot as plt

from Programmes_Annexes.chain_to_image_functions import *

from TP3.Partie4.main_functions_p4 import *
from Programmes_Annexes.set_params import *


all_params = [[120, 130, 1, 2],
              [127, 127, 1, 5],
              [127, 128, 1, 1],
              [127, 128, 0.1, 0.1],
              [127, 128, 2, 3],
              ]
all_a = [0.77, 0.46, 0.17, 0.8, 0.56]


def bruit_seg_all(img):
    X_ch = image_to_chain(img)
    A = calc_transit_prio2(X_ch)

    noised_images = []
    mv_images = []
    map_images = []
    mpm_images = []

    for i in range(len(all_params)):
        params = setParams(all_params[i], X_ch)
        params['A'] = A
        params['a'] = all_a[i]
        noised_images.append(bruit_gaussien(X_ch, params))
        mat_f = gauss2(noised_images[-1], params)

        mv_images.append(classif_gauss2(noised_images[-1], params))
        map_images.append(MAP_MPM(noised_images[-1], params))
        mpm_images.append(MPM_chaines2_with_rescaling(mat_f, params))

    return noised_images, mv_images, map_images, mpm_images



