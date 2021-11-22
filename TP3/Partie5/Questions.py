import matplotlib.pyplot as plt
import numpy as np

from Programmes_Annexes.set_params import *
from TP3.Partie5.main_functions_p5 import *

names = ['beee2', 'cible2','promenade2', 'zebre2']
path = '../../assets/images_binaires/'


def display(name):
    img = plt.imread(path + name + '.bmp', )
    mv_images, map_images, mpm_images = bruit_seg_all(img)
    fig, axs = plt.subplots(5, 3, figsize=(18, 11))

    fig.suptitle("Segmentation des images selon les trois méthodes : MV, MAP, MPM_CC", fontsize=18)
    for i in range(len(all_params)):
        axs[i, 0].set_title("Taux d'erreur MV: " + str(taux_erreur(mv_images[i], image_to_chain(img))), fontsize=16)
        axs[i, 1].set_title("Taux d'erreur MAP: " + str(taux_erreur(map_images[i], image_to_chain(img))), fontsize=16)
        axs[i, 2].set_title("Taux d'erreur MPM_CC: " + str(taux_erreur(mpm_images[i], image_to_chain(img))), fontsize=16)

        axs[i, 0].imshow(chain_to_image(mv_images[i]), cmap='gray')
        axs[i, 1].imshow(chain_to_image(map_images[i]), cmap='gray')
        axs[i, 2].imshow(chain_to_image(mpm_images[i]), cmap='gray')

    plt.tight_layout()
    plt.show()


def test(img): #Fonction pour tester la fonctionnalité du programme sur une image
    X_ch = image_to_chain(img)
    params = setParams(all_params[0], X_ch)
    params['a'] = np.random.rand()
    params['A'] = calc_transit_prio2(X_ch)
    noised_img = bruit_gaussien(X_ch, params)

    mv_img = chain_to_image(classif_gauss2(noised_img, params))
    map_img = chain_to_image(MAP_MPM(noised_img, params))

    mat_f = gauss2(noised_img, params)

    mpm_img = chain_to_image(MPM_chaines2_with_rescaling(mat_f, params))

    fig, axs = plt.subplots(1, 4, figsize=(16, 9), sharex=True, sharey=True)
    fig.suptitle("Test")

    axs[0].imshow(chain_to_image(X_ch), cmap='gray')
    axs[1].imshow(mv_img, cmap='gray')
    axs[2].imshow(map_img, cmap='gray')
    axs[3].imshow(mpm_img, cmap='gray')

    plt.tight_layout()
    plt.show()


for name in names:
    display(name)