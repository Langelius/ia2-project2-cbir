from skimage.feature import graycomatrix, graycoprops
from mahotas.features import haralick
from BiT import bio_taxo

import cv2
import numpy as np

PROPRIETES_GLCM = ['contrast', 'homogeneity', 'energy', 'dissimilarity', 'correlation', 'ASM']


def charger_image(chemin):
    """Charge une image depuis le disque et la convertit en RGB."""
    image = cv2.imread(chemin)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def glcm_RGB(image_rgb):
    """Extrait 18 caractéristiques GLCM (6 par canal RGB)."""
    caracteristiques = []
    for i in range(3):
        canal = image_rgb[:, :, i]
        matrice_cooccurrence = graycomatrix(canal, [1], [np.pi / 2], symmetric=False, normed=False)
        caracteristiques.extend([float(graycoprops(matrice_cooccurrence, p)[0, 0]) for p in PROPRIETES_GLCM])
    return caracteristiques  # 18 valeurs


def haralick_feat_RGB(image_rgb):
    """Extrait 39 caractéristiques Haralick (13 par canal RGB)."""
    caracteristiques = []
    for i in range(3):
        canal = image_rgb[:, :, i]
        caracteristiques.extend([float(x) for x in haralick(canal).mean(0).tolist()])
    return caracteristiques  # 39 valeurs


def bitdesc_feat_RGB(image_rgb):
    """Extrait 42 caractéristiques BiT (14 par canal RGB)."""
    caracteristiques = []
    for i in range(3):
        canal = image_rgb[:, :, i]
        caracteristiques.extend([float(x) for x in bio_taxo(canal)])
    return caracteristiques  # 42 valeurs


def concat_RGB(image_rgb):
    """Concatène GLCM + Haralick + BiT → 99 caractéristiques."""
    caracteristiques = []
    caracteristiques.extend(glcm_RGB(image_rgb))
    caracteristiques.extend(haralick_feat_RGB(image_rgb))
    caracteristiques.extend(bitdesc_feat_RGB(image_rgb))
    return caracteristiques  # 99 valeurs
