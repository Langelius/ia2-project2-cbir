from skimage.feature import graycomatrix, graycoprops
from mahotas.features import haralick
from BiT import bio_taxo

import cv2
import numpy as np

proprietes_glcm = ['contrast', 'homogeneity', 'energy', 'dissimilarity', 'correlation', 'ASM']


def charger_image(chemin):
    image = cv2.imread(chemin)
    if image is None:
        raise ValueError(f"Impossible de lire l'image : {chemin}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def glcm_RGB(image_rgb):
    caracteristiques = []
    for i in range(3):
        canal = image_rgb[:, :, i]
        matrice_cooccurrence = graycomatrix(canal, [1], [np.pi / 2], symmetric=False, normed=False)
        caracteristiques.extend([float(graycoprops(matrice_cooccurrence, p)[0, 0]) for p in proprietes_glcm])
    return caracteristiques


def haralick_feat_RGB(image_rgb):
    caracteristiques = []
    for i in range(3):
        canal = image_rgb[:, :, i]
        caracteristiques.extend([float(x) for x in haralick(canal).mean(0).tolist()])
    return caracteristiques


def bitdesc_feat_RGB(image_rgb):
    caracteristiques = []
    for i in range(3):
        canal = image_rgb[:, :, i]
        valeurs = np.nan_to_num(
            np.array([float(x) for x in bio_taxo(canal)]),
            nan=0.0, posinf=0.0, neginf=0.0
        )
        caracteristiques.extend(valeurs.tolist())
    return caracteristiques


def concat_RGB(image_rgb):
    caracteristiques = []
    caracteristiques.extend(glcm_RGB(image_rgb))
    caracteristiques.extend(haralick_feat_RGB(image_rgb))
    caracteristiques.extend(bitdesc_feat_RGB(image_rgb))
    return caracteristiques
