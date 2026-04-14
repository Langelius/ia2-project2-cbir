from skimage.feature import graycomatrix, graycoprops
from mahotas.features import haralick
from BiT import bio_taxo

import cv2
import numpy as np

PROPRIETES_GLCM = ['contrast', 'homogeneity', 'energy', 'dissimilarity', 'correlation', 'ASM']


def glcm_RGB(chemin):
    """Extrait 18 caractéristiques GLCM (6 par canal RGB)."""
    image = cv2.imread(chemin)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    caracteristiques = []
    for i in range(3):
        canal = image[:, :, i]
        matrice_cooccurrence = graycomatrix(canal, [1], [np.pi / 2], symmetric=False, normed=False)
        caracteristiques.extend([float(graycoprops(matrice_cooccurrence, p)[0, 0]) for p in PROPRIETES_GLCM])
    return caracteristiques  # 18 valeurs


def haralick_feat_RGB(chemin):
    """Extrait 39 caractéristiques Haralick (13 par canal RGB)."""
    image = cv2.imread(chemin)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    caracteristiques = []
    for i in range(3):
        canal = image[:, :, i]
        caracteristiques.extend([float(x) for x in haralick(canal).mean(0).tolist()])
    return caracteristiques  # 39 valeurs


def bitdesc_feat_RGB(chemin):
    """Extrait 42 caractéristiques BiT (14 par canal RGB)."""
    image = cv2.imread(chemin)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    caracteristiques = []
    for i in range(3):
        canal = image[:, :, i]
        caracteristiques.extend([float(x) for x in bio_taxo(canal)])
    return caracteristiques  # 42 valeurs


def concat_RGB(chemin):
    """Concatène GLCM + Haralick + BiT → 99 caractéristiques."""
    caracteristiques = []
    caracteristiques.extend(glcm_RGB(chemin))
    caracteristiques.extend(haralick_feat_RGB(chemin))
    caracteristiques.extend(bitdesc_feat_RGB(chemin))
    return caracteristiques  # 99 valeurs


NB_BINS_HIST = 32  # 32 bins par canal → 96 valeurs au total


def histogramme_RGB(chemin):
    """Extrait 96 caractéristiques d'histogramme de couleur (32 bins par canal RGB)."""
    image = cv2.imread(chemin)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    caracteristiques = []
    for i in range(3):
        canal = image[:, :, i]
        hist = cv2.calcHist([canal], [0], None, [NB_BINS_HIST], [0, 256])
        # Normalisation pour être invariant à la taille de l'image
        hist = cv2.normalize(hist, hist).flatten()
        caracteristiques.extend([float(x) for x in hist])
    return caracteristiques  # 96 valeurs


def glcm_hist_RGB(chemin):
    """Concatène GLCM + Histogramme → 114 caractéristiques."""
    caracteristiques = []
    caracteristiques.extend(glcm_RGB(chemin))
    caracteristiques.extend(histogramme_RGB(chemin))
    return caracteristiques  # 114 valeurs


def haralick_hist_RGB(chemin):
    """Concatène Haralick + Histogramme → 135 caractéristiques."""
    caracteristiques = []
    caracteristiques.extend(haralick_feat_RGB(chemin))
    caracteristiques.extend(histogramme_RGB(chemin))
    return caracteristiques  # 135 valeurs


def bitdesc_hist_RGB(chemin):
    """Concatène BiT + Histogramme → 138 caractéristiques."""
    caracteristiques = []
    caracteristiques.extend(bitdesc_feat_RGB(chemin))
    caracteristiques.extend(histogramme_RGB(chemin))
    return caracteristiques  # 138 valeurs


def concat_hist_RGB(chemin):
    """Concatène GLCM + Haralick + BiT + Histogramme → 195 caractéristiques."""
    caracteristiques = []
    caracteristiques.extend(glcm_RGB(chemin))
    caracteristiques.extend(haralick_feat_RGB(chemin))
    caracteristiques.extend(bitdesc_feat_RGB(chemin))
    caracteristiques.extend(histogramme_RGB(chemin))
    return caracteristiques  # 195 valeurs
