import os
import numpy as np
import joblib

from descripteurs import glcm_RGB, haralick_feat_RGB, bitdesc_feat_RGB, concat_RGB, charger_image

DOSSIER_SIGNATURES = 'signatures'
DOSSIER_MODELES = 'models'
DOSSIER_DATASET = 'dataset'

FONCTIONS_DESCRIPTEURS = {
    'glcm':     glcm_RGB,
    'haralick': haralick_feat_RGB,
    'bitdesc':  bitdesc_feat_RGB,
    'concat':   concat_RGB,
}


# ─────────────────────────────────────────────
#  Mesures de distance
# ─────────────────────────────────────────────

def distance_euclidienne(requete, signatures):
    """requete: (d,)  signatures: (n, d)  →  (n,)"""
    return np.sqrt(np.sum((signatures - requete) ** 2, axis=1))


def distance_canberra(requete, signatures):
    """requete: (d,)  signatures: (n, d)  →  (n,)"""
    numerateur = np.abs(signatures - requete)
    denominateur = np.abs(signatures) + np.abs(requete)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(denominateur != 0, numerateur / denominateur, 0.0)
    return np.sum(ratio, axis=1)


def distance_cosinus(requete, signatures):
    """requete: (d,)  signatures: (n, d)  →  (n,)"""
    normes = np.linalg.norm(signatures, axis=1) * np.linalg.norm(requete)
    with np.errstate(invalid='ignore', divide='ignore'):
        similarites = np.where(normes != 0, signatures @ requete / normes, 0.0)
    return 1.0 - similarites


MESURES_DISTANCE = {
    'euclidienne': distance_euclidienne,
    'canberra':    distance_canberra,
    'cosinus':     distance_cosinus,
}


# ─────────────────────────────────────────────
#  Chargement des ressources
# ─────────────────────────────────────────────

_cache = {}


def charger_signatures(nom_descripteur):
    key = f'signatures_{nom_descripteur}'
    if key not in _cache:
        chemin = os.path.join(DOSSIER_SIGNATURES, f'signatures_{nom_descripteur}.npy')
        tableau = np.load(chemin)
        caracteristiques = tableau[:, :-1].astype('float')
        etiquettes = tableau[:, -1].astype('int')
        _cache[key] = (caracteristiques, etiquettes)
    return _cache[key]


def charger_chemins_images(dossier_dataset, dict_classes):
    key = f'chemins_{dossier_dataset}'
    if key not in _cache:
        chemins = []
        etiquettes = []
        for racine, _, fichiers in os.walk(dossier_dataset):
            for fichier in sorted(fichiers):
                if not fichier.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                nom_classe = os.path.basename(racine)
                if nom_classe not in dict_classes:
                    continue
                chemins.append(os.path.join(racine, fichier))
                etiquettes.append(dict_classes[nom_classe])
        _cache[key] = (chemins, np.array(etiquettes))
    return _cache[key]


def charger_modele(nom_descripteur):
    key = f'modele_{nom_descripteur}'
    if key not in _cache:
        chemin = os.path.join(DOSSIER_MODELES, f'meilleur_modele_{nom_descripteur}.joblib')
        _cache[key] = joblib.load(chemin)
    return _cache[key]


def charger_dict_classes():
    if 'dict_classes' not in _cache:
        chemin = os.path.join(DOSSIER_SIGNATURES, 'class_mapping.npy')
        _cache['dict_classes'] = np.load(chemin, allow_pickle=True).item()
    return _cache['dict_classes']


# ─────────────────────────────────────────────
#  Moteur CBIR
# ─────────────────────────────────────────────

def rechercher(chemin_requete, nom_descripteur, nom_distance, nb_resultats=10):
    """
    Recherche les images les plus similaires à l'image requête.

    Paramètres
    ----------
    chemin_requete   : chemin vers l'image à rechercher
    nom_descripteur  : 'glcm', 'haralick', 'bitdesc' ou 'concat'
    nom_distance     : 'euclidienne', 'canberra' ou 'cosinus'
    nb_resultats     : nombre d'images similaires à retourner

    Retourne
    --------
    classe_predite   : nom de la classe prédite pour l'image requête
    resultats        : liste de (chemin_image, distance) triée par distance croissante
    """
    dict_classes = charger_dict_classes()
    index_vers_classe = {indice: nom for nom, indice in dict_classes.items()}

    # Extraction des caractéristiques de l'image requête
    fonction_descripteur = FONCTIONS_DESCRIPTEURS[nom_descripteur]
    image_rgb = charger_image(chemin_requete)
    caracteristiques_requete = np.array(fonction_descripteur(image_rgb))

    # Prédiction de la classe
    modele = charger_modele(nom_descripteur)
    indice_classe_predite = modele.predict([caracteristiques_requete])[0]
    classe_predite = index_vers_classe[indice_classe_predite]

    # Chargement des signatures et filtrage par classe prédite
    signatures, etiquettes_dataset = charger_signatures(nom_descripteur)
    chemins_images, etiquettes_images = charger_chemins_images(DOSSIER_DATASET, dict_classes)

    # Version longue :
    # signatures_classe = []
    # for i, etiquette in enumerate(etiquettes_images):
    #     if etiquette == indice_classe_predite:
    #         signatures_classe.append(signatures[i])
    # signatures_classe = np.array(signatures_classe)
    masque_classe = etiquettes_images == indice_classe_predite # masque booléen pour filtrer les images de la classe prédite
    signatures_classe = signatures[masque_classe]

    chemins_classe = [chemin for chemin, appartient_classe in zip(chemins_images, masque_classe) if appartient_classe]

    # Calcul des distances (vectorisé sur toute la classe)
    fonction_distance = MESURES_DISTANCE[nom_distance]
    distances = fonction_distance(caracteristiques_requete, signatures_classe)

    # Tri par distance croissante
    indices_tries = np.argsort(distances)
    resultats = [
        (chemins_classe[i], distances[i])
        for i in indices_tries[:nb_resultats]
    ]

    return classe_predite, resultats
