import os
import numpy as np
import joblib

from descripteurs import glcm_RGB, haralick_feat_RGB, bitdesc_feat_RGB, concat_RGB

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

def distance_euclidienne(vecteur_a, vecteur_b):
    return np.sqrt(np.sum((vecteur_a - vecteur_b) ** 2))


def distance_canberra(vecteur_a, vecteur_b):
    numerateur = np.abs(vecteur_a - vecteur_b)
    denominateur = np.abs(vecteur_a) + np.abs(vecteur_b)
    # Évite la division par zéro quand les deux valeurs sont nulles
    masque_non_nul = denominateur != 0
    return np.sum(numerateur[masque_non_nul] / denominateur[masque_non_nul])


def distance_cosinus(vecteur_a, vecteur_b):
    norme_a = np.linalg.norm(vecteur_a)
    norme_b = np.linalg.norm(vecteur_b)
    if norme_a == 0 or norme_b == 0:
        return 1.0  # distance maximale si un vecteur est nul
    similarite = np.dot(vecteur_a, vecteur_b) / (norme_a * norme_b)
    return 1.0 - similarite  # conversion similarité → distance


MESURES_DISTANCE = {
    'euclidienne': distance_euclidienne,
    'canberra':    distance_canberra,
    'cosinus':     distance_cosinus,
}


# ─────────────────────────────────────────────
#  Chargement des ressources
# ─────────────────────────────────────────────

def charger_signatures(nom_descripteur):
    chemin = os.path.join(DOSSIER_SIGNATURES, f'signatures_{nom_descripteur}.npy')
    tableau = np.load(chemin)
    caracteristiques = tableau[:, :-1].astype('float')
    # Nettoyage des valeurs invalides produites par bio_taxo (division par zéro dans le log)
    caracteristiques = np.nan_to_num(caracteristiques, nan=0.0, posinf=0.0, neginf=0.0)
    etiquettes = tableau[:, -1].astype('int')
    return caracteristiques, etiquettes


def charger_chemins_images(dossier_dataset, dict_classes):
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
    return chemins, np.array(etiquettes)


def charger_modele(nom_descripteur):
    chemin = os.path.join(DOSSIER_MODELES, f'meilleur_modele_{nom_descripteur}.joblib')
    return joblib.load(chemin)


def charger_dict_classes():
    chemin = os.path.join(DOSSIER_SIGNATURES, 'class_mapping.npy')
    return np.load(chemin, allow_pickle=True).item()


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
    caracteristiques_requete = np.array(fonction_descripteur(chemin_requete))

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

    # Calcul des distances
    fonction_distance = MESURES_DISTANCE[nom_distance]
    distances = [fonction_distance(caracteristiques_requete, signature) for signature in signatures_classe]

    # Tri par distance croissante
    indices_tries = np.argsort(distances)
    resultats = [
        (chemins_classe[i], distances[i])
        for i in indices_tries[:nb_resultats]
    ]

    return classe_predite, resultats
