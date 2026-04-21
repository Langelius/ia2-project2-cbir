import os
import numpy as np
import joblib

from descripteurs import glcm_RGB, haralick_feat_RGB, bitdesc_feat_RGB, concat_RGB, charger_image

dossier_signatures = 'signatures'
dossier_modeles = 'models'

fonctions_descripteurs = {
    'glcm':     glcm_RGB,
    'haralick': haralick_feat_RGB,
    'bitdesc':  bitdesc_feat_RGB,
    'concat':   concat_RGB,
}


def distance_euclidienne(requete, signatures):
    return np.sqrt(np.sum((signatures - requete) ** 2, axis=1))


def distance_canberra(requete, signatures):
    numerateur = np.abs(signatures - requete)
    denominateur = np.abs(signatures) + np.abs(requete)
    with np.errstate(invalid='ignore', divide='ignore'):
        ratio = np.where(denominateur != 0, numerateur / denominateur, 0.0)
    return np.sum(ratio, axis=1)


def distance_cosinus(requete, signatures):
    normes = np.linalg.norm(signatures, axis=1) * np.linalg.norm(requete)
    with np.errstate(invalid='ignore', divide='ignore'):
        similarites = np.where(normes != 0, signatures @ requete / normes, 0.0)
    return 1.0 - similarites


mesures_distance = {
    'euclidienne': distance_euclidienne,
    'canberra':    distance_canberra,
    'cosinus':     distance_cosinus,
}


stockage = {}


def charger_signatures(nom_descripteur):
    key = f'signatures_{nom_descripteur}'
    if key not in stockage:
        chemin = os.path.join(dossier_signatures, f'signatures_{nom_descripteur}.npy')
        tableau = np.load(chemin)
        caracteristiques = tableau[:, :-1].astype('float')
        etiquettes = tableau[:, -1].astype('int')
        stockage[key] = (caracteristiques, etiquettes)
    return stockage[key]


def charger_chemins():
    if 'chemins' not in stockage:
        chemin = os.path.join(dossier_signatures, 'chemins.npy')
        stockage['chemins'] = np.load(chemin, allow_pickle=True).tolist()
    return stockage['chemins']


def charger_modele(nom_descripteur):
    key = f'modele_{nom_descripteur}'
    if key not in stockage:
        chemin = os.path.join(dossier_modeles, f'meilleur_modele_{nom_descripteur}.joblib')
        if not os.path.exists(chemin):
            raise FileNotFoundError(f"Modèle introuvable pour le descripteur '{nom_descripteur}'. Lancez classification.py d'abord.")
        stockage[key] = joblib.load(chemin)
    return stockage[key]


def charger_dict_classes():
    if 'dict_classes' not in stockage:
        chemin = os.path.join(dossier_signatures, 'class_mapping.npy')
        stockage['dict_classes'] = np.load(chemin, allow_pickle=True).item()
    return stockage['dict_classes']


def rechercher(chemin_requete, nom_descripteur, nom_distance, nb_resultats=10):
    dict_classes = charger_dict_classes()
    index_vers_classe = {indice: nom for nom, indice in dict_classes.items()}

    fonction_descripteur = fonctions_descripteurs[nom_descripteur]
    image_rgb = charger_image(chemin_requete)
    caracteristiques_requete = np.array(fonction_descripteur(image_rgb))

    modele = charger_modele(nom_descripteur)
    indice_classe_predite = modele.predict([caracteristiques_requete])[0]
    classe_predite = index_vers_classe[indice_classe_predite]

    signatures, etiquettes = charger_signatures(nom_descripteur)
    chemins = charger_chemins()

    masque_classe = etiquettes == indice_classe_predite
    signatures_classe = signatures[masque_classe]
    chemins_classe = [c for c, ok in zip(chemins, masque_classe) if ok]

    steps = dict(modele.steps)
    if 'scaler' in steps:
        scaler = steps['scaler']
        caracteristiques_requete_norm = scaler.transform([caracteristiques_requete])[0]
        signatures_classe_norm = scaler.transform(signatures_classe)
    else:
        caracteristiques_requete_norm = caracteristiques_requete
        signatures_classe_norm = signatures_classe

    fonction_distance = mesures_distance[nom_distance]
    distances = fonction_distance(caracteristiques_requete_norm, signatures_classe_norm)

    nb_resultats = min(nb_resultats, len(chemins_classe))
    indices_tries = np.argsort(distances)
    resultats = [
        (chemins_classe[i], float(distances[i]))
        for i in indices_tries[:nb_resultats]
    ]

    return classe_predite, resultats
