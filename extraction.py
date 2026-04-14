import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

DOSSIER_DATASET = 'dataset'
DOSSIER_SIGNATURES = 'signatures'

NOMS_DESCRIPTEURS = ['glcm', 'haralick', 'bitdesc', 'concat',
                     'hist', 'glcm_hist', 'haralick_hist', 'bitdesc_hist', 'concat_hist']


def traiter_image(args):
    """
    Traite une image et retourne ses caractéristiques pour tous les descripteurs.
    Fonction au niveau module (obligatoire pour multiprocessing sur Windows).
    L'image est lue une seule fois par descripteur qui en a besoin.
    """
    import warnings
    warnings.filterwarnings("ignore")

    from descripteurs import (glcm_RGB, haralick_feat_RGB, bitdesc_feat_RGB, concat_RGB,
                              histogramme_RGB, glcm_hist_RGB, haralick_hist_RGB,
                              bitdesc_hist_RGB, concat_hist_RGB)

    descripteurs = {
        'glcm':          glcm_RGB,
        'haralick':      haralick_feat_RGB,
        'bitdesc':       bitdesc_feat_RGB,
        'concat':        concat_RGB,
        'hist':          histogramme_RGB,
        'glcm_hist':     glcm_hist_RGB,
        'haralick_hist': haralick_hist_RGB,
        'bitdesc_hist':  bitdesc_hist_RGB,
        'concat_hist':   concat_hist_RGB,
    }

    chemin, etiquette = args
    resultats = {}

    try:
        for nom_desc, fn in descripteurs.items():
            ligne = []
            ligne.extend(fn(chemin))
            ligne.append(etiquette)
            resultats[nom_desc] = ligne
    except Exception as e:
        return chemin, None, str(e)

    return chemin, resultats, None


def construire_dict_classes(dossier_dataset):
    classes = sorted([
        nom_dossier for nom_dossier in os.listdir(dossier_dataset)
        if os.path.isdir(os.path.join(dossier_dataset, nom_dossier))
    ])
    return {nom: indice_classe for indice_classe, nom in enumerate(classes)}


def collecter_images(dossier_dataset, dict_classes):
    """Retourne la liste de (chemin, etiquette) pour toutes les images du dataset."""
    images = []
    for racine, _, fichiers in os.walk(dossier_dataset):
        nom_classe = os.path.basename(racine)
        if nom_classe not in dict_classes:
            continue
        for fichier in sorted(fichiers):
            if fichier.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                images.append((os.path.join(racine, fichier), dict_classes[nom_classe]))
    return images


def extraction(dossier_dataset, dossier_signatures, dict_classes):
    os.makedirs(dossier_signatures, exist_ok=True)

    liste_images = collecter_images(dossier_dataset, dict_classes)
    nb_images = len(liste_images)
    nb_workers = max(1, multiprocessing.cpu_count() - 1)
    print(f'{nb_images} images à traiter avec {nb_workers} processus parallèles...')

    donnees = {nom: [] for nom in NOMS_DESCRIPTEURS}
    nb_ok = 0
    nb_erreurs = 0

    with ProcessPoolExecutor(max_workers=nb_workers) as executeur:
        futures = {executeur.submit(traiter_image, args): args[0] for args in liste_images}

        for future in as_completed(futures):
            chemin, resultats, erreur = future.result()
            if erreur:
                print(f'[ERREUR] {chemin} : {erreur}')
                nb_erreurs += 1
            else:
                for nom_desc, ligne in resultats.items():
                    donnees[nom_desc].append(ligne)
                nb_ok += 1
                print(f'[{nb_ok}/{nb_images}] {chemin}')

    print(f'\nExtraction terminée : {nb_ok} OK, {nb_erreurs} erreurs')

    for nom_desc, lignes in donnees.items():
        tableau = np.array(lignes)
        # Nettoyage des valeurs invalides produites par bio_taxo (division par zéro dans le log)
        tableau = np.nan_to_num(tableau, nan=0.0, posinf=0.0, neginf=0.0)
        chemin_npy = os.path.join(dossier_signatures, f'signatures_{nom_desc}.npy')
        chemin_csv = os.path.join(dossier_signatures, f'signatures_{nom_desc}.csv')
        np.save(chemin_npy, tableau)
        pd.DataFrame(tableau).to_csv(chemin_csv, index=False)
        print(f'Sauvegardé : {chemin_npy}  forme={tableau.shape}')

    chemin_mapping = os.path.join(dossier_signatures, 'class_mapping.npy')
    np.save(chemin_mapping, dict_classes)
    print(f'Mapping des classes sauvegardé : {chemin_mapping}')


def main():
    dict_classes = construire_dict_classes(DOSSIER_DATASET)
    print(f'{len(dict_classes)} classes détectées : {list(dict_classes.keys())}')
    extraction(DOSSIER_DATASET, DOSSIER_SIGNATURES, dict_classes)


if __name__ == '__main__':
    main()
