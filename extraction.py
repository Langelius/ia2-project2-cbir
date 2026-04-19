import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from descripteurs import glcm_RGB, haralick_feat_RGB, bitdesc_feat_RGB, concat_RGB, charger_image

DOSSIER_DATASET = 'dataset'
DOSSIER_SIGNATURES = 'signatures'


def construire_dict_classes(dossier_dataset):
    classes = sorted([
        nom_dossier for nom_dossier in os.listdir(dossier_dataset)
        if os.path.isdir(os.path.join(dossier_dataset, nom_dossier))
    ])
    return {nom: indice_classe for indice_classe, nom in enumerate(classes)}


def traiter_image(args):
    chemin, etiquette = args
    try:
        image_rgb = charger_image(chemin)
        return chemin, {
            'glcm':     glcm_RGB(image_rgb)          + [etiquette],
            'haralick': haralick_feat_RGB(image_rgb) + [etiquette],
            'bitdesc':  bitdesc_feat_RGB(image_rgb)  + [etiquette],
            'concat':   concat_RGB(image_rgb)        + [etiquette],
        }
    except Exception as e:
        return chemin, None, str(e)


def collecter_images(dossier_dataset, dict_classes):
    args = []
    for racine, _, fichiers in os.walk(dossier_dataset):
        for fichier in sorted(fichiers):
            if not fichier.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            nom_classe = os.path.basename(racine)
            if nom_classe not in dict_classes:
                continue
            chemin = os.path.join(racine, fichier)
            etiquette = dict_classes[nom_classe]
            args.append((chemin, etiquette))
    return args


def extraction(dossier_dataset, dossier_signatures, dict_classes):
    os.makedirs(dossier_signatures, exist_ok=True)

    liste_args = collecter_images(dossier_dataset, dict_classes)
    nb_workers = (os.cpu_count() or 4) - 1
    print(f'{len(liste_args)} images à traiter sur {nb_workers} cœurs...')

    donnees = {'glcm': [], 'haralick': [], 'bitdesc': [], 'concat': []}

    with ProcessPoolExecutor(max_workers=nb_workers) as executor:
        futures = {executor.submit(traiter_image, args): args[0] for args in liste_args}
        for future in as_completed(futures):
            resultat = future.result()
            if len(resultat) == 3:
                print(f'[ERREUR] {resultat[0]} : {resultat[2]}')
                continue
            chemin, descripteurs = resultat
            for nom_desc in donnees:
                donnees[nom_desc].append(descripteurs[nom_desc])
            print(f'[OK] {chemin}')

    for nom_desc, lignes in donnees.items():
        tableau = np.array(lignes)
        chemin_npy = os.path.join(dossier_signatures, f'signatures_{nom_desc}.npy')
        np.save(chemin_npy, tableau)
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
