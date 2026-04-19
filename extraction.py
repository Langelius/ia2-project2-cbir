import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd

from descripteurs import glcm_RGB, haralick_feat_RGB, bitdesc_feat_RGB, concat_RGB, charger_image

DOSSIER_DATASET = 'dataset'
DOSSIER_SIGNATURES = 'signatures'

DESCRIPTEURS = {
    'glcm':     glcm_RGB,
    'haralick': haralick_feat_RGB,
    'bitdesc':  bitdesc_feat_RGB,
    'concat':   concat_RGB,
}


def construire_dict_classes(dossier_dataset):
    classes = sorted([
        nom_dossier for nom_dossier in os.listdir(dossier_dataset)
        if os.path.isdir(os.path.join(dossier_dataset, nom_dossier))
    ])
    return {nom: indice_classe for indice_classe, nom in enumerate(classes)}


def extraction(dossier_dataset, dossier_signatures, dict_classes):
    os.makedirs(dossier_signatures, exist_ok=True)

    donnees = {nom: [] for nom in DESCRIPTEURS}

    for racine, _, fichiers in os.walk(dossier_dataset):
        for fichier in sorted(fichiers):
            if not fichier.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            chemin = os.path.join(racine, fichier)
            nom_classe = os.path.basename(racine)
            if nom_classe not in dict_classes:
                continue
            etiquette = dict_classes[nom_classe]

            try:
                image_rgb = charger_image(chemin)
                for nom_desc, fn in DESCRIPTEURS.items():
                    ligne = []
                    ligne.extend(fn(image_rgb))
                    ligne.append(etiquette)
                    donnees[nom_desc].append(ligne)
                print(f'[OK] {chemin}')
            except Exception as e:
                print(f'[ERREUR] {chemin} : {e}')

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
