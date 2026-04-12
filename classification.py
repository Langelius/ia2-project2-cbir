import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC

DOSSIER_SIGNATURES = 'signatures'
DOSSIER_MODELES = 'models'

DESCRIPTEURS_DISPO = ['glcm', 'haralick', 'bitdesc', 'concat']

MODELES = [
    ('Decision Tree',  DTC()),
    ('Random Forest',  RFC()),
    ('SVM',            SVC()),
]

METRIQUES = [
    ('Accuracy',  accuracy_score),
    ('Précision', precision_score),
    ('Rappel',    recall_score),
    ('F1-Score',  f1_score),
]


def charger_signatures(nom_descripteur):
    chemin = os.path.join(DOSSIER_SIGNATURES, f'signatures_{nom_descripteur}.npy')
    tableau = np.load(chemin)
    X = tableau[:, :-1].astype('float')
    y = tableau[:, -1].astype('int')
    return X, y


def evaluer_modeles(X_entrainement, X_test, y_entrainement, y_test):
    resultats = []
    modeles_entraines = {}

    for nom_modele, modele in MODELES:
        modele.fit(X_entrainement, y_entrainement)
        predictions = modele.predict(X_test)
        modeles_entraines[nom_modele] = (modele, predictions)

        for nom_metrique, fn_metrique in METRIQUES:
            if nom_metrique == 'Accuracy':
                score = fn_metrique(y_test, predictions)
            else:
                score = fn_metrique(y_test, predictions, average='weighted', zero_division=0)

            resultats.append({
                'Modèle':   nom_modele,
                'Métrique': nom_metrique,
                'Score':    round(score, 4),
            })

    return pd.DataFrame(resultats), modeles_entraines


def afficher_matrices_confusion(modeles_entraines, y_test, dict_classes):
    noms_classes = [nom for nom, _ in sorted(dict_classes.items(), key=lambda x: x[1])]
    for nom_modele, (_, predictions) in modeles_entraines.items():
        matrice = confusion_matrix(y_test, predictions)
        df_matrice = pd.DataFrame(matrice, index=noms_classes, columns=noms_classes)
        print(f'\nMatrice de confusion — {nom_modele}:')
        print(df_matrice)


def sauvegarder_meilleur_modele(df_resultats, modeles_entraines, nom_descripteur):
    os.makedirs(DOSSIER_MODELES, exist_ok=True)

    # Sélection du meilleur modèle selon l'accuracy
    df_accuracy = df_resultats[df_resultats['Métrique'] == 'Accuracy']
    meilleur_nom = df_accuracy.loc[df_accuracy['Score'].idxmax(), 'Modèle']
    meilleur_modele, _ = modeles_entraines[meilleur_nom]

    chemin_modele = os.path.join(DOSSIER_MODELES, f'meilleur_modele_{nom_descripteur}.joblib')
    joblib.dump(meilleur_modele, chemin_modele)
    print(f'\nMeilleur modèle : {meilleur_nom} (sauvegardé dans {chemin_modele})')
    return meilleur_nom, chemin_modele


def entrainer(nom_descripteur):
    print(f'\n{"="*60}')
    print(f' Descripteur : {nom_descripteur.upper()}')
    print(f'{"="*60}')

    X, y = charger_signatures(nom_descripteur)
    X_entrainement, X_test, y_entrainement, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    df_resultats, modeles_entraines = evaluer_modeles(
        X_entrainement, X_test, y_entrainement, y_test
    )
    print(df_resultats.to_string(index=False))

    dict_classes = np.load(
        os.path.join(DOSSIER_SIGNATURES, 'class_mapping.npy'), allow_pickle=True
    ).item()
    afficher_matrices_confusion(modeles_entraines, y_test, dict_classes)

    meilleur_nom, chemin_modele = sauvegarder_meilleur_modele(
        df_resultats, modeles_entraines, nom_descripteur
    )
    return df_resultats


def main():
    tous_resultats = []

    for nom_desc in DESCRIPTEURS_DISPO:
        chemin_sig = os.path.join(DOSSIER_SIGNATURES, f'signatures_{nom_desc}.npy')
        if not os.path.exists(chemin_sig):
            print(f'[IGNORÉ] Signatures introuvables pour : {nom_desc}')
            continue
        df = entrainer(nom_desc)
        df['Descripteur'] = nom_desc
        tous_resultats.append(df)

    if tous_resultats:
        df_global = pd.concat(tous_resultats, ignore_index=True)
        print(f'\n{"="*60}')
        print(' Récapitulatif global')
        print(f'{"="*60}')
        print(df_global.pivot_table(
            index=['Descripteur', 'Modèle'], columns='Métrique', values='Score'
        ).to_string())


if __name__ == '__main__':
    main()
