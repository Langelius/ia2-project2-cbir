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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.pipeline import Pipeline

DOSSIER_SIGNATURES = 'signatures'
DOSSIER_MODELES = 'models'

DESCRIPTEURS_DISPO = ['glcm', 'haralick', 'bitdesc', 'concat']

MODELES = [
    ('Decision Tree',  DTC()),
    ('Random Forest',  RFC()),
    ('SVM',            SVC()),
]

# Chaque scaler est combiné avec chaque modèle via un Pipeline
SCALERS = [
    ('Aucune',     None),
    ('Standard',   StandardScaler()),
    ('MinMax',     MinMaxScaler()),
    ('Normalizer', Normalizer()),
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
    # Nettoyage des valeurs invalides produites par bio_taxo (division par zéro dans le log)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = tableau[:, -1].astype('int')
    return X, y


def construire_pipeline(nom_scaler, scaler, nom_modele, modele):
    """Construit un Pipeline sklearn combinant scaler et modèle."""
    nom = f'{nom_scaler} + {nom_modele}'
    if scaler is None:
        pipeline = Pipeline([('modele', modele)])
    else:
        pipeline = Pipeline([('scaler', scaler), ('modele', modele)])
    return nom, pipeline


def evaluer_modeles(X_entrainement, X_test, y_entrainement, y_test):
    resultats = []
    pipelines_entraines = {}

    for nom_scaler, scaler in SCALERS:
        for nom_modele, modele in MODELES:
            nom_pipeline, pipeline = construire_pipeline(nom_scaler, scaler, nom_modele, modele)

            pipeline.fit(X_entrainement, y_entrainement)
            predictions = pipeline.predict(X_test)
            pipelines_entraines[nom_pipeline] = (pipeline, predictions)

            for nom_metrique, fn_metrique in METRIQUES:
                if nom_metrique == 'Accuracy':
                    score = fn_metrique(y_test, predictions)
                else:
                    score = fn_metrique(y_test, predictions, average='weighted', zero_division=0)

                resultats.append({
                    'Normalisation': nom_scaler,
                    'Modèle':        nom_modele,
                    'Métrique':      nom_metrique,
                    'Score':         round(score, 4),
                })

    return pd.DataFrame(resultats), pipelines_entraines


def afficher_matrices_confusion(pipelines_entraines, y_test, dict_classes):
    noms_classes = [nom for nom, _ in sorted(dict_classes.items(), key=lambda x: x[1])]
    for nom_pipeline, (_, predictions) in pipelines_entraines.items():
        matrice = confusion_matrix(y_test, predictions)
        df_matrice = pd.DataFrame(matrice, index=noms_classes, columns=noms_classes)
        print(f'\nMatrice de confusion — {nom_pipeline}:')
        print(df_matrice)


def sauvegarder_meilleur_modele(df_resultats, pipelines_entraines, nom_descripteur):
    os.makedirs(DOSSIER_MODELES, exist_ok=True)

    # Sélection du meilleur pipeline selon l'accuracy
    df_accuracy = df_resultats[df_resultats['Métrique'] == 'Accuracy']
    idx_meilleur = df_accuracy['Score'].idxmax()
    meilleure_norm = df_accuracy.loc[idx_meilleur, 'Normalisation']
    meilleur_modele_nom = df_accuracy.loc[idx_meilleur, 'Modèle']
    nom_pipeline = f'{meilleure_norm} + {meilleur_modele_nom}'

    meilleur_pipeline, _ = pipelines_entraines[nom_pipeline]

    chemin_modele = os.path.join(DOSSIER_MODELES, f'meilleur_modele_{nom_descripteur}.joblib')
    joblib.dump(meilleur_pipeline, chemin_modele)
    print(f'\nMeilleur pipeline : {nom_pipeline} (sauvegardé dans {chemin_modele})')
    return nom_pipeline, chemin_modele


def entrainer(nom_descripteur):
    print(f'\n{"="*60}')
    print(f' Descripteur : {nom_descripteur.upper()}')
    print(f'{"="*60}')

    X, y = charger_signatures(nom_descripteur)
    X_entrainement, X_test, y_entrainement, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    df_resultats, pipelines_entraines = evaluer_modeles(
        X_entrainement, X_test, y_entrainement, y_test
    )
    print(df_resultats.pivot_table(
        index=['Normalisation', 'Modèle'], columns='Métrique', values='Score'
    ).to_string())

    dict_classes = np.load(
        os.path.join(DOSSIER_SIGNATURES, 'class_mapping.npy'), allow_pickle=True
    ).item()
    afficher_matrices_confusion(pipelines_entraines, y_test, dict_classes)

    sauvegarder_meilleur_modele(df_resultats, pipelines_entraines, nom_descripteur)
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
            index=['Descripteur', 'Normalisation', 'Modèle'],
            columns='Métrique',
            values='Score'
        ).to_string())


if __name__ == '__main__':
    main()
