import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
from sklearn.pipeline import Pipeline

DOSSIER_SIGNATURES = 'signatures'
DOSSIER_MODELES = 'models'
N_FOLDS = 5

DESCRIPTEURS_DISPO = ['glcm', 'haralick', 'bitdesc', 'concat']

MODELES = [
    ('Decision Tree', DTC(random_state=42)),
    ('Random Forest', RFC(random_state=42)),
    ('SVM',           SVC(random_state=42)),
]

SCALERS = [
    ('Aucune',     None),
    ('Standard',   StandardScaler()),
    ('MinMax',     MinMaxScaler()),
    ('Normalizer', Normalizer()),
]


def charger_signatures(nom_descripteur):
    chemin = os.path.join(DOSSIER_SIGNATURES, f'signatures_{nom_descripteur}.npy')
    tableau = np.load(chemin)
    caracteristiques = tableau[:, :-1].astype('float')
    etiquettes = tableau[:, -1].astype('int')
    return caracteristiques, etiquettes


def construire_pipeline(nom_scaler, scaler, nom_modele, modele):
    nom = f'{nom_scaler} + {nom_modele}'
    if scaler is None:
        pipeline = Pipeline([('modele', clone(modele))])
    else:
        pipeline = Pipeline([('scaler', clone(scaler)), ('modele', clone(modele))])
    return nom, pipeline


def evaluer_modeles(caracteristiques, etiquettes):
    resultats = []
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    scoring = {
        'accuracy':  'accuracy',
        'precision': 'precision_weighted',
        'recall':    'recall_weighted',
        'f1':        'f1_weighted',
    }

    for nom_scaler, scaler in SCALERS:
        for nom_modele, modele in MODELES:
            _, pipeline = construire_pipeline(nom_scaler, scaler, nom_modele, modele)

            scores = cross_validate(pipeline, caracteristiques, etiquettes, cv=cv, scoring=scoring)

            resultats.append({
                'Normalisation': nom_scaler,
                'Modèle':        nom_modele,
                'Accuracy':      round(scores['test_accuracy'].mean(), 4),
                'Précision':     round(scores['test_precision'].mean(), 4),
                'Rappel':        round(scores['test_recall'].mean(), 4),
                'F1-Score':      round(scores['test_f1'].mean(), 4),
            })

    return pd.DataFrame(resultats)


def afficher_matrices_confusion(caracteristiques, etiquettes, df_resultats, dict_classes):
    noms_classes = [nom for nom, _ in sorted(dict_classes.items(), key=lambda x: x[1])]
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    for _, ligne in df_resultats.iterrows():
        nom_scaler = ligne['Normalisation']
        nom_modele = ligne['Modèle']
        scaler = dict(SCALERS)[nom_scaler]
        modele = dict(MODELES)[nom_modele]
        nom_pipeline, pipeline = construire_pipeline(nom_scaler, scaler, nom_modele, modele)

        matrice_totale = np.zeros((len(noms_classes), len(noms_classes)), dtype=int)
        for idx_train, idx_test in cv.split(caracteristiques, etiquettes):
            pipeline_fold = clone(pipeline)
            pipeline_fold.fit(caracteristiques[idx_train], etiquettes[idx_train])
            predictions = pipeline_fold.predict(caracteristiques[idx_test])
            matrice_totale += confusion_matrix(etiquettes[idx_test], predictions, labels=list(range(len(noms_classes))))

        df_matrice = pd.DataFrame(matrice_totale, index=noms_classes, columns=noms_classes)
        print(f'\nMatrice de confusion — {nom_pipeline}:')
        print(df_matrice)


def sauvegarder_meilleur_modele(caracteristiques, etiquettes, df_resultats, nom_descripteur):
    os.makedirs(DOSSIER_MODELES, exist_ok=True)

    idx_meilleur = df_resultats['Accuracy'].idxmax()
    meilleure_norm      = df_resultats.loc[idx_meilleur, 'Normalisation']
    meilleur_modele_nom = df_resultats.loc[idx_meilleur, 'Modèle']

    scaler = dict(SCALERS)[meilleure_norm]
    modele = dict(MODELES)[meilleur_modele_nom]
    nom_pipeline, pipeline = construire_pipeline(meilleure_norm, scaler, meilleur_modele_nom, modele)

    pipeline.fit(caracteristiques, etiquettes)

    chemin_modele = os.path.join(DOSSIER_MODELES, f'meilleur_modele_{nom_descripteur}.joblib')
    joblib.dump(pipeline, chemin_modele)
    print(f'\nMeilleur pipeline : {nom_pipeline} (sauvegardé dans {chemin_modele})')
    return nom_pipeline, chemin_modele


def entrainer(nom_descripteur):
    print(f'\n{"="*60}')
    print(f' Descripteur : {nom_descripteur.upper()}')
    print(f'{"="*60}')

    caracteristiques, etiquettes = charger_signatures(nom_descripteur)

    df_resultats = evaluer_modeles(caracteristiques, etiquettes)
    print(df_resultats.set_index(['Normalisation', 'Modèle']).to_string())

    dict_classes = np.load(
        os.path.join(DOSSIER_SIGNATURES, 'class_mapping.npy'), allow_pickle=True
    ).item()
    afficher_matrices_confusion(caracteristiques, etiquettes, df_resultats, dict_classes)

    sauvegarder_meilleur_modele(caracteristiques, etiquettes, df_resultats, nom_descripteur)
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
        print(df_global.set_index(['Descripteur', 'Normalisation', 'Modèle']).to_string())


if __name__ == '__main__':
    main()
