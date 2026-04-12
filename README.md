# Projet 2 — CBIR : Recherche d'images par le contenu

**IA 2 : Vision artificielle et reconnaissance de formes**  
420-1AB-TT — Hivers 2026 — Institut Teccart

---

## Description

Application web de recherche d'images par le contenu (CBIR — *Content-Based Image Retrieval*) intégrant une phase de classification automatique. L'application prédit la classe d'une image requête avant d'effectuer la recherche, ce qui restreint la comparaison aux images pertinentes et améliore la précision.

---

## Structure du projet

```
Projet2/
├── dataset/              # Dataset d'animaux (20 classes, 60 images/classe)
├── signatures/           # Fichiers .npy générés par extraction.py (gitignore)
├── models/               # Modèles sauvegardés par classification.py (gitignore)
├── descripteurs.py       # Fonctions d'extraction de caractéristiques RGB
├── extraction.py         # Extraction et sauvegarde des signatures du dataset
├── classification.py     # Entraînement, évaluation et sauvegarde du meilleur modèle
├── cbir.py               # Moteur de recherche CBIR avec mesures de distance
├── app.py                # Interface web Streamlit
└── requirements.txt      # Dépendances Python
```

---

## Étapes de développement

### Partie 1 — Extraction et classification

#### 1. Descripteurs (`descripteurs.py`)
Extraction de caractéristiques visuelles sur les 3 canaux RGB indépendamment :

| Descripteur | Caractéristiques | Détail |
|-------------|-----------------|--------|
| GLCM | 18 | 6 propriétés × 3 canaux (contraste, homogénéité, énergie, dissimilarité, corrélation, ASM) |
| Haralick | 39 | 13 propriétés × 3 canaux |
| BiT | 42 | 14 propriétés × 3 canaux |
| Concaténation | 99 | GLCM + Haralick + BiT |

**Choix technique :** traitement canal par canal (RGB) plutôt qu'en niveaux de gris pour conserver l'information couleur, essentielle à la discrimination d'animaux.

#### 2. Extraction (`extraction.py`)
- Parcours récursif du dataset
- Construction automatique du dictionnaire de classes depuis les noms de dossiers (trié alphabétiquement pour garantir un mapping stable)
- Sauvegarde de 4 fichiers `.npy` et `.csv` (un par descripteur) dans `signatures/`
- Sauvegarde du mapping `classe → indice` dans `class_mapping.npy`

#### 3. Classification (`classification.py`)
Trois modèles comparés sur chacun des 4 descripteurs :

| Modèle | Alias |
|--------|-------|
| Decision Tree | DTC |
| Random Forest | RFC |
| SVM | SVC |

Métriques évaluées : Accuracy, Précision, Rappel, F1-Score (average weighted).  
Le meilleur modèle (selon l'accuracy) est sauvegardé automatiquement en `.joblib` dans `models/`.

**Choix technique :** séparation train/test 70/30 avec `random_state=42` pour la reproductibilité.

---

### Partie 2 — Moteur CBIR (`cbir.py`)

#### Mesures de distance implémentées

| Distance | Formule |
|----------|---------|
| Euclidienne | √Σ(aᵢ − bᵢ)² |
| Canberra | Σ \|aᵢ − bᵢ\| / (\|aᵢ\| + \|bᵢ\|) |
| Cosinus | 1 − (a·b) / (‖a‖ × ‖b‖) |

#### Pipeline de recherche
1. Extraction des caractéristiques de l'image requête
2. Prédiction de sa classe avec le meilleur modèle sauvegardé
3. Filtrage du dataset — seules les images de la classe prédite sont comparées
4. Calcul des distances entre la requête et les signatures filtrées
5. Retour des N images les plus proches (distance croissante)

**Choix technique :** le filtrage par classe réduit la recherche de 1200 à ~60 images, améliorant la précision et la vitesse.

---

### Partie 3 — Interface web (`app.py`)

Interface développée avec **Streamlit** permettant :
- Téléversement d'une image requête
- Sélection du descripteur (GLCM, Haralick, BiT, Concat)
- Sélection de la mesure de distance (Euclidienne, Canberra, Cosinus)
- Choix du nombre d'images similaires à afficher (1 à 20)
- Affichage de la classe prédite et des résultats en grille avec leur distance

**Choix technique :** Streamlit plutôt que React pour sa simplicité d'intégration avec l'écosystème Python/scikit-learn, sans nécessiter de backend séparé.

---

## Instructions d'exécution

### Prérequis

```bash
pip install -r requirements.txt
```

### Étape 1 — Extraction des signatures

```bash
python extraction.py
```

Génère les fichiers dans `signatures/` :
- `signatures_glcm.npy`
- `signatures_haralick.npy`
- `signatures_bitdesc.npy`
- `signatures_concat.npy`
- `class_mapping.npy`

> ⚠️ Cette étape peut prendre plusieurs minutes selon la machine (1200 images × 4 descripteurs).

### Étape 2 — Entraînement et sélection du meilleur modèle

```bash
python classification.py
```

Affiche les métriques comparatives et sauvegarde les meilleurs modèles dans `models/`.

### Étape 3 — Lancement de l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur à l'adresse `http://localhost:8501`.

---

## Dataset

- **20 classes** d'animaux : bear, bee, beetle, bison, boar, butterfly, cat, caterpillar, chimpanzee, cockroach, cow, coyote, crab, crow, deer, dog, dolphin, donkey, dragonfly, duck
- **60 images par classe** — 1200 images au total
- Format : JPEG

> Le dataset est versionné via **Git LFS** en raison de sa taille (147 MB).

---

## Dépendances principales

| Bibliothèque | Usage |
|-------------|-------|
| OpenCV | Lecture et conversion des images |
| scikit-image | Calcul de la matrice GLCM |
| mahotas | Descripteurs Haralick |
| bitdesc | Descripteurs BiT (Bio-inspired Taxonomy) |
| scikit-learn | Modèles de classification et métriques |
| numpy / pandas | Manipulation des données |
| joblib | Sauvegarde et chargement des modèles |
| Streamlit | Interface web |
