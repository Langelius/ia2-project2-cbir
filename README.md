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
├── extraction.py         # Extraction parallèle et sauvegarde des signatures
├── classification.py     # Entraînement par validation croisée et sauvegarde du meilleur modèle
├── cbir.py               # Moteur de recherche CBIR avec mesures de distance
├── app.py                # Interface web Streamlit
└── requirements.txt      # Dépendances Python
```

---

## Étapes de développement

### Partie 1 — Extraction et classification

#### 1. Descripteurs (`descripteurs.py`)
Extraction de caractéristiques visuelles sur les 3 canaux RGB indépendamment. L'image est chargée **une seule fois** puis transmise à toutes les fonctions de descripteurs pour éviter les lectures disque redondantes.

| Descripteur | Caractéristiques | Détail |
|-------------|-----------------|--------|
| GLCM | 18 | 6 propriétés × 3 canaux (contraste, homogénéité, énergie, dissimilarité, corrélation, ASM) |
| Haralick | 39 | 13 propriétés × 3 canaux |
| BiT | 42 | 14 propriétés × 3 canaux |
| Concaténation | 99 | GLCM + Haralick + BiT |

**Choix technique :** traitement canal par canal (RGB) plutôt qu'en niveaux de gris pour conserver l'information couleur, essentielle à la discrimination d'animaux.

**Robustesse :** les valeurs NaN/inf produites par le descripteur BiT (division par zéro dans les indices de biodiversité) sont corrigées à la source dans `bitdesc_feat_RGB` via `np.nan_to_num`.

#### 2. Extraction (`extraction.py`)
- Parcours récursif du dataset
- Construction automatique du dictionnaire de classes depuis les noms de dossiers (trié alphabétiquement pour garantir un mapping stable)
- **Traitement parallèle** sur tous les cœurs CPU disponibles (`ProcessPoolExecutor`) avec `executor.map` pour garantir l'ordre d'insertion
- Sauvegarde de 4 fichiers `.npy` (un par descripteur) dans `signatures/`
- Sauvegarde du fichier `chemins.npy` (liste ordonnée des chemins d'images) pour garantir la synchronisation avec les signatures dans `cbir.py`
- Sauvegarde du mapping `classe → indice` dans `class_mapping.npy`

#### 3. Classification (`classification.py`)
Trois modèles combinés avec 4 normalisations via `sklearn.Pipeline` :

| Modèle | Alias |
|--------|-------|
| Decision Tree | DTC |
| Random Forest | RFC |
| SVM | SVC |

| Normalisation | Rôle |
|--------------|------|
| Aucune | Données brutes |
| StandardScaler | Centrage et réduction (moyenne=0, écart-type=1) |
| MinMaxScaler | Mise à l'échelle entre 0 et 1 |
| Normalizer | Normalisation L2 par ligne |

Métriques évaluées : Accuracy, Précision, Rappel, F1-Score (average weighted).

**Évaluation par validation croisée 5-fold (`StratifiedKFold`) :** les métriques affichées sont des moyennes sur 5 partitions, ce qui rend la sélection de modèle plus robuste qu'un seul split 70/30. La stratification garantit une représentation équilibrée de chaque classe dans chaque fold.

**Réentraînement sur 100% des données :** une fois le meilleur pipeline sélectionné par CV, il est réentraîné sur l'ensemble du dataset avant d'être sauvegardé en `.joblib` dans `models/`.

**Choix technique :** utilisation de `Pipeline` sklearn avec `clone()` pour que chaque combinaison reçoive une instance fraîche du modèle et du scaler — évite le partage d'état entre pipelines. Le scaler est ainsi appliqué automatiquement à l'inférence dans `cbir.py` sans modification.

**Reproductibilité :** `random_state=42` sur tous les modèles et sur le `StratifiedKFold`.

---

### Partie 2 — Moteur CBIR (`cbir.py`)

#### Mesures de distance implémentées

| Distance | Formule |
|----------|---------|
| Euclidienne | √Σ(aᵢ − bᵢ)² |
| Canberra | Σ \|aᵢ − bᵢ\| / (\|aᵢ\| + \|bᵢ\|) |
| Cosinus | 1 − (a·b) / (‖a‖ × ‖b‖) |

Les trois fonctions de distance sont **vectorisées** (opérations NumPy sur matrices) — aucune boucle Python sur les images.

#### Pipeline de recherche
1. Extraction des caractéristiques de l'image requête
2. Prédiction de sa classe avec le meilleur modèle sauvegardé (le scaler du pipeline est appliqué automatiquement)
3. Filtrage du dataset — seules les images de la classe prédite sont comparées
4. Application du scaler du pipeline aux features requête et aux signatures filtrées pour des distances cohérentes avec l'espace de classification
5. Calcul vectorisé des distances
6. Retour des N images les plus proches (distance croissante), N plafonné à la taille de la classe

**Choix technique :** le filtrage par classe réduit la recherche de 1200 à ~60 images, améliorant la précision et la vitesse.

**Performance :** les modèles, signatures et chemins sont mis en cache en mémoire (`_cache`) — les requêtes successives n'impliquent aucune lecture disque.

**Synchronisation signatures/chemins :** `chemins.npy` est généré par `extraction.py` dans le même ordre que les signatures, garantissant la correspondance entre chaque ligne de signature et son chemin d'image.

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
- `chemins.npy`
- `class_mapping.npy`

> ⚠️ Cette étape utilise tous les cœurs CPU disponibles. Le temps d'extraction est réduit proportionnellement au nombre de cœurs.

### Étape 2 — Entraînement et sélection du meilleur modèle

```bash
python classification.py
```

Affiche les métriques comparatives (moyennes sur 5 folds) et sauvegarde les meilleurs modèles dans `models/`.

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
| scikit-learn | Modèles de classification, Pipeline, validation croisée |
| numpy / pandas | Manipulation des données |
| joblib | Sauvegarde et chargement des modèles |
| Streamlit | Interface web |
