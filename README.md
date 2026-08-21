# Atelier Scikit-learn — Prédiction de l'état des capteurs IoT

Deuxième atelier de la série sur les capteurs IoT. Après l'exploration des données
avec Seaborn, l'objectif ici est différent : construire un vrai modèle de Machine
Learning capable de prédire automatiquement si un capteur est en état OK, ALERTE
ou ERREUR, à partir de ses mesures (température, humidité, pression, consommation).

## Ce que fait le projet

Le notebook suit le workflow classique d'un projet ML avec scikit-learn :

- chargement et exploration du dataset, suppression des doublons
- définition de X (les caractéristiques) et y (l'état à prédire)
- découpage train/test en conservant les proportions de chaque classe
- traitement des valeurs manquantes avec un imputeur basé sur la médiane
- mise à l'échelle des données avec StandardScaler
- entraînement d'un modèle KNN (k plus proches voisins)
- évaluation avec accuracy, matrice de confusion et rapport de classification
- sauvegarde du modèle (Joblib et Pickle) puis rechargement pour une nouvelle
  prédiction

## Ce qu'on en retient

La cible `etat` est très déséquilibrée : la grande majorité des mesures sont en
état OK, avec beaucoup moins d'ALERTE et très peu d'ERREUR. Ça change la façon
d'évaluer le modèle : l'accuracy seule ne suffit pas, il faut regarder la
precision, le recall et le F1-score classe par classe pour vérifier que le modèle
détecte vraiment les cas problématiques, et pas seulement les cas OK qui sont
faciles à deviner. Le prétraitement (imputation par la médiane, standardisation)
est calé uniquement sur les données d'entraînement pour éviter toute fuite
d'information vers le jeu de test.

## Structure du projet

```
atelier_scikit-learn_iot/
├── data/
│   └── mesures_capteurs.csv
├── notebooks/
│   └── atelier_scikit-learn_iot.ipynb
└── models/
    ├── modele_capteur.joblib
    └── modele_capteur.pkl
```

## Pour lancer le notebook

```bash
pip install pandas numpy seaborn matplotlib scikit-learn joblib jupyter
jupyter notebook notebooks/atelier_scikit-learn_iot.ipynb
```

## Outils utilisés

Python, Pandas, NumPy, Seaborn, Matplotlib, Scikit-learn, Joblib.
