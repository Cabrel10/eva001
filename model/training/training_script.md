# Documentation du Script d'Entraînement `training_script.py`

## Objectif

Le script `training_script.py` est le composant central pour l'entraînement du modèle Morningstar. Il orchestre le chargement des données, la construction de l'architecture du modèle (définie par l'Agent 1), la compilation avec les fonctions de perte et métriques appropriées pour chaque tâche, l'exécution de la boucle d'entraînement, et la sauvegarde du modèle entraîné.

## Fonctionnement Général

Le script exécute les étapes suivantes :

1.  **Configuration et Imports** : Importe les bibliothèques nécessaires (TensorFlow, Pandas, etc.) et les modules locaux (`data_loader`, `enhanced_hybrid_model`). Définit les chemins importants (données, sauvegarde modèle) et les hyperparamètres par défaut.
2.  **Chargement des Données** : Utilise `load_and_split_data` du module `data_loader.py` pour charger un fichier Parquet spécifique à un actif (ex: `btc_final.parquet`) et le sépare en features (X) et un dictionnaire de labels (y), convertis en tenseurs TensorFlow.
3.  **Séparation Train/Validation** : Divise les données chargées en ensembles d'entraînement et de validation en utilisant un simple découpage temporel (les dernières `validation_split` % données pour la validation).
4.  **Construction du Modèle** : Appelle la fonction `build_enhanced_hybrid_model` (du module `model.architecture.enhanced_hybrid_model`) pour instancier l'architecture du modèle. La forme d'entrée (`input_shape`) est déterminée à partir des données chargées, et le nombre de classes pour les sorties de classification est passé en argument (valeurs codées en dur pour l'instant, à améliorer potentiellement via configuration).
5.  **Compilation du Modèle** : Compile le modèle Keras en spécifiant :
    *   Un optimiseur (`Adam` par défaut).
    *   Des fonctions de perte distinctes pour chaque tête de sortie (`SparseCategoricalCrossentropy` pour la classification, `MeanSquaredError` pour la régression de volatilité). Les noms des sorties (`trading_signal_output`, `volatility_output`, `market_regime_output`) doivent correspondre aux noms des couches finales dans l'architecture du modèle.
    *   Des métriques pour chaque tête de sortie (`accuracy` pour la classification, `RMSE` et `MAE` pour la régression).
    *   Optionnellement, une pondération des pertes peut être appliquée (commentée par défaut).
6.  **Entraînement** : Lance l'entraînement avec `model.fit()`, en utilisant les données d'entraînement et de validation. Deux callbacks sont utilisés :
    *   `ModelCheckpoint` : Sauvegarde le meilleur modèle (basé sur `val_loss`) dans un fichier `.h5` au chemin spécifié.
    *   `EarlyStopping` : Arrête l'entraînement prématurément si la perte de validation ne s'améliore pas pendant un certain nombre d'époques (`patience`), et restaure les poids du meilleur modèle trouvé.
7.  **Sauvegarde** : Le meilleur modèle est automatiquement sauvegardé par `ModelCheckpoint`. Le script affiche le chemin du fichier sauvegardé.
8.  **Évaluation** : Évalue le modèle entraîné sur l'ensemble de validation et affiche les métriques finales.
9.  **Interface Ligne de Commande** : Le script peut être exécuté directement depuis le terminal, acceptant plusieurs arguments pour personnaliser l'entraînement.

## Utilisation en Ligne de Commande

Le script peut être lancé depuis la racine du projet (`Morningstar/`) comme suit :

```bash
# Activer l'environnement conda
conda activate trading_env 

# Lancer l'entraînement avec les paramètres par défaut (actif=btc, epochs=50, etc.)
python model/training/training_script.py

# Lancer l'entraînement pour un autre actif (ex: eth) avec 100 epochs
python model/training/training_script.py --asset eth --epochs 100

# Spécifier un taux d'apprentissage et un chemin de sauvegarde différent
python model/training/training_script.py --asset sol --lr 0.0005 --save_path model/training/sol_model_v1.h5 
```

### Arguments Disponibles

*   `--asset` : Nom de l'actif à utiliser (défaut: `btc`). Le script cherchera `data/processed/{asset}_final.parquet`.
*   `--epochs` : Nombre d'époques d'entraînement (défaut: 50).
*   `--batch_size` : Taille du batch (défaut: 32).
*   `--lr` : Taux d'apprentissage pour l'optimiseur Adam (défaut: 0.001).
*   `--val_split` : Proportion des données à utiliser pour la validation (défaut: 0.2).
*   `--save_path` : Chemin complet pour sauvegarder le meilleur modèle `.h5` (défaut: `model/training/saved_model.h5`).

## Dépendances

*   `model/training/data_loader.py` : Pour charger et préparer les données.
*   `model/architecture/enhanced_hybrid_model.py` : Pour fournir la fonction `build_enhanced_hybrid_model`.
*   Fichiers de données Parquet dans `data/processed/`.
*   Bibliothèques listées dans `requirements.txt` (TensorFlow, Pandas, NumPy, PyYAML).

## Sortie Attendue

*   Affichage de la progression de l'entraînement dans le terminal.
*   Un fichier `.h5` contenant le meilleur modèle sauvegardé au chemin spécifié (`model/training/saved_model.h5` par défaut).
*   Affichage des métriques d'évaluation finales sur l'ensemble de validation.
