# Documentation du Module `data_loader.py`

## Objectif

Le module `data_loader.py` est responsable du chargement des datasets prétraités (au format Parquet) et de leur préparation pour l'entraînement du modèle Morningstar. Il sépare les features (variables explicatives, X) des différentes cibles (labels, y) nécessaires pour le modèle multi-tâches.

## Fonctions Principales

1.  **`load_data(file_path)`**:
    *   Charge un fichier Parquet spécifié par `file_path`.
    *   Retourne un DataFrame Pandas.

2.  **`split_features_labels(data, label_columns=['trading_signal', 'volatility', 'market_regime'])`**:
    *   Prend en entrée un DataFrame Pandas (`data`).
    *   Sépare les features (X) des labels (y) en se basant sur la liste `label_columns`.
    *   **Features (X)**: Contient toutes les colonnes du DataFrame `data` *sauf* celles spécifiées dans `label_columns`. Seules les colonnes de type numérique sont conservées.
    *   **Labels (y)**: Est un dictionnaire Python où :
        *   Les clés sont les noms des colonnes de labels (ex: `'trading_signal'`, `'volatility'`, `'market_regime'`).
        *   Les valeurs sont les Series Pandas correspondantes à ces colonnes.
    *   Retourne un tuple `(X, y)`.
    *   Lève une `ValueError` si une des colonnes de labels spécifiées n'est pas trouvée dans le DataFrame.

3.  **`load_and_split_data(file_path, label_columns=['trading_signal', 'volatility', 'market_regime'], as_tensor=False)`**:
    *   Fonction principale qui orchestre le chargement et la séparation.
    *   Appelle `load_data` pour charger le fichier Parquet.
    *   Appelle `split_features_labels` pour séparer X et y.
    *   **Paramètre `as_tensor`**:
        *   Si `False` (défaut), retourne X (DataFrame) et y (dictionnaire de Series).
        *   Si `True`, convertit X et chaque Series dans le dictionnaire y en tenseurs TensorFlow (`tf.Tensor`). Le `dtype` des tenseurs de labels est ajusté (`tf.int64` pour `trading_signal` et `market_regime`, `tf.float32` pour `volatility` par défaut).
    *   Retourne le tuple `(X, y)` sous forme de DataFrames/Series ou Tensors.

## Utilisation Typique

```python
from model.training.data_loader import load_and_split_data

# Chemin vers un fichier de données prétraité
file_path = 'data/processed/btc_final.parquet' 

# Noms des colonnes cibles pour le modèle
label_cols = ['trading_signal', 'volatility', 'market_regime'] 

# Charger les données sous forme de Tensors pour l'entraînement TensorFlow
X_tensor, y_tensors_dict = load_and_split_data(file_path, label_columns=label_cols, as_tensor=True)

# X_tensor est un tf.Tensor contenant les features
# y_tensors_dict est un dictionnaire {'trading_signal': tf.Tensor, 'volatility': tf.Tensor, 'market_regime': tf.Tensor}

# Charger les données sous forme de DataFrames/Series pour analyse
X_df, y_series_dict = load_and_split_data(file_path, label_columns=label_cols, as_tensor=False) 
```

## Prérequis

*   Les fichiers de données d'entrée doivent être au format Parquet et se trouver dans le dossier `data/processed/`.
*   Ces fichiers doivent contenir les colonnes de features numériques ainsi que les colonnes de labels spécifiées (par défaut : `trading_signal`, `volatility`, `market_regime`).
