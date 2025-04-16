import pandas as pd
import tensorflow as tf
from pathlib import Path

def load_data(file_path):
    """
    Charge les données prétraitées à partir d'un fichier Parquet.

    Args:
        file_path (str): Chemin vers le fichier Parquet.

    Returns:
        pd.DataFrame: Données chargées.
    """
    return pd.read_parquet(file_path)

def split_features_labels(data, label_columns=['trading_signal', 'volatility', 'market_regime']):
    """
    Sépare les features (X) et les labels (y) à partir d'un DataFrame pour un modèle multi-tâches.

    Args:
        data (pd.DataFrame): Données à séparer.
        label_columns (list): Liste des noms des colonnes contenant les labels.

    Returns:
        tuple: Tuple contenant les features (X) et un dictionnaire de labels (y).
               Le dictionnaire y a pour clés les noms des labels et pour valeurs les Series Pandas.
    """
    # Vérifier que toutes les colonnes de labels existent
    missing_labels = [col for col in label_columns if col not in data.columns]
    if missing_labels:
        raise ValueError(f"Les colonnes de labels suivantes sont manquantes dans les données : {missing_labels}")

    # Les features X sont toutes les colonnes sauf les colonnes de labels.
    X = data.drop(columns=label_columns)
    
    # Sélectionner uniquement les colonnes numériques pour X
    # Exclure également les colonnes qui pourraient être catégorielles ou temporelles si elles n'ont pas été gérées en amont.
    # Exemple : si une colonne 'timestamp' ou 'category_feature' existe encore.
    potential_non_numeric_cols = [] # Ajoutez ici si nécessaire
    cols_to_exclude_from_X = [col for col in potential_non_numeric_cols if col in X.columns]
    if cols_to_exclude_from_X:
        X = X.drop(columns=cols_to_exclude_from_X)
        
    X = X.select_dtypes(include=['number']) 
    
    # Les labels y sont un dictionnaire contenant chaque colonne de label spécifiée.
    y = {col: data[col] for col in label_columns}
    
    return X, y

def load_and_split_data(file_path, label_columns=['trading_signal', 'volatility', 'market_regime'], as_tensor=False):
    """
    Charge les données prétraitées et les sépare en features (X) et un dictionnaire de labels (y).

    Args:
        file_path (str): Chemin vers le fichier Parquet.
        label_columns (list): Liste des noms des colonnes contenant les labels.
        as_tensor (bool): Si True, retourne les données sous forme de tenseurs TensorFlow.

    Returns:
        tuple: Tuple contenant les features (X) et un dictionnaire de labels (y).
               Si as_tensor est True, X et les valeurs de y sont des tf.Tensor.
    """
    data = load_data(file_path)
    X, y_dict = split_features_labels(data, label_columns)
    
    if as_tensor:
        X = tf.convert_to_tensor(X.values, dtype=tf.float32)
        # Convertir chaque label dans le dictionnaire en tenseur
        # Adapter le dtype si nécessaire (ex: tf.int64 pour classification)
        y_tensors = {}
        for name, series in y_dict.items():
            dtype = tf.float32 # Par défaut
            if name == 'trading_signal' or name == 'market_regime': # Supposons que ce sont des labels de classification
                 dtype = tf.int64 # Ou le type approprié pour vos labels de classification
            y_tensors[name] = tf.convert_to_tensor(series.values, dtype=dtype)
        return X, y_tensors
        
    return X, y_dict
