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

def load_and_split_data(file_path, label_columns=['trading_signal', 'volatility', 'market_regime'], label_mappings=None, as_tensor=False):
    """
    Charge les données prétraitées et les sépare en features (X) et un dictionnaire de labels (y).
    Gère le mappage des labels textuels vers des entiers si un dictionnaire de mappage est fourni.

    Args:
        file_path (str): Chemin vers le fichier Parquet.
        label_columns (list): Liste des noms des colonnes contenant les labels.
        label_mappings (dict, optional): Dictionnaire où les clés sont les noms des colonnes de labels
                                         catégoriels et les valeurs sont des dictionnaires mappant
                                         les strings aux entiers. Exemple:
                                         {'market_regime': {'bearish': 0, 'bullish': 1}, ...}.
                                         Defaults to None.
        as_tensor (bool): Si True, retourne les données sous forme de tenseurs TensorFlow.

    Returns:
        tuple: Tuple contenant les features (X) et un dictionnaire de labels (y).
               Si as_tensor est True, X et les valeurs de y sont des tf.Tensor.
    """
    data = load_data(file_path)
    # Séparer les features et labels *après* avoir potentiellement traité sl_tp
    # Garder une copie des colonnes originales pour le cas sl_tp
    original_data_columns = data.columns.tolist()
    
    # Identifier les colonnes de features (tout sauf les labels, sauf si sl/tp sont nécessaires)
    feature_columns = [col for col in data.columns if col not in label_columns]
    # Si sl_tp est un label, s'assurer que level_sl/level_tp ne sont pas dans les features X finales
    # (ils sont utilisés pour créer le label sl_tp mais ne sont pas des inputs directs au modèle dans ce cas)
    if 'sl_tp' in label_columns:
         if 'level_sl' in feature_columns: feature_columns.remove('level_sl')
         if 'level_tp' in feature_columns: feature_columns.remove('level_tp')
            
    X_df = data[feature_columns].select_dtypes(include=['number'])
    y_dict = {col: data[col] for col in label_columns if col != 'sl_tp'} # Exclure sl_tp pour l'instant
    
    if as_tensor:
        X = tf.convert_to_tensor(X_df.values, dtype=tf.float32)
        y_tensors = {}
        
        for name in label_columns: # Itérer sur la liste demandée
            if name == 'sl_tp':
                # Cas spécial pour sl_tp: empiler level_sl et level_tp
                if 'level_sl' in data.columns and 'level_tp' in data.columns:
                    # Assurer que les colonnes sont numériques
                    sl_values = pd.to_numeric(data['level_sl'], errors='coerce').fillna(0).values
                    tp_values = pd.to_numeric(data['level_tp'], errors='coerce').fillna(0).values
                    # Empiler pour obtenir une forme (N, 2)
                    y_tensors['sl_tp'] = tf.stack([sl_values, tp_values], axis=1)
                    y_tensors['sl_tp'] = tf.cast(y_tensors['sl_tp'], dtype=tf.float32) # Assurer float32
                    print("INFO: Label 'sl_tp' créé en empilant 'level_sl' et 'level_tp'.")
                else:
                    raise ValueError("Les colonnes 'level_sl' et 'level_tp' sont nécessaires pour créer le label 'sl_tp' mais sont manquantes.")
            else:
                # Traitement standard pour les autres labels
                series = data[name] # Récupérer la série depuis les données originales
                dtype = tf.float32 # Par défaut
                values_to_convert = series.values

                # Appliquer le mapping si la colonne est dans label_mappings
                if label_mappings is not None and name in label_mappings:
                    dtype = tf.int64
                    current_map = label_mappings[name]
                    
                    # Conversion robuste et vérification
                    str_series = series.astype(str).str.strip()
                    mapped_series = str_series.map(lambda x: current_map.get(x.strip(), float('nan')))
                    
                    # Vérification des valeurs non mappées
                    if mapped_series.isnull().any():
                        unmapped_values = str_series[mapped_series.isnull()].unique()
                        raise ValueError(f"Valeurs non mappées dans '{name}': {list(unmapped_values)}")
                    
                    # Conversion FORCÉE en numérique avant création du tenseur
                    values_to_convert = pd.to_numeric(mapped_series, downcast='integer').values
                    # Si la colonne n'est pas de type 'object' mais qu'un mappage est fourni, 
                    # on pourrait vouloir vérifier si les valeurs numériques correspondent aux clés/valeurs du map?
                    # Pour l'instant, on suppose que si ce n'est pas 'object', c'est déjà numérique et correct.
                    
                elif name in ['trading_signal', 'market_regime']: # Fallback si pas de map fourni mais nom connu
                 dtype = tf.int64 # Supposer classification pour ces noms par défaut
                 # Si la colonne est de type object, lever une erreur car aucun mappage n'a été fourni
                 if series.dtype == 'object':
                     raise ValueError(f"Colonne '{name}' est de type object mais aucun `label_mappings` n'a été fourni pour la convertir.")
            
            # Convertir en tenseur avec le dtype approprié
            try:
                y_tensors[name] = tf.convert_to_tensor(values_to_convert, dtype=dtype)
            except Exception as e:
                print(f"Erreur lors de la conversion en tenseur pour la colonne '{name}' avec dtype {dtype}.")
                # Afficher quelques valeurs pour aider au débogage
                print(f"Premières valeurs avant conversion: {values_to_convert[:10]}") 
                raise e # Relancer l'exception originale
                
        return X, y_tensors
        
    return X, y_dict
