import pandas as pd
import numpy as np
import tensorflow as tf
from pathlib import Path

def load_data(file_path):
    """Charge les données prétraitées à partir d'un fichier Parquet."""
    return pd.read_parquet(file_path)

def _convert_labels_to_tensors(data, label_columns, label_mappings):
    """Convertit les labels en tenseurs TensorFlow avec gestion des mappings."""
    y_tensors = {}

    for name in label_columns:
        # Cas spécial pour sl_tp qui n'est pas une colonne directe mais construite
        if name == 'sl_tp':
            if 'level_sl' in data.columns and 'level_tp' in data.columns:
                 y_tensors.update(_handle_sl_tp(data))
            else:
                 # Si les colonnes sources manquent, on ne peut pas créer sl_tp
                 print(f"WARN: Colonnes 'level_sl'/'level_tp' manquantes, impossible de créer le label 'sl_tp'.")
            continue # Passe au label suivant

        # Pour les autres labels, vérifier qu'ils existent comme colonnes
        if name not in data.columns:
             print(f"WARN: Colonne de label '{name}' non trouvée dans les données.")
             continue # Ignore ce label s'il manque

        series = data[name]
        values, dtype = _process_label_series(series, name, label_mappings)

        try:
            y_tensors[name] = tf.convert_to_tensor(values, dtype=dtype)
        except Exception as e:
            print(f"Erreur conversion {name} (dtype={dtype}): {values[:10]}")
            raise e

    return y_tensors


def _handle_sl_tp(data):
    """Gère la création du tenseur sl_tp à partir de level_sl et level_tp."""
    # La vérification de l'existence des colonnes est faite en amont
    sl_values = pd.to_numeric(data['level_sl'], errors='coerce').fillna(0).values
    tp_values = pd.to_numeric(data['level_tp'], errors='coerce').fillna(0).values

    return {
        'sl_tp': tf.cast(tf.stack([sl_values, tp_values], axis=1), dtype=tf.float32)
    }

def _process_label_series(series, name, label_mappings):
    """Traite une série de labels et retourne les valeurs + dtype approprié."""
    dtype = tf.float32
    values = series.values

    if label_mappings and name in label_mappings:
        return _apply_label_mapping(series, label_mappings[name])

    if name in ['trading_signal', 'market_regime'] and series.dtype == 'object':
        raise ValueError(f"Colonne '{name}' est textuelle mais aucun mapping fourni")

    if name in ['trading_signal', 'market_regime']:
        dtype = tf.int64

    return values, dtype

def _apply_label_mapping(series, mapping_dict):
    """Applique un mapping de valeurs textuelles à numériques."""
    str_series = series.astype(str).str.strip()
    mapped = str_series.map(lambda x, m=mapping_dict: m.get(x.strip(), float('nan')))

    if mapped.isnull().any():
        unmapped = str_series[mapped.isnull()].unique()
        raise ValueError(f"Valeurs non mappées dans '{series.name}': {list(unmapped)}")

    return pd.to_numeric(mapped, downcast='integer').values, tf.int64

def load_and_split_data(file_path, label_columns=None, label_mappings=None, as_tensor=False):
    """Charge et sépare les données avec gestion des types."""
    if label_columns is None:
        label_columns = ['trading_signal', 'volatility', 'market_regime']

    data = load_data(file_path)

    # Vérification des colonnes de labels requises (celles qui ne sont pas 'sl_tp')
    required_label_cols = [col for col in label_columns if col != 'sl_tp']
    missing_labels = [col for col in required_label_cols if col not in data.columns]
    if missing_labels:
        raise ValueError(f"Colonnes de labels manquantes: {missing_labels}")
    # Vérifier aussi les colonnes sources pour sl_tp si sl_tp est demandé
    if 'sl_tp' in label_columns and ('level_sl' not in data.columns or 'level_tp' not in data.columns):
         raise ValueError("Colonnes 'level_sl' et/ou 'level_tp' manquantes pour générer 'sl_tp'.")


    # Extraction des features
    llm_cols = [col for col in data.columns if col.startswith('llm_')]
    
    # Colonnes à exclure (uniquement les labels explicites)
    cols_to_exclude = {'trading_signal', 'volatility', 'market_regime', 'level_sl', 'level_tp'}
    
    # Simplest logic: Include all columns that are NOT explicit labels and NOT LLM embeddings.
    feature_cols = [
        col for col in data.columns
        if col not in cols_to_exclude and not col.startswith('llm_')
    ]

    # Debug logging
    print(f"--- Debug: Data Loader Feature Selection ---")
    print(f"Columns to exclude (labels): {sorted(list(cols_to_exclude))}")
    print(f"LLM columns prefix: 'llm_'")
    print(f"Selected technical features ({len(feature_cols)}):")
    # Print only first 5 and last 5 for brevity if many
    if len(feature_cols) > 10:
         print(f"  First 5: {feature_cols[:5]}")
         print(f"  Last 5: {feature_cols[-5:]}")
    else:
         print(f"  {feature_cols}")
    print(f"--- End Debug ---")
    
    # Debug logging
    print(f"Colonnes techniques détectées ({len(feature_cols)}):")
    for i, col in enumerate(sorted(feature_cols), 1):
        print(f"{i}. {col}")

    # Validation des dimensions
    if len(feature_cols) != 38: # Rétabli la validation pour 38 features
        raise ValueError(f"38 features techniques requises (trouvé {len(feature_cols)}). Vérifiez data_pipeline.py")
    if len(llm_cols) != 768:
        raise ValueError(f"768 embeddings LLM requis (trouvé {len(llm_cols)})")
    print(f"Dimensions validées - Features: 38, LLM: 768")

    # Conversion en tenseurs si demandé
    x_technical = data[feature_cols].values.astype(np.float32)
    x_llm = data[llm_cols].values.astype(np.float32)

    if as_tensor:
        x_technical = tf.convert_to_tensor(x_technical)
        x_llm = tf.convert_to_tensor(x_llm)
        # Passer les colonnes de labels demandées explicitement
        y_data = _convert_labels_to_tensors(data, label_columns, label_mappings)
    else:
        # Retourner uniquement les colonnes de labels demandées
        y_data = {col: data[col] for col in label_columns if col in data.columns}
        # Gérer sl_tp séparément s'il est demandé et possible
        if 'sl_tp' in label_columns and 'level_sl' in data.columns and 'level_tp' in data.columns:
             y_data['sl_tp'] = data[['level_sl', 'level_tp']] # Retourner le DataFrame pour sl_tp

    return (x_technical, x_llm), y_data
