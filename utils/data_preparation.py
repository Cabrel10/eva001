import pandas as pd
import numpy as np
import os
import logging # Importer le module logging
from datetime import timezone # Ajouter l'import
from typing import List, Optional

# Initialiser un logger pour ce module
logger = logging.getLogger(__name__)

# Colonnes minimales attendues dans les fichiers bruts
MINIMAL_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

def load_raw_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Charge un fichier de données brutes (CSV ou JSON) depuis le chemin spécifié.

    Args:
        file_path (str): Chemin complet vers le fichier brut (ex: 'data/raw/BTC_USDT_1h.csv').

    Returns:
        Optional[pd.DataFrame]: DataFrame contenant les données brutes, ou None si le fichier
                                 n'existe pas, n'est pas lisible, ou manque des colonnes essentielles.
    """
    if not os.path.exists(file_path):
        logger.error(f"Le fichier {file_path} n'existe pas.")
        return None

    try:
        if file_path.endswith('.csv'):
            # Charger le CSV en spécifiant que la première colonne est l'index (timestamp)
            # et la parser directement comme date
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
            # Renommer l'index en 'timestamp' s'il n'a pas de nom
            if df.index.name is None or df.index.name == '':
                df.index.name = 'timestamp'
            # S'assurer que l'index est bien de type DatetimeIndex et UTC
            if not isinstance(df.index, pd.DatetimeIndex):
                 raise ValueError("Index n'est pas de type DatetimeIndex après chargement.")
            if df.index.tz is None:
                 df = df.tz_localize('UTC')
            # Comparer directement avec datetime.timezone.utc
            elif df.index.tz is not timezone.utc:
                 df = df.tz_convert('UTC')
            # Remettre le timestamp en colonne pour la vérification initiale
            df = df.reset_index()

        elif file_path.endswith('.json'):
            df = pd.read_json(file_path, lines=True) # Ajuster si le JSON n'est pas line-delimited
        else:
            logger.error(f"Format de fichier non supporté pour {file_path}. Utilisez CSV ou JSON.")
            return None

        logger.info(f"Fichier {file_path} chargé avec succès. Shape: {df.shape}")

        # Vérification des colonnes minimales
        missing_cols = [col for col in MINIMAL_COLUMNS if col not in df.columns]
        if missing_cols:
            logger.error(f"Colonnes manquantes dans {file_path}: {', '.join(missing_cols)}")
            return None

        # Normalisation du timestamp
        # Essayer de convertir en datetime et vérifier si c'est déjà en UTC
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            # Si la conversion réussit sans timezone ou avec une timezone différente, forcer UTC
            if df['timestamp'].dt.tz is None:
                 df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
            # Comparer directement avec datetime.timezone.utc
            elif df['timestamp'].dt.tz is not timezone.utc:
                 df['timestamp'] = df['timestamp'].dt.tz_convert('UTC')

            # Vérifier les NaT introduits par la conversion
            if df['timestamp'].isnull().any():
                logger.warning(f"Des timestamps invalides ont été trouvés et mis à NaT dans {file_path}.")
                # Optionnel: supprimer les lignes avec NaT ou les logger
                # df = df.dropna(subset=['timestamp'])

        except Exception as e:
            logger.error(f"Erreur lors de la conversion du timestamp en UTC pour {file_path}: {e}")
            # Tenter une conversion depuis un format epoch (millisecondes ou secondes) si applicable
            # Exemple: df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            # Si échec, retourner None ou lever une exception plus spécifique
            return None

        # Remettre le timestamp en index après les vérifications et conversions
        if 'timestamp' in df.columns:
             df = df.set_index('timestamp').sort_index()
        else:
             # Cela ne devrait pas arriver si la logique ci-dessus est correcte
             logger.critical("Colonne 'timestamp' non trouvée après traitement initial.")
             return None

        return df

    except Exception as e:
        logger.error(f"Erreur inattendue lors du chargement ou de la validation de {file_path}: {e}")
        return None

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie le DataFrame brut : gère les valeurs manquantes et les outliers.

    Args:
        df (pd.DataFrame): DataFrame brut chargé avec un index Timestamp UTC.

    Returns:
        pd.DataFrame: DataFrame nettoyé.
    """
    logger.info(f"Nettoyage des données - Shape initiale: {df.shape}")
    initial_rows = len(df)

    # 1. Gestion des valeurs manquantes (NaN)
    # Utiliser ffill car les données de marché dépendent souvent de la valeur précédente
    cols_to_fill = ['open', 'high', 'low', 'close', 'volume'] # Adapter si d'autres colonnes numériques existent
    for col in cols_to_fill:
        if col in df.columns:
            original_nan_count = df[col].isnull().sum()
            if original_nan_count > 0:
                df[col] = df[col].ffill()
                filled_nan_count = original_nan_count - df[col].isnull().sum() # Compte après ffill
                if filled_nan_count > 0:
                     logger.debug(f"  - Colonne '{col}': {filled_nan_count}/{original_nan_count} NaNs remplis avec ffill.")
                # S'il reste des NaNs (au début du df), on peut utiliser bfill ou une valeur constante (ex: 0 pour volume)
                remaining_nan = df[col].isnull().sum()
                if remaining_nan > 0:
                    if col == 'volume':
                        df[col] = df[col].fillna(0)
                        logger.debug(f"  - Colonne '{col}': {remaining_nan} NaNs restants remplis avec 0.")
                    else:
                        df[col] = df[col].bfill()
                        logger.debug(f"  - Colonne '{col}': {remaining_nan} NaNs restants remplis avec bfill.")

    # Optionnel: Supprimer les lignes où des NaNs critiques persistent (ex: 'close')
    df.dropna(subset=['close'], inplace=True)
    rows_after_nan = len(df)
    if rows_after_nan < initial_rows:
        logger.info(f"  - {initial_rows - rows_after_nan} lignes supprimées car 'close' était NaN et non remplissable.")


    # 2. Gestion des outliers (exemple simple avec IQR sur les rendements journaliers)
    # Cette partie est indicative et pourrait nécessiter une approche plus sophistiquée
    # selon la nature des données et la stratégie de trading.
    df['returns'] = df['close'].pct_change()
    Q1 = df['returns'].quantile(0.25)
    Q3 = df['returns'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df['returns'] < lower_bound) | (df['returns'] > upper_bound)]
    if not outliers.empty:
        logger.info(f"  - {len(outliers)} outliers potentiels détectés sur les rendements (basé sur IQR).")
        # Action sur les outliers: les remplacer (ex: par les bornes) ou les supprimer
        # Exemple simple: Capping (remplacement par les bornes)
        # df['returns'] = np.clip(df['returns'], lower_bound, upper_bound)
        # Ou suppression:
        # df = df[~((df['returns'] < lower_bound) | (df['returns'] > upper_bound))]

    df = df.drop(columns=['returns']) # Supprimer la colonne temporaire

    logger.info(f"Nettoyage terminé - Shape finale: {df.shape}")
    return df

def save_processed_data(df: pd.DataFrame, output_path: str, format: str = 'parquet'):
    """
    Sauvegarde le DataFrame traité dans le dossier spécifié.

    Args:
        df (pd.DataFrame): Le DataFrame à sauvegarder.
        output_path (str): Chemin complet du fichier de sortie (ex: 'data/processed/BTC_USDT_1h_clean.parquet').
        format (str): Format de sortie ('parquet' ou 'csv'). Par défaut 'parquet'.
    """
    try:
        # Créer le répertoire parent s'il n'existe pas
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Répertoire créé: {output_dir}")

        if format == 'parquet':
            # Sauvegarder SANS l'index comme demandé
            df.to_parquet(output_path, index=False)
            logger.info(f"DataFrame sauvegardé avec succès au format Parquet (sans index): {output_path}")
        elif format == 'csv':
            df.to_csv(output_path, index=True)
            logger.info(f"DataFrame sauvegardé avec succès au format CSV: {output_path}")
        else:
            logger.error(f"Format de sauvegarde '{format}' non supporté. Utilisez 'parquet' ou 'csv'.")

    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du DataFrame vers {output_path}: {e}")

# Exemple d'utilisation (à commenter ou supprimer pour l'intégration finale)
if __name__ == '__main__':
    # Créer un fichier CSV brut factice pour le test
    raw_dir = 'data/raw'
    processed_dir = 'data/processed'
    if not os.path.exists(raw_dir): os.makedirs(raw_dir)
    if not os.path.exists(processed_dir): os.makedirs(processed_dir)

    dummy_data = {
        'timestamp': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 01:00:00', '2023-01-01 02:00:00', '2023-01-01 03:00:00']),
        'open': [100, 101, 102, 1000], # Outlier
        'high': [102, 102, 103, 1010],
        'low': [99, 100, 101, 990],
        'close': [101, 101.5, np.nan, 1005], # NaN
        'volume': [1000, 1500, 1200, 5000]
    }
    dummy_df_raw = pd.DataFrame(dummy_data)
    raw_file_path = os.path.join(raw_dir, 'dummy_test_data.csv')
    dummy_df_raw.to_csv(raw_file_path, index=False)
    logger.info(f"Fichier brut factice créé: {raw_file_path}")

    # 1. Charger les données brutes
    df_loaded = load_raw_data(raw_file_path)

    if df_loaded is not None:
        logger.info("\n--- Données chargées ---")
        logger.info(f"\n{df_loaded.head()}")
        # Utiliser logger.info pour info() car elle print par défaut
        # df_loaded.info() # Peut être trop verbeux pour les logs réguliers

        # 2. Nettoyer les données
        df_cleaned = clean_data(df_loaded.copy()) # Utiliser copy() pour éviter SettingWithCopyWarning

        logger.info("\n--- Données nettoyées ---")
        logger.info(f"\n{df_cleaned.head()}")
        # df_cleaned.info()

        # 3. Sauvegarder les données traitées
        processed_file_path_parquet = os.path.join(processed_dir, 'dummy_test_data_clean.parquet')
        processed_file_path_csv = os.path.join(processed_dir, 'dummy_test_data_clean.csv')
        save_processed_data(df_cleaned, processed_file_path_parquet, format='parquet')
        save_processed_data(df_cleaned, processed_file_path_csv, format='csv')

        # Nettoyage des fichiers de test
        # os.remove(raw_file_path)
        # os.remove(processed_file_path_parquet)
        # os.remove(processed_file_path_csv)
        # logger.info("\nFichiers de test nettoyés.")
    else:
        logger.error("Échec du chargement des données brutes.")
