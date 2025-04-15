import pandas as pd
import pandas_ta as ta
import numpy as np
from typing import Tuple, Optional

# Configuration (peut être externalisée dans config.yaml)
DEFAULT_RSI_PERIOD = 14
DEFAULT_SMA_SHORT = 20
DEFAULT_SMA_LONG = 50
DEFAULT_EMA_SHORT = 12
DEFAULT_EMA_LONG = 26
DEFAULT_MACD_SIGNAL = 9
DEFAULT_BBANDS_PERIOD = 20
DEFAULT_BBANDS_STDDEV = 2
DEFAULT_ATR_PERIOD = 14
DEFAULT_STOCH_K = 14
DEFAULT_STOCH_D = 3
DEFAULT_STOCH_SMOOTH_K = 3

def compute_sma(df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
    """Calcule la Moyenne Mobile Simple (SMA)."""
    return ta.sma(df[column], length=period)

def compute_ema(df: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
    """Calcule la Moyenne Mobile Exponentielle (EMA)."""
    return ta.ema(df[column], length=period)

def compute_rsi(df: pd.DataFrame, period: int = DEFAULT_RSI_PERIOD, column: str = 'close') -> pd.Series:
    """Calcule l'Indice de Force Relative (RSI)."""
    return ta.rsi(df[column], length=period)

def compute_macd(df: pd.DataFrame, fast: int = DEFAULT_EMA_SHORT, slow: int = DEFAULT_EMA_LONG, signal: int = DEFAULT_MACD_SIGNAL, column: str = 'close') -> pd.DataFrame:
    """
    Calcule la Convergence/Divergence de Moyenne Mobile (MACD).
    Retourne un DataFrame avec MACD, histogramme (MACDh) et signal (MACDs).
    """
    macd_df = ta.macd(df[column], fast=fast, slow=slow, signal=signal)
    # Renommer les colonnes pour la clarté
    macd_df.columns = ['MACD', 'MACDh', 'MACDs']
    return macd_df[['MACD', 'MACDs', 'MACDh']] # Réorganiser pour correspondre à l'ordre commun

def compute_bollinger_bands(df: pd.DataFrame, period: int = DEFAULT_BBANDS_PERIOD, std_dev: float = DEFAULT_BBANDS_STDDEV, column: str = 'close') -> pd.DataFrame:
    """
    Calcule les Bandes de Bollinger.
    Retourne un DataFrame avec les bandes supérieure (BBU), médiane (BBM) et inférieure (BBL).
    """
    bbands_df = ta.bbands(df[column], length=period, std=std_dev)
    # Renommer les colonnes pour la clarté et l'ordre standard
    bbands_df.columns = ['BBL', 'BBM', 'BBU', 'BBB', 'BBP'] # Lower, Middle, Upper, Bandwidth, Percent
    return bbands_df[['BBU', 'BBM', 'BBL']] # Garder seulement Upper, Middle, Lower

def compute_atr(df: pd.DataFrame, period: int = DEFAULT_ATR_PERIOD, high_col: str = 'high', low_col: str = 'low', close_col: str = 'close') -> pd.Series:
    """Calcule l'Average True Range (ATR)."""
    return ta.atr(df[high_col], df[low_col], df[close_col], length=period)

def compute_stochastics(df: pd.DataFrame, k: int = DEFAULT_STOCH_K, d: int = DEFAULT_STOCH_D, smooth_k: int = DEFAULT_STOCH_SMOOTH_K, high_col: str = 'high', low_col: str = 'low', close_col: str = 'close') -> pd.DataFrame:
    """
    Calcule l'Oscillateur Stochastique.
    Retourne un DataFrame avec %K (STOCHk) et %D (STOCHd).
    """
    stoch_df = ta.stoch(df[high_col], df[low_col], df[close_col], k=k, d=d, smooth_k=smooth_k)
    # Renommer les colonnes pour la clarté
    stoch_df.columns = ['STOCHk', 'STOCHd']
    return stoch_df

def integrate_llm_context(df: pd.DataFrame) -> pd.DataFrame:
    """
    [Placeholder] Intègre les données contextuelles générées par un LLM.
    Cette fonction simulera l'ajout de colonnes pour le résumé et l'embedding.
    Dans une implémentation réelle, elle appellerait une API LLM.
    """
    # Simule l'ajout de colonnes - à remplacer par un appel API réel
    df['llm_context_summary'] = "Placeholder LLM Summary"
    # Simule un embedding (par exemple, un vecteur de petite taille ou un JSON)
    # Pour un vrai embedding, ce serait une liste/array numpy ou stocké différemment
    df['llm_embedding'] = "[0.1, -0.2, 0.3]" # Exemple simplifié en string

    # TODO: Ajouter la logique d'appel à l'API LLM (OpenAI, HuggingFace, etc.)
    # TODO: Gérer la synchronisation temporelle précise entre les données de marché et le contexte LLM.
    print("WARNING: integrate_llm_context is a placeholder and does not call a real LLM API.")
    return df

def apply_feature_pipeline(df: pd.DataFrame, include_llm: bool = False) -> pd.DataFrame:
    """
    Applique le pipeline complet de feature engineering au DataFrame.
    Calcule les indicateurs techniques et intègre (optionnellement) le contexte LLM.
    """
    print("Applying feature engineering pipeline...")

    # Vérifications initiales (exemple)
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Input DataFrame missing required columns: {required_cols}")

    # --- Calcul des Indicateurs Techniques ---
    print("Calculating technical indicators...")
    df['SMA_short'] = compute_sma(df, period=DEFAULT_SMA_SHORT)
    df['SMA_long'] = compute_sma(df, period=DEFAULT_SMA_LONG)
    df['EMA_short'] = compute_ema(df, period=DEFAULT_EMA_SHORT)
    df['EMA_long'] = compute_ema(df, period=DEFAULT_EMA_LONG)
    df['RSI'] = compute_rsi(df)

    macd_df = compute_macd(df)
    df = pd.concat([df, macd_df], axis=1)

    bbands_df = compute_bollinger_bands(df)
    df = pd.concat([df, bbands_df], axis=1)

    df['ATR'] = compute_atr(df)

    stoch_df = compute_stochastics(df)
    df = pd.concat([df, stoch_df], axis=1)

    # Ajouter d'autres indicateurs si nécessaire (ex: ADX, Ichimoku via pandas_ta)
    # df['ADX'] = ta.adx(df['high'], df['low'], df['close'])['ADX_14'] # Exemple ADX

    # --- Intégration du Contexte LLM (Placeholder) ---
    if include_llm:
        print("Integrating LLM context (placeholder)...")
        df = integrate_llm_context(df)

    # --- Nettoyage final (optionnel) ---
    # Supprimer les lignes avec NaN introduites par les fenêtres glissantes
    initial_rows = len(df)
    df.dropna(inplace=True)
    print(f"Removed {initial_rows - len(df)} rows with NaN values after feature calculation.")

    print("Feature engineering pipeline completed.")
    # TODO: Vérifier la conformité avec le schéma final de 38 colonnes (ajouter/supprimer si besoin)
    # Pour l'instant, on retourne le df avec les colonnes ajoutées.
    return df

# Exemple d'utilisation (peut être mis dans un script de test ou notebook)
if __name__ == '__main__':
    # Créer un DataFrame d'exemple
    data = {
        'timestamp': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:15:00', '2023-01-01 00:30:00', '2023-01-01 00:45:00', '2023-01-01 01:00:00'] * 10), # Assez de données pour les calculs
        'open': np.random.rand(50) * 100 + 1000,
        'high': np.random.rand(50) * 5 + 1005,
        'low': 1000 - np.random.rand(50) * 5,
        'close': np.random.rand(50) * 10 + 1000,
        'volume': np.random.rand(50) * 1000 + 100
    }
    sample_df = pd.DataFrame(data)
    sample_df['high'] = sample_df[['open', 'close']].max(axis=1) + np.random.rand(50) * 2
    sample_df['low'] = sample_df[['open', 'close']].min(axis=1) - np.random.rand(50) * 2
    sample_df.set_index('timestamp', inplace=True)

    print("Original DataFrame:")
    print(sample_df.head())

    # Appliquer le pipeline
    features_df = apply_feature_pipeline(sample_df.copy(), include_llm=True)

    print("\nDataFrame with Features:")
    print(features_df.head())
    print("\nColumns added:")
    print(features_df.columns)
