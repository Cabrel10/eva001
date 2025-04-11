from dataclasses import dataclass
from enum import Enum

class Timeframe(Enum):
    M1 = '1m'
    M15 = '15m' 
    H1 = '1h'
    H4 = '4h'
    D1 = '1d'

class TradingStyle(Enum):
    SCALPING = 'scalping'
    DAY = 'day'
    SWING = 'swing'
    POSITION = 'position'

class RiskLevel(Enum):
    PRUDENT = 'prudent'
    MODERE = 'modere'
    AGGRESSIF = 'aggressif'

@dataclass 
class MorningstarConfig:
    # Paramètres temporels
    timeframe: Timeframe = Timeframe.H1
    trading_style: TradingStyle = TradingStyle.SWING
    
    # Colonnes de données
    base_columns = ['open', 'high', 'low', 'close', 'volume']
    technical_columns = [
        'rsi', 'macd', 'macd_signal', 'macd_hist', 
        'bb_upper', 'bb_middle', 'bb_lower', 
        'volume_ma', 'volume_anomaly'
    ]
    # Features sociales (désactivées car non présentes dans le dataset)
    social_columns = []
    # Features de corrélation (désactivées car non présentes dans le dataset) 
    correlation_columns = []
    
    # Paramètres de risque
    risk_level: RiskLevel = RiskLevel.MODERE
    
    # Architecture du modèle
    cnn_filters: int = 64
    lstm_units: int = 64
    dense_units: int = 64
    
    # Hyperparamètres d'entraînement
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    
    def get_model_config(self):
        """Retourne la configuration spécifique au modèle"""
        return {
            'cnn_filters': self.cnn_filters,
            'lstm_units': self.lstm_units,
            'dense_units': self.dense_units
        }
