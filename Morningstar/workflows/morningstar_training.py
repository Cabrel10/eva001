import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from Morningstar.model.architecture.morningstar_model import MorningstarTradingModel
from Morningstar.configs.morningstar_config import MorningstarConfig, TradingStyle
from Morningstar.utils.data_manager import ExchangeDataManager
import pandas as pd
import numpy as np
import logging

class MorningstarTrainingWorkflow:
    """Workflow complet pour l'entraînement du modèle Morningstar"""
    
    def __init__(self, config: MorningstarConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def prepare_data(self) -> tuple:
        """Charge et prépare les données pour l'entraînement"""
        # Chargement des données
        data = pd.read_parquet(self.config.DATA_PATH)
        
        # Vérification et sélection des colonnes disponibles
        available_cols = data.columns.tolist()
        features = [col for col in self.config.base_columns if col in available_cols]
        features += [col for col in self.config.technical_columns if col in available_cols]
        features += [col for col in self.config.social_columns if col in available_cols]
        
        self.logger.info(f"Colonnes sélectionnées: {features}")
        if not features:
            raise ValueError("Aucune colonne valide trouvée - vérifiez la configuration")
            
        # Gestion des NaN
        data = data[features]
        
        # Remplissage des NaN techniques (seulement si <5% de NaN)
        tech_cols = [col for col in features if col in self.config.technical_columns]
        for col in tech_cols:
            if data[col].isna().mean() < 0.05:  # Moins de 5% de NaN
                data[col] = data[col].fillna(0)
            else:
                features.remove(col)  # Supprime la colonne si trop de NaN
                self.logger.warning(f"Colonne {col} supprimée (trop de valeurs manquantes)")
        
        # Suppression des lignes avec NaN dans les colonnes de base
        base_cols = [col for col in features if col in self.config.base_columns]
        data = data.dropna(subset=base_cols)
        
        self.logger.info(f"Données finales: {len(data)} lignes, {len(features)} colonnes")
        
        # Normalisation par groupe
        data[self.config.base_columns] = (data[self.config.base_columns] - data[self.config.base_columns].mean()) / data[self.config.base_columns].std()
        data[self.config.technical_columns] = (data[self.config.technical_columns] - data[self.config.technical_columns].mean()) / data[self.config.technical_columns].std() 
        data[self.config.social_columns] = (data[self.config.social_columns] - data[self.config.social_columns].mean()) / data[self.config.social_columns].std()
        
        # Conversion en séquences
        sequences = []
        for i in range(len(data) - self.config.time_window):
            seq = data.iloc[i:i+self.config.time_window].values
            sequences.append(seq)
            
        sequences = np.array(sequences)
        return self._split_data(sequences)
    
    def _split_data(self, data: np.ndarray, test_size=0.2):
        """Division des données en train/test"""
        split_idx = int(len(data) * (1 - test_size))
        return data[:split_idx], data[split_idx:]
    
    def train_model(self, train_data: np.ndarray):
        """Entraînement du modèle avec les nouvelles données"""
        model = MorningstarTradingModel(
            input_shape=train_data.shape[1:],
            num_classes=3
        )
        
        model.compile_model(self.config.learning_rate)
        
        history = model.model.fit(
            x=train_data,
            y=self._generate_labels(train_data),
            batch_size=self.config.batch_size,
            epochs=self.config.epochs,
            validation_split=0.1,
            callbacks=[
                ModelCheckpoint('best_model.h5', save_best_only=True),
                EarlyStopping(patience=10)
            ]
        )
        
        return model, history
    
    def _generate_labels(self, data: np.ndarray) -> np.ndarray:
        """Génération des labels en fonction du style de trading"""
        price_changes = data[:, -1, 0] - data[:, 0, 0]  # close[-1] - close[0]
        
        if self.config.trading_style == TradingStyle.SCALPING:
            thresholds = (0.002, -0.002)
        elif self.config.trading_style == TradingStyle.DAY:
            thresholds = (0.01, -0.01)
        else:  # SWING
            thresholds = (0.03, -0.03)
            
        labels = np.zeros((len(data), 3))
        labels[price_changes > thresholds[0], 0] = 1  # Achat
        labels[price_changes < thresholds[1], 1] = 1  # Vente
        labels[(price_changes >= thresholds[1]) & (price_changes <= thresholds[0]), 2] = 1  # Neutre
        
        return labels
