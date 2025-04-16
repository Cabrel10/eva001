import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Concatenate, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from typing import Tuple

class MorningstarHybridModel:
    """Modèle hybride multi-tâches pour le trading crypto avec fusion multimodale."""
    
    def __init__(self, 
                 num_technical_features: int = 38,
                 llm_embedding_dim: int = 768,
                 num_signal_classes: int = 5,
                 num_volatility_regimes: int = 3,
                 num_market_regimes: int = 4):
        """
        Initialise l'architecture du modèle.
        
        Args:
            num_technical_features: Nombre de features techniques en entrée
            llm_embedding_dim: Dimension des embeddings LLM
            num_signal_classes: Nombre de classes pour le signal de trading
            num_volatility_regimes: Nombre de régimes de volatilité
            num_market_regimes: Nombre de régimes de marché
        """
        self.num_technical_features = num_technical_features
        self.llm_embedding_dim = llm_embedding_dim
        self.num_signal_classes = num_signal_classes
        self.num_volatility_regimes = num_volatility_regimes
        self.num_market_regimes = num_market_regimes
        
    def build_model(self) -> Model:
        """Construit l'architecture complète du modèle."""
        
        # Entrées
        technical_input = Input(shape=(self.num_technical_features,), name='technical_input')
        llm_input = Input(shape=(self.llm_embedding_dim,), name='llm_input')
        
        # Fusion multimodale
        merged = self._build_fusion_module(technical_input, llm_input)
        
        # Têtes de prédiction
        signal_output = self._build_signal_head(merged)
        # La tête de volatilité retourne deux sorties : quantiles et régime
        volatility_quantiles_output, volatility_regime_output = self._build_volatility_head(merged)
        market_regime_output = self._build_market_regime_head(merged)
        sl_tp_output = self._build_sl_tp_head(merged)
        
        # Modèle complet
        model = Model(
            inputs=[technical_input, llm_input],
            outputs=[
                signal_output, 
                volatility_quantiles_output, 
                volatility_regime_output, 
                market_regime_output, 
                sl_tp_output
            ],
            name='morningstar_hybrid_model'
        )
        
        return model
    
    def _build_fusion_module(self, 
                           technical_input: tf.Tensor,
                           llm_input: tf.Tensor) -> tf.Tensor:
        """Module de fusion des données techniques et embeddings LLM."""
        # Traitement des features techniques
        tech_processed = Dense(128, activation='relu')(technical_input)
        tech_processed = BatchNormalization()(tech_processed)
        tech_processed = Dropout(0.3)(tech_processed)
        
        # Traitement des embeddings LLM
        llm_processed = Dense(256, activation='relu')(llm_input)
        llm_processed = BatchNormalization()(llm_processed)
        llm_processed = Dropout(0.3)(llm_processed)
        
        # Fusion
        merged = Concatenate()([tech_processed, llm_processed])
        merged = Dense(512, activation='relu')(merged)
        merged = BatchNormalization()(merged)
        merged = Dropout(0.4)(merged)
        
        return merged
    
    def _build_signal_head(self, merged: tf.Tensor) -> tf.Tensor:
        """Tête de prédiction pour le signal de trading."""
        x = Dense(256, activation='relu')(merged)
        x = Dropout(0.3)(x)
        return Dense(self.num_signal_classes, activation='softmax', name='signal_output')(x)
    
    def _build_volatility_head(self, merged: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        """Tête de prédiction pour la volatilité (quantiles + classification)."""
        # Branche quantiles
        quantile_output = Dense(3, activation='linear', name='volatility_quantiles')(merged)
        
        # Branche classification
        x = Dense(128, activation='relu')(merged)
        regime_output = Dense(self.num_volatility_regimes, activation='softmax', 
                            name='volatility_regime')(x)
        
        return quantile_output, regime_output
    
    def _build_market_regime_head(self, merged: tf.Tensor) -> tf.Tensor:
        """Tête de prédiction pour les régimes de marché."""
        x = Dense(128, activation='relu')(merged)
        x = Dropout(0.3)(x)
        return Dense(self.num_market_regimes, activation='softmax', name='market_regime_output')(x)
    
    def _build_sl_tp_head(self, merged: tf.Tensor) -> tf.Tensor:
        """Tête de prédiction pour les niveaux SL/TP (placeholder pour RL)."""
        # Placeholder - sera remplacé par l'implémentation RL
        x = Dense(64, activation='relu')(merged)
        return Dense(2, activation='linear', name='sl_tp_output')(x)
