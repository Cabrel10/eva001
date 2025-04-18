import tensorflow as tf
import numpy as np
import logging
from typing import Dict, Tuple, Union
from .enhanced_hybrid_model import build_enhanced_hybrid_model

class MorningstarModel:
    """Wrapper pour intégrer le modèle hybride dans le workflow de trading."""
    
    def __init__(self, model_config: Dict = None):
        """
        Initialise le wrapper du modèle.
        
        Args:
            model_config: Configuration du modèle (peut être chargée depuis un fichier YAML)
        """
        self.logger = logging.getLogger('MorningstarModel')
        self.config = model_config or {
            'num_technical_features': 38,
            'llm_embedding_dim': 768,
            'num_signal_classes': 5,
            'num_volatility_regimes': 3,
            'num_market_regimes': 4
        }
        self.model = None
        self.initialize_model()  # Initialisation immédiate
        
    def initialize_model(self) -> None:
        """Initialise l'architecture du modèle."""
        try:
            # Construction directe via la fonction wrapper
            self.model = build_enhanced_hybrid_model(
                input_shape=(self.config['num_technical_features'],),
                num_trading_classes=self.config['num_signal_classes'],
                num_regime_classes=self.config['num_market_regimes']
            )
            self.logger.info("Modèle initialisé avec succès")
            self.logger.info(f"Architecture du modèle:\n{self.get_model_summary()}")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation du modèle: {str(e)}")
            raise RuntimeError(f"Échec de l'initialisation: {str(e)}") from e
            
    def predict(self, 
               technical_data: np.ndarray,
               llm_embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Effectue une prédiction complète.
        
        Args:
            technical_data: Données techniques (shape: [batch_size, 38])
            llm_embeddings: Embeddings LLM (shape: [batch_size, 768])
            
        Returns:
            Dictionnaire contenant toutes les prédictions avec les clés:
            - 'signal' (shape: [batch_size, 5])
            - 'volatility_quantiles' (shape: [batch_size, 3]) 
            - 'volatility_regime' (shape: [batch_size, 3])
            - 'market_regime' (shape: [batch_size, num_regime_classes])
            - 'sl_tp' (shape: [batch_size, 2])
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été initialisé")
            
        try:
            # Validation stricte des shapes d'entrée
            if technical_data.shape[1] != self.config['num_technical_features']:
                raise ValueError(f"Shape technique incorrecte: attendu {self.config['num_technical_features']}, reçu {technical_data.shape[1]}")
            if llm_embeddings.shape[1] != self.config['llm_embedding_dim']:
                raise ValueError(f"Shape LLM incorrecte: attendu {self.config['llm_embedding_dim']}, reçu {llm_embeddings.shape[1]}")

            # Prédiction avec les deux entrées
            predictions = self.model.predict([technical_data, llm_embeddings])
            
            # Mapping des sorties selon les noms définis dans le modèle
            self.logger.info(f"Prédiction réussie pour {technical_data.shape[0]} échantillons")
            return {
                'signal': predictions[0],
                'volatility_quantiles': predictions[1],
                'volatility_regime': predictions[2], 
                'market_regime': predictions[3],
                'sl_tp': predictions[4]
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction: {str(e)}")
            raise  # Relance l'exception originale
            
    def save_weights(self, filepath: str) -> None:
        """Sauvegarde les poids du modèle."""
        if self.model is None:
            raise ValueError("Le modèle n'a pas été initialisé")
        self.model.save_weights(filepath)
        self.logger.info(f"Poids du modèle sauvegardés dans {filepath}")
        
    def load_weights(self, filepath: str) -> None:
        """Charge les poids du modèle."""
        if self.model is None:
            self.initialize_model()
        self.model.load_weights(filepath)
        self.logger.info(f"Poids du modèle chargés depuis {filepath}")
        
    def get_model_summary(self) -> str:
        """Retourne un résumé textuel de l'architecture du modèle."""
        if self.model is None:
            return "Modèle non initialisé"
            
        string_list = []
        self.model.summary(print_fn=lambda x: string_list.append(x))
        return "\n".join(string_list)
        
    def prepare_for_inference(self) -> None:
        """Prépare le modèle pour l'inférence (optimisations)."""
        if self.model is None:
            raise ValueError("Le modèle n'a pas été initialisé")
            
        # Optimisations pour l'inférence
        self.model.compile(optimizer='adam')  # Recompilation légère
        self.logger.info("Modèle optimisé pour l'inférence")
