import tensorflow as tf
import numpy as np
import logging
from typing import Dict, Tuple, Union
from .enhanced_hybrid_model import MorningstarHybridModel

class MorningstarModel:
    """Wrapper pour intégrer le modèle hybride dans le workflow de trading."""
    
    def __init__(self, model_config: Dict = None):
        """
        Initialise le wrapper du modèle.
        
        Args:
            model_config: Configuration du modèle (peut être chargée depuis un fichier YAML)
        """
        self.logger = logging.getLogger('MorningstarModel')
        self.model = None
        self.config = model_config or {
            'num_technical_features': 38,
            'llm_embedding_dim': 768,
            'num_signal_classes': 5,
            'num_volatility_regimes': 3,
            'num_market_regimes': 4
        }
        
    def initialize_model(self) -> None:
        """Initialise l'architecture du modèle."""
        try:
            builder = MorningstarHybridModel(**self.config)
            self.model = builder.build_model()
            self.logger.info("Modèle initialisé avec succès")
        except Exception as e:
            self.logger.error(f"Erreur lors de l'initialisation du modèle: {str(e)}")
            raise
            
    def predict(self, 
               technical_data: np.ndarray,
               llm_embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Effectue une prédiction complète.
        
        Args:
            technical_data: Données techniques (shape: [batch_size, 38])
            llm_embeddings: Embeddings LLM (shape: [batch_size, 768])
            
        Returns:
            Dictionnaire contenant toutes les prédictions
        """
        if self.model is None:
            raise ValueError("Le modèle n'a pas été initialisé")
            
        try:
            # Vérification simple des shapes d'entrée (peut être plus robuste)
            if technical_data.shape[1] != self.config['num_technical_features']:
                raise ValueError(f"Shape technique incorrecte: attendu {self.config['num_technical_features']}, reçu {technical_data.shape[1]}")
            if llm_embeddings.shape[1] != self.config['llm_embedding_dim']:
                 raise ValueError(f"Shape LLM incorrecte: attendu {self.config['llm_embedding_dim']}, reçu {llm_embeddings.shape[1]}")

            predictions_list = self.model.predict([technical_data, llm_embeddings])
            
            # Vérification de base du nombre de sorties
            if len(predictions_list) != 5:
                 raise ValueError(f"Nombre de sorties inattendu: attendu 5, reçu {len(predictions_list)}")

            # Structuration de la sortie comme demandé
            result_dict = {
                'signal': predictions_list[0],
                'volatility_quantiles': predictions_list[1],
                'volatility_regime': predictions_list[2],
                'market_regime': predictions_list[3],
                'sl_tp': predictions_list[4]
            }
            
            # Vérification des shapes de sortie (exemple pour le signal)
            expected_signal_shape = (technical_data.shape[0], self.config['num_signal_classes'])
            if result_dict['signal'].shape != expected_signal_shape:
                 raise ValueError(f"Shape de sortie signal incorrecte: attendu {expected_signal_shape}, reçu {result_dict['signal'].shape}")
                 
            # Ajouter d'autres vérifications de shape si nécessaire...

            self.logger.info(f"Prédiction réussie pour {technical_data.shape[0]} échantillons.")
            return {
                'status': 'success',
                'result': result_dict
            }
            
        except Exception as e:
            self.logger.error(f"Erreur lors de la prédiction: {str(e)}")
            # Retourne un dictionnaire d'erreur au lieu de lever l'exception directement
            return {
                'status': 'error',
                'message': str(e)
            }
            
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
