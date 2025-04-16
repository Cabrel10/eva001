import logging
import numpy as np
from typing import Dict, Any
from model.architecture.morningstar_model import MorningstarModel
from utils.api_manager import APIManager
from utils.log_config import setup_logging

class TradingWorkflow:
    """Workflow principal pour exécuter la stratégie de trading basée sur le modèle Morningstar."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le workflow de trading.
        
        Args:
            config: Configuration du workflow (chargée depuis config.yaml)
        """
        setup_logging()
        self.logger = logging.getLogger('TradingWorkflow')
        self.config = config
        self.model = MorningstarModel(config.get('model'))
        self.api = APIManager(config.get('api'))
        
        # Charger le modèle entraîné
        model_path = config.get('model', {}).get('weights_path')
        if model_path:
            self.model.load_weights(model_path)
            self.model.prepare_for_inference()
        
    def process_market_data(self, raw_data: Dict) -> Dict[str, np.ndarray]:
        """
        Prétraite les données de marché pour le modèle.
        
        Args:
            raw_data: Données brutes de l'API de marché
            
        Returns:
            Données prétraitées sous forme de numpy arrays
        """
        # TODO: Implémenter le prétraitement des données
        technical_data = np.array(raw_data['technical'])
        llm_embeddings = np.array(raw_data['sentiment_embeddings'])
        return {
            'technical': technical_data,
            'llm_embeddings': llm_embeddings
        }
        
    def execute_strategy(self, market_data: Dict) -> Dict:
        """
        Exécute la stratégie de trading complète.
        
        Args:
            market_data: Données de marché brutes (doit contenir 'technical' et 'sentiment_embeddings')
            
        Returns:
            Décisions de trading et métadonnées
        """
        try:
            # Validation des données d'entrée
            if not all(key in market_data for key in ['technical', 'sentiment_embeddings']):
                raise ValueError("Les données de marché doivent contenir 'technical' et 'sentiment_embeddings'")
                
            # Prétraitement des données
            processed_data = self.process_market_data(market_data)
            
            # Prédiction du modèle
            # Reshape les données pour le modèle (batch_size=1)
            technical_data = processed_data['technical'].reshape(1, -1)
            llm_embeddings = processed_data['llm_embeddings'].reshape(1, -1)
            
            # Obtenir les prédictions du modèle
            raw_predictions = self.model.predict(
                technical_data=technical_data,
                llm_embeddings=llm_embeddings
            )
            
            # Formater les prédictions pour garantir la bonne structure
            predictions = {
                'signal': np.array(raw_predictions['signal']).reshape(-1, 5),
                'volatility_quantiles': np.array(raw_predictions['volatility_quantiles']).reshape(-1, 3),
                'volatility_regime': np.array(raw_predictions['volatility_regime']).reshape(-1, 3),
                'market_regime': np.array(raw_predictions['market_regime']).reshape(-1, 4),
                'sl_tp': np.array(raw_predictions['sl_tp']).reshape(-1, 2)
            }
            
            # Conversion des prédictions en décisions
            trading_decisions = self._generate_trading_signals(predictions)
            
            return {
                'decisions': trading_decisions,
                'predictions': predictions,
                'status': 'success'
            }
            
        except Exception as e:
            self.logger.error(f"Erreur dans execute_strategy: {str(e)}")
            return {
                'status': 'error',
                'error': str(e)
            }
    
    def _generate_trading_signals(self, predictions: Dict) -> Dict:
        """
        Convertit les prédictions du modèle en signaux de trading exploitables.
        
        Args:
            predictions: Sorties du modèle Morningstar
            
        Returns:
            Dictionnaire de signaux de trading
        """
        # TODO: Implémenter la logique de conversion des prédictions en signaux
        signal_map = {
            0: 'STRONG_BUY',
            1: 'BUY', 
            2: 'NEUTRAL',
            3: 'SELL',
            4: 'STRONG_SELL'
        }
        
        # Assurer que les prédictions ont la bonne forme (batch_size=1)
        signal_pred = predictions['signal'][0]
        volatility_regime_pred = predictions['volatility_regime'][0]
        market_regime_pred = predictions['market_regime'][0]
        sl_tp_pred = predictions['sl_tp'][0] # Shape (2,) attendue

        return {
            'signal': signal_map[np.argmax(signal_pred)],
            'volatility_regime': volatility_regime_pred, # Ou np.argmax si c'est des probas
            'market_regime': market_regime_pred,       # Ou np.argmax si c'est des probas
            'stop_loss': sl_tp_pred[0],
            'take_profit': sl_tp_pred[1]
        }
        
    def run(self):
        """Point d'entrée principal pour exécuter le workflow en continu."""
        self.logger.info("Démarrage du workflow de trading")
        
        try:
            # Récupérer les données de marché
            market_data = self.api.get_market_data()
            
            # Exécuter la stratégie
            result = self.execute_strategy(market_data)
            
            if result['status'] == 'success':
                # Exécuter les ordres
                self.api.execute_orders(result['decisions'])
                return True
            return False
                
        except KeyboardInterrupt:
            self.logger.info("Arrêt demandé par l'utilisateur")
            return False
        except Exception as e:
            self.logger.error(f"Erreur critique dans le workflow: {str(e)}")
            # TODO: Implémenter un mécanisme de reprise sûr
            return False
