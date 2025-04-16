import unittest
import numpy as np
from unittest.mock import MagicMock, patch
from workflows.trading_workflow import TradingWorkflow

class TestTradingWorkflow(unittest.TestCase):
    """Tests d'intégration pour le workflow de trading."""
    
    def setUp(self):
        """Initialisation des tests avec une configuration mock."""
        self.config = {
            'model': {
                'weights_path': 'tests/fixtures/mock_model.h5',
                'num_technical_features': 38,
                'llm_embedding_dim': 768
            },
            'api': {
                'base_url': 'http://mock-api',
                'timeout': 10
            }
        }
        
        # Mock du modèle et de l'API
        self.mock_model = MagicMock()
        self.mock_api = MagicMock()
        
        # Configuration des valeurs de retour mock
        self.mock_predictions = {
            'signal': np.array([[0.8, 0.1, 0.05, 0.03, 0.02]]),
            'volatility_quantiles': np.array([[0.1, 0.5, 0.9]]),
            'volatility_regime': np.array([[0, 1, 0]]),
            'market_regime': np.array([[0, 0, 1, 0]]),
            'sl_tp': np.array([[0.95, 1.05]])
        }
        
    @patch('workflows.trading_workflow.MorningstarModel')
    @patch('workflows.trading_workflow.APIManager')
    def test_workflow_initialization(self, mock_api, mock_model):
        """Teste l'initialisation correcte du workflow."""
        mock_model.return_value = self.mock_model
        mock_api.return_value = self.mock_api
        
        workflow = TradingWorkflow(self.config)
        
        # Vérifications
        mock_model.assert_called_once()
        mock_api.assert_called_once()
        self.mock_model.load_weights.assert_called_with('tests/fixtures/mock_model.h5')
        
    @patch('workflows.trading_workflow.MorningstarModel')
    @patch('workflows.trading_workflow.APIManager')
    def test_execute_strategy(self, mock_api, mock_model):
        """Teste l'exécution complète de la stratégie."""
        # Configuration des mocks
        mock_model.return_value = self.mock_model
        mock_api.return_value = self.mock_api
        self.mock_model.predict.return_value = self.mock_predictions
        
        # Données de marché mock
        market_data = {
            'technical': np.random.rand(38),  # format attendu par process_market_data
            'sentiment_embeddings': np.random.rand(768)  # format attendu
        }
        self.mock_api.get_market_data.return_value = market_data
        
        # Exécution
        workflow = TradingWorkflow(self.config)
        result = workflow.execute_strategy(market_data)
        
        # Vérifications
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['decisions']['signal'], 'STRONG_BUY')
        self.mock_model.predict.assert_called_once()
        
    @patch('workflows.trading_workflow.MorningstarModel')
    @patch('workflows.trading_workflow.APIManager')
    def test_error_handling(self, mock_api, mock_model):
        """Teste la gestion des erreurs dans le workflow."""
        # Configuration des mocks pour simuler une erreur
        mock_model.return_value = self.mock_model
        mock_api.return_value = self.mock_api
        self.mock_model.predict.side_effect = Exception("Mock error")
        
        # Exécution
        workflow = TradingWorkflow(self.config)
        market_data = {'technical': [], 'sentiment_embeddings': []}
        result = workflow.execute_strategy(market_data)
        
        # Vérifications
        self.assertEqual(result['status'], 'error')
        self.assertIn('Mock error', result['error'])

if __name__ == '__main__':
    unittest.main()
