import unittest
from unittest.mock import patch, MagicMock, ANY
from utils.llm_integration import LLMIntegration, run_in_llm_env # Importer run_in_llm_env pour le mocker
import json
from pathlib import Path
import shutil # Pour nettoyer le répertoire de cache

# Désactiver Redis pour les tests unitaires pour éviter les dépendances externes
@patch('utils.llm_integration.redis_available', False)
class TestLLMIntegration(unittest.TestCase):

    def setUp(self):
        # Pas besoin de mocker genai ici car on mockera run_in_llm_env
        
        # Create test instance
        self.llm = LLMIntegration()
        
        # Setup test cache dir
        self.test_cache_dir = Path("tests/test_cache")
        self.llm.cache_dir = self.test_cache_dir
        # Nettoyer et recréer le cache avant chaque test
        if self.test_cache_dir.exists():
            shutil.rmtree(self.test_cache_dir)
        self.test_cache_dir.mkdir(exist_ok=True)
        
        # Assurer que les services sont considérés comme disponibles au début
        self.llm.google_available = True 
        self.llm.openrouter_available = True # Pas utilisé directement mais bon à initialiser

    def tearDown(self):
        # Clean up test cache directory after tests
        if self.test_cache_dir.exists():
            shutil.rmtree(self.test_cache_dir)

    @patch('utils.llm_integration.run_in_llm_env') # Mocker la fonction qui appelle le sous-processus
    def test_cache_creation(self, mock_run_in_llm_env):
        """Test that cache is created and works properly"""
        test_text = "test cache content"
        test_embeddings = [0.1, 0.2, 0.3]
        # Simuler la sortie JSON du sous-processus
        mock_run_in_llm_env.return_value = json.dumps(test_embeddings) 
        
        # First call - should call mocked run_in_llm_env and cache
        result = self.llm.get_embeddings(test_text)
        self.assertEqual(result, test_embeddings)
        mock_run_in_llm_env.assert_called_once() # Vérifier que le mock a été appelé
        
        # Reset mock pour le deuxième appel
        mock_run_in_llm_env.reset_mock() 
        
        # Second call - should use file cache, not call run_in_llm_env again
        result_cached = self.llm.get_embeddings(test_text)
        self.assertEqual(result_cached, test_embeddings)
        mock_run_in_llm_env.assert_not_called() # Vérifier que le mock n'a PAS été appelé
        
        # Verify cache file exists
        cache_file = self.llm.get_cache_path(test_text)
        self.assertTrue(cache_file.exists())
        
        # Verify cache content
        with open(cache_file) as f:
            cached_data = json.load(f)
        self.assertEqual(cached_data, test_embeddings)

    # Pas besoin de mocker requests.post car on mock run_in_llm_env
    @patch('utils.llm_integration.run_in_llm_env') 
    def test_fallback_behavior(self, mock_run_in_llm_env):
        """Test behavior when Google fails (mocked) - should still work if run_in_llm_env is mocked"""
        # Simuler l'indisponibilité de Google (même si run_in_llm_env est mocké, pour tester la logique interne)
        self.llm.google_available = False 
        
        # Définir la sortie simulée pour run_in_llm_env (même si Google est "indisponible")
        # Dans la vraie fonction, run_in_llm_env ne serait pas appelé si google_available est False.
        # Mais ici, on teste surtout que get_embeddings retourne None si google_available est False
        # ET que run_in_llm_env n'est pas appelé.
        # Si on voulait tester un fallback vers OpenRouter via run_in_llm_env, il faudrait adapter la logique.
        # Pour l'instant, on vérifie juste que ça retourne None si Google est down.
        mock_run_in_llm_env.return_value = json.dumps([0.4, 0.5, 0.6]) # Valeur non utilisée
        
        result = self.llm.get_embeddings("test fallback")
        # Comme google_available est False et qu'il n'y a pas de logique OpenRouter dans get_embeddings,
        # cela devrait retourner None et ne pas appeler run_in_llm_env.
        self.assertIsNone(result) 
        mock_run_in_llm_env.assert_not_called()

    @patch('utils.llm_integration.run_in_llm_env')
    def test_no_service_available(self, mock_run_in_llm_env):
        """Test behavior when no LLM service is available"""
        self.llm.google_available = False
        self.llm.openrouter_available = False # Bien que non utilisé dans get_embeddings
        
        # Configurer le mock au cas où il serait appelé (ne devrait pas l'être)
        mock_run_in_llm_env.return_value = json.dumps([0.7, 0.8, 0.9])
        
        result = self.llm.get_embeddings("test no service")
        self.assertIsNone(result)
        mock_run_in_llm_env.assert_not_called() # Ne devrait pas appeler le sous-processus

    @patch('utils.llm_integration.run_in_llm_env')
    def test_llm_analysis_cache(self, mock_run_in_llm_env):
        """Test caching for get_llm_analysis"""
        test_text = "analyze this text"
        test_analysis = {"sentiment": "positive", "confidence": 0.8}
        # Simuler la sortie JSON du sous-processus pour l'analyse
        mock_run_in_llm_env.return_value = json.dumps(test_analysis)

        # First call
        result = self.llm.get_llm_analysis(test_text)
        self.assertEqual(result, test_analysis)
        mock_run_in_llm_env.assert_called_once()

        # Second call (should use cache)
        mock_run_in_llm_env.reset_mock()
        result_cached = self.llm.get_llm_analysis(test_text)
        self.assertEqual(result_cached, test_analysis)
        mock_run_in_llm_env.assert_not_called()

        # Verify cache file
        cache_file = self.llm.get_cache_path(f"analysis_{hashlib.md5(test_text.encode()).hexdigest()}")
        self.assertTrue(cache_file.exists())
        with open(cache_file) as f:
            cached_data = json.load(f)
        self.assertEqual(cached_data, test_analysis)
        
    @patch('utils.llm_integration.run_in_llm_env')
    def test_llm_analysis_parsing_error(self, mock_run_in_llm_env):
        """Test handling of invalid JSON from analysis subprocess"""
        test_text = "bad json text"
        invalid_json_output = "```json\n{\"sentiment\": \"neutral\"" # JSON incomplet
        mock_run_in_llm_env.return_value = invalid_json_output
        
        result = self.llm.get_llm_analysis(test_text)
        
        self.assertIn("error", result)
        self.assertEqual(result["error"], "Failed to parse LLM response")
        self.assertEqual(result["raw_response"], invalid_json_output)
        mock_run_in_llm_env.assert_called_once()

# Ajouter l'import de hashlib manquant
import hashlib

if __name__ == '__main__':
    unittest.main()
