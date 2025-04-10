import pytest
from unittest.mock import patch
from Morningstar.configs.tf_config import TFConfig
import tensorflow as tf
import os

class TestTFConfig:
    def test_config_values(self):
        """Vérifie les valeurs de configuration de base"""
        assert TFConfig.BATCH_SIZE == 64
        assert TFConfig.EPOCHS == 100
        assert TFConfig.LEARNING_RATE == 0.001
        assert TFConfig.VALIDATION_SPLIT == 0.2

    def test_optimized_config(self):
        """Test la configuration optimisée"""
        config = TFConfig.get_optimized_config()
        assert config['batch_size'] == 64
        assert isinstance(config['optimizer'], tf.keras.optimizers.Adam)
        assert config['loss'] == 'mse'
        assert 'mae' in config['metrics']

    @patch('tensorflow.config.experimental.set_memory_growth')
    @patch('tensorflow.config.list_physical_devices')
    def test_setup_gpu(self, mock_list_gpu, mock_set_growth):
        """Test la configuration GPU"""
        # Test avec GPU disponible
        mock_list_gpu.return_value = [tf.config.PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
        TFConfig.setup_gpu()
        # Vérifie que set_memory_growth a été appelé car un GPU est listé
        mock_set_growth.assert_called_once() 
        mock_set_growth.reset_mock() # Reset mock for the next part

        # Test sans GPU
        mock_list_gpu.return_value = []
        TFConfig.setup_gpu() 
        # Vérifie que set_memory_growth n'a PAS été appelé
        mock_set_growth.assert_not_called() 

    # @patch.dict(os.environ, {
    #     'TWITTER_API_KEY': 'test_key',
    #     'REDDIT_CLIENT_ID': 'test_id'
    # })
    # def test_api_keys(self):
    #     """Test le chargement des clés API (Commenté car TFConfig ne charge pas les clés)"""
    #     # Cette fonctionnalité semble manquer dans TFConfig, les clés sont dans social_config.py
    #     # assert TFConfig.TWITTER_KEYS['api_key'] == 'test_key' 
    #     # assert TFConfig.REDDIT_KEYS['client_id'] == 'test_id'
    #     # assert TFConfig.TWITTER_KEYS['api_secret'] == ''
    #     pass # Test désactivé pour le moment
