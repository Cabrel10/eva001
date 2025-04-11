import pytest
import numpy as np
import tensorflow as tf
from Morningstar.model.architecture.morningstar_model import MorningstarTradingModel

class TestMorningstarModel:
    @pytest.fixture
    def sample_model(self):
        """Fixture pour initialiser le modèle de test"""
        return MorningstarTradingModel(input_shape=(60, 5), num_classes=3)

    def test_model_initialization(self, sample_model):
        """Test l'initialisation correcte du modèle"""
        assert sample_model.model is not None
        assert len(sample_model.model.layers) == 6  # Input + 3 modules + concat + output
        assert sample_model.input_shape == (60, 5)
        assert sample_model.num_classes == 3

    def test_prediction_shape(self, sample_model):
        """Test que les prédictions ont la bonne forme"""
        test_data = np.random.rand(10, 60, 5)
        predictions = sample_model.model.predict(test_data)
        assert predictions.shape == (10, 3)  # 10 samples, 3 classes

    def test_compile_model(self, sample_model):
        """Test la compilation du modèle"""
        sample_model.compile_model(learning_rate=0.01)
        assert isinstance(sample_model.model.optimizer, tf.keras.optimizers.Adam)
        assert sample_model.model.loss == 'categorical_crossentropy'
        assert 'accuracy' in sample_model.model.metrics_names

    def test_module_output_shapes(self, sample_model):
        """Test les formes de sortie de chaque module"""
        test_input = np.random.rand(1, 60, 5)
        
        # Test GA Module
        ga_output = sample_model._build_ga_module()(test_input)
        assert ga_output.shape[1] == 64  # 64 unités Dense
        
        # Test CNN Module 
        cnn_output = sample_model._build_cnn_module()(test_input)
        assert cnn_output.shape[1] == 29  # (60-3)/2 + 1
        
        # Test LSTM Module
        lstm_output = sample_model._build_lstm_module()(test_input)
        assert lstm_output.shape[1] == 60  # return_sequences=True

    @pytest.mark.parametrize("learning_rate", [0.001, 0.01, 0.1])
    def test_learning_rate_effect(self, sample_model, learning_rate):
        """Test différents taux d'apprentissage"""
        sample_model.compile_model(learning_rate=learning_rate)
        assert sample_model.model.optimizer.learning_rate.numpy() == learning_rate
