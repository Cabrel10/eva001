import pytest
import numpy as np
import tensorflow as tf
from Morningstar.model.architecture.base_model import BaseTradingModel

class TestBaseTradingModel:
    @pytest.fixture
    def model(self):
        return BaseTradingModel(input_shape=(32, 4))

    @pytest.fixture
    def sample_data(self):
        """Génère des données de test réalistes"""
        return {
            'features': np.random.rand(32, 4).astype(np.float32),
            'target': np.random.rand(32, 1).astype(np.float32)
        }

    def test_model_construction(self, model):
        """Test la structure du modèle"""
        assert len(model.model.layers) == 9
        assert isinstance(model.model.layers[1], tf.keras.layers.Conv1D)
        assert isinstance(model.model.layers[-1], tf.keras.layers.Dense)

    def test_model_prediction(self, model, sample_data):
        """Test que le modèle peut faire des prédictions"""
        prediction = model.model.predict(sample_data['features'][np.newaxis,...])
        assert prediction.shape == (1, 1)

    def test_model_training(self, model, sample_data):
        """Test l'entraînement basique"""
        dataset = tf.data.Dataset.from_tensor_slices((
            sample_data['features'][np.newaxis,...],
            sample_data['target'][np.newaxis,...]
        )).batch(1)
        
        history = model.train(dataset, epochs=2)
        assert 'loss' in history.history
