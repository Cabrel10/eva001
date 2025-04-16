import pytest
import numpy as np
from model.architecture.morningstar_model import MorningstarModel

class TestMorningstarModel:
    """Tests unitaires pour le modèle Morningstar."""
    
    @pytest.fixture
    def model(self):
        """Fixture pour initialiser le modèle."""
        model = MorningstarModel()
        model.initialize_model()
        return model
        
    @pytest.fixture
    def sample_data(self):
        """Fixture pour créer des données de test."""
        return {
            'technical': np.random.rand(10, 38).astype(np.float32),
            'llm': np.random.rand(10, 768).astype(np.float32)
        }
    
    def test_model_initialization(self, model):
        """Teste que le modèle s'initialise correctement."""
        assert model.model is not None
        assert len(model.model.inputs) == 2
        assert len(model.model.outputs) == 5  # 5 sorties (signal, volatility_quantiles, volatility_regime, market_regime, sl_tp)
        
        # Vérifie que l'input technique attend bien 38 features
        technical_input = model.model.inputs[0]
        assert technical_input.shape[-1] == 38
        
    def test_predict_output_shapes(self, model, sample_data):
        """Teste que les prédictions ont les bonnes shapes."""
        predictions = model.predict(sample_data['technical'], sample_data['llm'])
        
        assert predictions['signal'].shape == (10, 5)
        assert predictions['volatility_quantiles'].shape == (10, 3)
        assert predictions['volatility_regime'].shape == (10, 3)
        assert predictions['market_regime'].shape == (10, 4)
        assert predictions['sl_tp'].shape == (10, 2)
        
    def test_invalid_input_shapes(self, model):
        """Teste la gestion des inputs invalides."""
        with pytest.raises(ValueError):
            # Mauvaise shape pour les données techniques
            model.predict(np.random.rand(10, 20), np.random.rand(10, 768))
            
        with pytest.raises(ValueError):
            # Mauvaise shape pour les embeddings LLM
            model.predict(np.random.rand(10, 38), np.random.rand(10, 100))
            
    def test_save_load_weights(self, model, sample_data, tmp_path):
        """Teste la sauvegarde et le chargement des poids."""
        # Fait une prédiction de référence
        original_pred = model.predict(sample_data['technical'], sample_data['llm'])
        
        # Sauvegarde et recharge
        weights_path = tmp_path / "weights.h5"
        model.save_weights(str(weights_path))
        
        new_model = MorningstarModel()
        new_model.initialize_model()
        new_model.load_weights(str(weights_path))
        
        # Vérifie que les prédictions sont identiques
        new_pred = new_model.predict(sample_data['technical'], sample_data['llm'])
        
        for key in original_pred:
            np.testing.assert_allclose(original_pred[key], new_pred[key], atol=1e-6)
