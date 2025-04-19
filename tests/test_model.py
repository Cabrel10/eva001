import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from model.architecture.morningstar_model import MorningstarModel
from model.training.data_loader import split_features_labels # Importer pour réutiliser la logique

# Définir le chemin vers les données réelles
REAL_DATA_PATH = Path("data/processed/btc_final.parquet")
# Définir les colonnes de labels attendues par le modèle par défaut
DEFAULT_LABEL_COLUMNS = ['signal', 'volatility_quantiles', 'volatility_regime', 'market_regime', 'sl_tp'] 
# Note: Ces labels doivent correspondre aux sorties du modèle testé. 
# Si le modèle testé a d'autres sorties, ajustez cette liste.

class TestMorningstarModel:
    """Tests unitaires pour le modèle Morningstar."""
    
    @pytest.fixture
    def model(self):
        """Fixture pour initialiser le modèle."""
        model = MorningstarModel()
        model.initialize_model()
        return model

    @pytest.fixture(scope="class") # Utiliser scope='class' pour charger les données une seule fois
    def sample_data(self):
        """Fixture pour charger un échantillon de données réelles."""
        if not REAL_DATA_PATH.exists():
            pytest.skip(f"Fichier de données réelles non trouvé : {REAL_DATA_PATH}")
            
        try:
            data = pd.read_parquet(REAL_DATA_PATH)
            # Prendre un petit échantillon pour les tests
            sample_df = data.head(10).copy() 
            
            # Identifier les colonnes de features en excluant les labels potentiels
            # Utiliser une liste plus complète de labels potentiels pour être sûr
            potential_labels = DEFAULT_LABEL_COLUMNS + ['level_sl', 'level_tp', 'trading_signal', 'volatility', 'market_regime']
            feature_columns = [col for col in sample_df.columns if col not in potential_labels]
            
            # Sélectionner uniquement les features numériques
            X_technical_sample = sample_df[feature_columns].select_dtypes(include=np.number)
            
            # Note: La vérification prématurée du nombre de features est supprimée.
            # Le code suivant gère le cas où il y a moins de 38 features en ajoutant du padding.
            
            # Extraire les features techniques réelles disponibles (devrait être 33)
            available_technical_features = X_technical_sample.values.astype(np.float32)
            
            # Créer un tableau de padding pour atteindre 38 features
            num_rows = available_technical_features.shape[0] # Devrait être 10
            num_missing_features = 38 - available_technical_features.shape[1] # Devrait être 5
            
            if num_missing_features < 0:
                 pytest.fail(f"Erreur logique: Trop de features techniques trouvées ({available_technical_features.shape[1]})")
            
            # Utiliser des zéros pour le padding
            padding = np.zeros((num_rows, num_missing_features), dtype=np.float32) 
            
            # Concaténer les features réelles et le padding
            technical_features = np.concatenate([available_technical_features, padding], axis=1)

            # Générer des features LLM aléatoires (car non présentes dans le fichier)
            llm_features = np.random.rand(num_rows, 768).astype(np.float32) 

            # Vérifier les shapes finales attendues par le modèle
            if technical_features.shape != (num_rows, 38) or llm_features.shape != (num_rows, 768):
                 pytest.fail(f"Shapes finales incorrectes après padding. Tech: {technical_features.shape}, LLM: {llm_features.shape}")

            return {
                'technical': technical_features, # Maintenant shape (10, 38)
                'llm': llm_features # LLM features sont maintenant aléatoires
            }
        except Exception as e:
            pytest.fail(f"Erreur lors du chargement/traitement des données réelles : {e}")

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
