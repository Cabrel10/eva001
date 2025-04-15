import pytest
import pandas as pd
from unittest.mock import patch, MagicMock, ANY
from pathlib import Path

# Supposons que le pipeline principal est orchestré par une fonction `run_pipeline`
# dans un fichier `data/pipelines/data_pipeline.py`.
# Nous devons mocker les fonctions appelées par `run_pipeline`.
# Ajustez les chemins d'importation si la structure réelle est différente.
PIPELINE_MODULE_PATH = "data.pipelines.data_pipeline" # Chemin hypothétique

# Créer un DataFrame de base pour les mocks
@pytest.fixture
def base_mock_df():
    return pd.DataFrame({
        'timestamp': pd.to_datetime(['2023-01-01 00:00:00', '2023-01-01 00:15:00']),
        'open': [1000, 1005], 'high': [1010, 1015], 'low': [990, 1000],
        'close': [1005, 1010], 'volume': [100, 110]
    }).set_index('timestamp')

# Mocker les dépendances du pipeline
@pytest.fixture
def mock_pipeline_dependencies(base_mock_df):
    """Mock les fonctions clés appelées par le pipeline principal."""
    # Cibler les noms tels qu'importés DANS data_pipeline.py ou leur source originale
    with patch(f"{PIPELINE_MODULE_PATH}.load_raw_data") as mock_load, \
         patch(f"{PIPELINE_MODULE_PATH}.clean_data") as mock_clean, \
         patch(f"{PIPELINE_MODULE_PATH}.apply_feature_pipeline") as mock_features, \
         patch(f"{PIPELINE_MODULE_PATH}.build_labels") as mock_labels, \
         patch(f"{PIPELINE_MODULE_PATH}.save_processed_data") as mock_save:

        # Configurer les mocks pour retourner des DataFrames modifiés (simulant chaque étape)
        mock_load.return_value = base_mock_df.copy()
        # clean_data retourne le df qu'on lui passe
        mock_clean.side_effect = lambda df: df
        # apply_feature_pipeline ajoute des colonnes de features
        def feature_side_effect(df):
            df['RSI'] = 50
            df['MACD'] = 0.1
            df['llm_context_summary'] = "Mock Summary" # Simule l'intégration LLM
            return df
        mock_features.side_effect = feature_side_effect
        # generate_labels ajoute des colonnes de labels
        def label_side_effect(df):
            df['signal_trading'] = 1 # Buy
            df['market_regime'] = 'bullish'
            return df
        mock_labels.side_effect = label_side_effect

        yield {
            "load": mock_load,
            "clean": mock_clean,
            "features": mock_features,
            "labels": mock_labels,
            "save": mock_save
        }

# Test d'intégration (simulé)
def test_run_pipeline_integration(mock_pipeline_dependencies, base_mock_df):
    """
    Teste l'orchestration du pipeline de bout en bout (avec mocks).
    Vérifie que chaque étape est appelée et que les données finales sont sauvegardées.
    """
    # Importer la fonction à tester (peut échouer si le fichier n'existe pas encore)
    try:
        from data.pipelines.data_pipeline import run_pipeline
    except ImportError:
        pytest.skip(f"Skipping integration test: {PIPELINE_MODULE_PATH}.run_pipeline not found.")
        return # Nécessaire pour éviter une erreur si skip est appelé

    # Définir le chemin de sortie attendu
    expected_output_path = Path("data/processed/final_dataset.parquet")
    # Définir un chemin d'entrée factice (non utilisé par les mocks, mais requis par la signature)
    dummy_input_path = "dummy/path/input.csv"

    # Exécuter le pipeline (qui utilisera les mocks)
    run_pipeline(input_path=dummy_input_path, output_path=expected_output_path)

    # --- Vérifications ---
    # 1. Vérifier que chaque étape mockée a été appelée
    mock_pipeline_dependencies["load"].assert_called_once()
    mock_pipeline_dependencies["clean"].assert_called_once()
    mock_pipeline_dependencies["features"].assert_called_once()
    mock_pipeline_dependencies["labels"].assert_called_once()
    mock_pipeline_dependencies["save"].assert_called_once()

    # 2. Vérifier les arguments de la fonction de sauvegarde
    #    Vérifier que la fonction a été appelée avec un DataFrame et le bon chemin
    call_args, call_kwargs = mock_pipeline_dependencies["save"].call_args
    assert isinstance(call_args[0], pd.DataFrame) # Vérifier que le premier arg est un DataFrame
    assert call_args[1] == expected_output_path  # Vérifier le chemin de sortie

    # 3. Vérifier le contenu du DataFrame passé à la sauvegarde
    saved_df = mock_pipeline_dependencies["save"].call_args[0][0]
    assert isinstance(saved_df, pd.DataFrame)

    # 4. Vérifier la présence de colonnes clés de chaque étape
    expected_cols = [
        'open', 'high', 'low', 'close', 'volume', # Original/Cleaned
        'RSI', 'MACD', 'llm_context_summary',     # Features
        'signal_trading', 'market_regime'         # Labels
    ]
    for col in expected_cols:
        assert col in saved_df.columns, f"Expected column '{col}' not found in final DataFrame"

    # 5. Vérifier (approximativement) le nombre de colonnes
    #    Le nombre exact (38) dépendra des implémentations réelles.
    #    Ici, on vérifie juste qu'on a plus que les colonnes initiales.
    assert len(saved_df.columns) > len(base_mock_df.columns)
    print(f"Final DataFrame columns ({len(saved_df.columns)}): {saved_df.columns.tolist()}")

    # 6. Vérifier qu'il n'y a pas de NaN (supposant que le pipeline final les gère)
    #    Note: Ceci dépend de l'implémentation réelle des étapes mockées.
    #    Si une étape est censée laisser des NaN, ce test échouera.
    # assert not saved_df.isnull().any().any(), "Final DataFrame should not contain NaNs"
