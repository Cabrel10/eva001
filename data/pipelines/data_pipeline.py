import argparse
import logging
import os
import pandas as pd
import sys
import numpy as np # Assurer que numpy est importé

# Assurer que le répertoire 'utils' est dans le PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
utils_path = os.path.join(project_root, 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

# Importer les fonctions des modules utils
try:
    from utils.data_preparation import load_raw_data, clean_data, save_processed_data
    from utils.feature_engineering import apply_feature_pipeline
    from utils.labeling import build_labels # build_labels doit retourner df avec labels + 14 dummies
except ImportError as e:
    print(f"Erreur d'importation: {e}")
    sys.exit(1)

# Configuration du Logging
LOG_DIR = os.path.join(project_root, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
log_file_path = os.path.join(LOG_DIR, 'data_pipeline.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_file_path), logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Nombre de colonnes techniques attendu AVANT ajout LLM
EXPECTED_TECHNICAL_COLUMNS = 38

# Colonnes qui ne sont PAS des features techniques (labels ou sources de labels)
NON_FEATURE_COLS = {
    'trading_signal', 'volatility', 'market_regime', 'level_sl', 'level_tp',
    # Exclure uniquement les colonnes de label, pas les features
    'dummy_placeholder_*', 'dummy_pipeline_*'
}

def run_pipeline(input_path: str, output_path: str):
    """Exécute le pipeline complet de préparation des données."""
    logger.info(f"Début du pipeline pour: {input_path}")
    pipeline_success = False

    # 1. Charger
    logger.info("Étape 1: Chargement...")
    df_raw = load_raw_data(input_path)
    if df_raw is None: return False
    logger.info(f"Données brutes: {df_raw.shape}")

    # 2. Nettoyer
    logger.info("Étape 2: Nettoyage...")
    df_clean = clean_data(df_raw.copy())
    logger.info(f"Données nettoyées: {df_clean.shape}")

    # 3. Feature Engineering
    logger.info("Étape 3: Feature Engineering...")
    try:
        df_features = apply_feature_pipeline(df_clean.copy()) # Devrait retourner 19 features
        logger.info(f"Features ajoutées: {df_features.shape}")
    except Exception as e:
        logger.error(f"Erreur Feature Engineering: {e}")
        return False

    # 4. Labeling
    logger.info("Étape 4: Labeling...")
    try:
        df_labeled = build_labels(df_features.copy()) # Devrait ajouter labels + 14 dummies
        logger.info(f"Labels ajoutés: {df_labeled.shape}")
    except Exception as e:
        logger.error(f"Erreur Labeling: {e}")
        return False

    # 5. Validation et ajout de colonnes factices pour atteindre 38 features techniques
    logger.info("Étape 5: Validation et ajout features factices...")
    # Identifier les colonnes techniques actuelles (toutes sauf labels)
    current_cols = set(df_labeled.columns)
    current_technical_cols = [col for col in current_cols
                            if not any(col.startswith(excl)
                                     for excl in ['trading_signal', 'volatility',
                                                'market_regime', 'level_sl', 'level_tp'])]
    num_current_technical = len(current_technical_cols)
    logger.info(f"Trouvé {num_current_technical} colonnes techniques actuelles: {current_technical_cols}")

    num_missing_technical = EXPECTED_TECHNICAL_COLUMNS - num_current_technical
    if num_missing_technical > 0:
        logger.warning(f"Ajout de {num_missing_technical} colonnes factices ('dummy_final_') pour atteindre {EXPECTED_TECHNICAL_COLUMNS} features techniques.")
        start_index = 1 # Commencer le nommage des dummies finaux
        # Trouver le plus grand index existant pour les dummies placeholders pour éviter collision
        existing_dummy_indices = [int(c.split('_')[-1]) for c in current_cols if c.startswith('dummy_placeholder_')]
        if existing_dummy_indices:
             start_index = max(existing_dummy_indices) + 1 # Nommer après les placeholders existants

        for i in range(num_missing_technical):
            # Utiliser un nommage différent pour éviter confusion avec ceux de build_labels
            col_name = f'dummy_pipeline_{start_index + i}'
            df_labeled[col_name] = 0.0
        logger.info(f"Colonnes après ajout factice: {df_labeled.shape[1]}")
    elif num_missing_technical < 0:
         logger.error(f"Trop de colonnes techniques trouvées ({num_current_technical}). Attendu {EXPECTED_TECHNICAL_COLUMNS}. Vérifiez les étapes précédentes.")
         return False
    else:
         logger.info(f"Nombre correct ({EXPECTED_TECHNICAL_COLUMNS}) de features techniques trouvé.")


    # 6. [SIMULATION] Ajouter embeddings LLM
    logger.info("Étape 6: [SIMULATION] Ajout embeddings LLM...")
    try:
        n_samples = df_labeled.shape[0]
        rng = np.random.default_rng()
        llm_embeddings_simulated = rng.normal(size=(n_samples, 768)).astype(np.float32)
        llm_columns = [f"llm_{i}" for i in range(768)]
        df_llm_simulated = pd.DataFrame(llm_embeddings_simulated, columns=llm_columns, index=df_labeled.index)
        df_final = pd.concat([df_labeled, df_llm_simulated], axis=1)
        logger.info(f"Embeddings LLM ajoutés. Shape finale: {df_final.shape}")
    except Exception as e:
        logger.error(f"Erreur ajout embeddings LLM: {e}")
        return False

    # 7. Sauvegarder
    logger.info(f"Étape 7: Sauvegarde: {output_path}")
    if save_processed_data(df_final, output_path, format='parquet'):
        logger.info("Pipeline terminé avec succès.")
        pipeline_success = True
    else:
        logger.error("Échec sauvegarde.")
        pipeline_success = False

    return pipeline_success

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de préparation des données.")
    parser.add_argument("--input", type=str, required=True, help="Fichier CSV brut.")
    parser.add_argument("--output", type=str, required=True, help="Fichier Parquet traité.")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        logger.error(f"Fichier d'entrée non trouvé: {args.input}")
        sys.exit(1)

    try:
        success = run_pipeline(args.input, args.output)
        sys.exit(0 if success else 1)
    except Exception as e:
        logger.exception(f"Exception non interceptée: {e}")
        sys.exit(1)
