import argparse
import logging
import os
import pandas as pd
import sys

# Assurer que le répertoire 'utils' est dans le PYTHONPATH
# Ceci est une approche simple; une meilleure solution serait d'utiliser des packages installables.
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir)) # Remonter de deux niveaux (pipelines -> data -> project_root)
utils_path = os.path.join(project_root, 'utils')
if utils_path not in sys.path:
    sys.path.append(utils_path)

# Importer les fonctions des modules utils
try:
    from data_preparation import load_raw_data, clean_data, save_processed_data
    # Importer les fonctions des autres développeurs (supposées exister)
    # Si ces fichiers ou fonctions n'existent pas encore, Python lèvera une ImportError
    from feature_engineering import apply_feature_pipeline # Fonction attendue de Dev B
    from labeling import build_labels # Fonction attendue de Dev C
except ImportError as e:
    print(f"Erreur d'importation: Assurez-vous que les fichiers et fonctions nécessaires existent dans le dossier 'utils'. Détails: {e}")
    sys.exit(1) # Arrêter si les dépendances ne sont pas prêtes

# Configuration du Logging
LOG_DIR = os.path.join(project_root, 'logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_file_path = os.path.join(LOG_DIR, 'data_pipeline.log')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout) # Afficher aussi les logs dans la console
    ]
)

logger = logging.getLogger(__name__)

# Nombre de colonnes attendu dans le dataset final
EXPECTED_COLUMNS = 38

def run_pipeline(input_path: str, output_path: str):
    """
    Exécute le pipeline complet de préparation des données.

    Args:
        input_path (str): Chemin vers le fichier de données brutes (CSV).
        output_path (str): Chemin où sauvegarder le dataset final (Parquet).

    Returns:
        bool: True si le pipeline s'est terminé avec succès, False sinon.
    """
    logger.info(f"Début du pipeline de données pour le fichier: {input_path}")
    pipeline_success = False # Flag pour suivre le succès

    # 1. Charger les données brutes
    logger.info("Étape 1: Chargement des données brutes...")
    df_raw = load_raw_data(input_path)
    if df_raw is None:
        logger.error("Échec du chargement des données brutes. Arrêt du pipeline.")
        return False # Indiquer l'échec
    logger.info(f"Données brutes chargées. Shape: {df_raw.shape}")

    # 2. Nettoyer les données
    logger.info("Étape 2: Nettoyage des données...")
    df_clean = clean_data(df_raw.copy()) # Utiliser copy() pour éviter les modifications sur l'original
    logger.info(f"Données nettoyées. Shape: {df_clean.shape}")

    # 3. Appliquer le Feature Engineering (Fonction de Dev B)
    logger.info("Étape 3: Application du Feature Engineering...")
    try:
        # Supposons que apply_feature_pipeline prend le df nettoyé et retourne le df avec features
        df_features = apply_feature_pipeline(df_clean.copy())
        logger.info(f"Feature Engineering appliqué. Shape: {df_features.shape}")
    except NameError:
         logger.error("La fonction 'apply_feature_pipeline' n'est pas définie. Assurez-vous qu'elle existe dans utils/feature_engineering.py.")
         # Optionnel: Continuer sans features ou arrêter
         # df_features = df_clean # Continuer sans features
         return False # Arrêter et indiquer l'échec
    except Exception as e:
        logger.error(f"Erreur lors de l'application du Feature Engineering: {e}")
        return False # Indiquer l'échec

    # 4. Générer les Labels (Fonction de Dev C)
    logger.info("Étape 4: Génération des Labels...")
    try:
        # Supposons que build_labels prend le df avec features et retourne le df avec labels
        df_labeled = build_labels(df_features.copy())
        logger.info(f"Labels générés. Shape: {df_labeled.shape}")
    except NameError:
         logger.error("La fonction 'build_labels' n'est pas définie. Assurez-vous qu'elle existe dans utils/labeling.py.")
         # Optionnel: Continuer sans labels ou arrêter
         # df_labeled = df_features # Continuer sans labels
         return False # Arrêter et indiquer l'échec
    except Exception as e:
        logger.error(f"Erreur lors de la génération des Labels: {e}")
        return False # Indiquer l'échec

    # 5. Validation finale du nombre de colonnes
    final_columns = len(df_labeled.columns)
    logger.info(f"Validation finale: Le DataFrame final contient {final_columns} colonnes.")
    if final_columns != EXPECTED_COLUMNS:
        logger.warning(f"Attention: Le nombre de colonnes final ({final_columns}) ne correspond pas aux {EXPECTED_COLUMNS} colonnes attendues!")
        # Lister les colonnes présentes vs attendues pourrait être utile ici
        # logger.debug(f"Colonnes présentes: {df_labeled.columns.tolist()}")

    # 6. Sauvegarder le dataset final
    logger.info(f"Étape 5: Sauvegarde du dataset final au format Parquet: {output_path}")
    # Utiliser la fonction de data_preparation qui retourne un booléen
    if save_processed_data(df_labeled, output_path, format='parquet'):
        logger.info("Pipeline de données terminé avec succès.")
        pipeline_success = True
    else:
        logger.error("Échec de la sauvegarde du dataset final.")
        pipeline_success = False

    return pipeline_success


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline de préparation des données pour le modèle Morningstar.")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Chemin vers le fichier de données brutes d'entrée (format CSV)."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Chemin vers le fichier de sortie pour le dataset traité (format Parquet)."
    )

    args = parser.parse_args()

    # Vérifier si le fichier d'entrée existe
    if not os.path.exists(args.input):
        logger.error(f"Le fichier d'entrée spécifié n'existe pas: {args.input}")
        sys.exit(1) # Exit code 1 si input file missing

    # Exécuter le pipeline et gérer le code de sortie
    try:
        success = run_pipeline(args.input, args.output)
        if success:
            sys.exit(0) # Exit code 0 si succès
        else:
            sys.exit(1) # Exit code 1 si échec signalé par run_pipeline
    except Exception as e:
        logger.exception(f"Exception inattendue non interceptée dans run_pipeline: {e}")
        sys.exit(1) # Exit code 1 en cas d'exception imprévue
