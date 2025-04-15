import argparse
import ccxt
import pandas as pd
import time
import os
import logging
from datetime import datetime

# Configuration du Logging (similaire au pipeline, mais peut être spécifique ici)
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs') # ../logs
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_file_path = os.path.join(LOG_DIR, 'api_manager.log')
report_file_path = os.path.join(LOG_DIR, 'data_download_report.txt')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

MIN_ROWS_THRESHOLD = 500 # Seuil minimum de lignes pour la vérification

def fetch_ohlcv_data(exchange_id, token, timeframe, start_date, end_date):
    """
    Récupère les données OHLCV depuis un exchange via ccxt.
    Gère la pagination pour récupérer toutes les données sur la période.
    """
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class()
        logger.info(f"Connecté à l'exchange: {exchange_id}")
    except AttributeError:
        logger.error(f"Exchange '{exchange_id}' non trouvé par ccxt.")
        return None
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation de l'exchange {exchange_id}: {e}")
        return None

    if not exchange.has['fetchOHLCV']:
        logger.error(f"L'exchange {exchange_id} ne supporte pas fetchOHLCV.")
        return None

    # Convertir les dates en timestamps millisecondes UTC
    try:
        since = exchange.parse8601(start_date + 'T00:00:00Z')
        end_msec = exchange.parse8601(end_date + 'T23:59:59Z') # Inclure la fin de la journée
        logger.info(f"Période demandée: {start_date} à {end_date}")
    except Exception as e:
        logger.error(f"Erreur lors de la conversion des dates: {e}. Utilisez le format YYYY-MM-DD.")
        return None

    all_ohlcv = []
    limit = 1000 # Nombre de bougies par requête (certains exchanges ont des limites plus basses)

    while since < end_msec:
        try:
            logger.debug(f"Récupération des données pour {token} sur {exchange_id} depuis {exchange.iso8601(since)}...")
            ohlcv = exchange.fetch_ohlcv(token, timeframe, since, limit)
            if not ohlcv: # Si aucune donnée n'est retournée, arrêter
                logger.warning(f"Aucune donnée retournée pour {token} depuis {exchange.iso8601(since)}. Arrêt de la pagination.")
                break

            # Filtrer les données pour ne pas dépasser end_date (certains exchanges retournent plus)
            ohlcv = [candle for candle in ohlcv if candle[0] <= end_msec]
            if not ohlcv:
                 break # Si le filtrage vide la liste, on a dépassé la date de fin

            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            since = last_timestamp + exchange.parse_timeframe(timeframe) * 1000 # Aller à la bougie suivante
            logger.info(f"{len(ohlcv)} bougies récupérées. Prochain timestamp: {exchange.iso8601(since)}")

            # Petite pause pour éviter le rate limiting
            time.sleep(exchange.rateLimit / 1000)

        except ccxt.NetworkError as e:
            logger.error(f"Erreur réseau lors de la récupération des données: {e}. Nouvelle tentative dans 5s...")
            time.sleep(5)
        except ccxt.ExchangeError as e:
            logger.error(f"Erreur de l'exchange {exchange_id}: {e}. Arrêt.")
            return None # Erreur fatale de l'exchange
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la récupération des données: {e}. Arrêt.")
            return None

    if not all_ohlcv:
        logger.warning(f"Aucune donnée OHLCV n'a pu être récupérée pour {token} sur la période spécifiée.")
        return None

    # Convertir en DataFrame pandas
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first')
    df = df.set_index('timestamp') # Optionnel: mettre le timestamp en index

    logger.info(f"Total de {len(df)} bougies uniques récupérées pour {token}.")
    return df

def save_data(df, output_path):
    """Sauvegarde le DataFrame en CSV."""
    try:
        # Créer le répertoire parent si nécessaire
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Répertoire créé: {output_dir}")

        df.to_csv(output_path, index=True) # Sauvegarder avec l'index timestamp
        logger.info(f"Données sauvegardées avec succès dans: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du fichier {output_path}: {e}")
        return False

def verify_downloaded_file(file_path, min_rows=MIN_ROWS_THRESHOLD):
    """Vérifie si le fichier existe et contient un nombre minimum de lignes."""
    if not os.path.exists(file_path):
        logger.error(f"Vérification échouée: Le fichier {file_path} n'existe pas.")
        return False
    try:
        # Lire juste assez pour compter les lignes rapidement (peut être optimisé)
        # Ou lire le fichier complet si la taille n'est pas un problème
        df = pd.read_csv(file_path)
        num_rows = len(df)
        if num_rows >= min_rows:
            logger.info(f"Vérification réussie: Le fichier {file_path} contient {num_rows} lignes (>= {min_rows}).")
            return True
        else:
            logger.warning(f"Vérification échouée: Le fichier {file_path} contient seulement {num_rows} lignes (< {min_rows}).")
            return False
    except Exception as e:
        logger.error(f"Erreur lors de la vérification du fichier {file_path}: {e}")
        return False

def write_report(report_path, token, exchange, timeframe, start, end, status, num_rows=None, error_msg=None):
    """Écrit un rapport simple sur le téléchargement."""
    try:
        with open(report_path, 'a') as f:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"--- Rapport de Téléchargement ({timestamp}) ---\n")
            f.write(f"Token: {token}\n")
            f.write(f"Exchange: {exchange}\n")
            f.write(f"Timeframe: {timeframe}\n")
            f.write(f"Période: {start} à {end}\n")
            f.write(f"Statut: {status}\n")
            if num_rows is not None:
                f.write(f"Lignes téléchargées: {num_rows}\n")
            if error_msg:
                f.write(f"Erreur: {error_msg}\n")
            f.write("-" * 40 + "\n\n")
        logger.info(f"Rapport de téléchargement mis à jour: {report_path}")
    except Exception as e:
        logger.error(f"Impossible d'écrire le rapport {report_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Télécharge les données OHLCV depuis un exchange crypto.")
    parser.add_argument("--token", type=str, required=True, help="Symbole du token (ex: BTC/USDT, ETH/USDT).")
    parser.add_argument("--exchange", type=str, required=True, help="ID de l'exchange ccxt (ex: binance, kucoin, bitget).")
    parser.add_argument("--start", type=str, required=True, help="Date de début (format YYYY-MM-DD).")
    parser.add_argument("--end", type=str, required=True, help="Date de fin (format YYYY-MM-DD).")
    parser.add_argument("--timeframe", type=str, required=True, help="Timeframe (ex: 1m, 5m, 15m, 1h, 4h, 1d).")
    parser.add_argument("--output", type=str, required=True, help="Chemin du fichier CSV de sortie.")

    args = parser.parse_args()

    # Convertir le token au format BASE/QUOTE si nécessaire (plus standard pour ccxt)
    if '/' not in args.token and 'USDT' in args.token:
        base = args.token.replace('USDT', '').replace('_SPBL', '') # Nettoyer les suffixes potentiels
        token_symbol = f"{base}/USDT"
        logger.info(f"Conversion du token '{args.token}' au format standard ccxt: '{token_symbol}'")
    else:
        token_symbol = args.token # Utiliser tel quel s'il contient déjà / ou n'est pas USDT

    logger.info(f"Début du téléchargement pour {token_symbol} sur {args.exchange} [{args.timeframe}] de {args.start} à {args.end}")

    download_status = "Échec"
    final_num_rows = 0
    error_message = None

    try:
        df_data = fetch_ohlcv_data(args.exchange, token_symbol, args.timeframe, args.start, args.end)

        if df_data is not None and not df_data.empty:
            if save_data(df_data, args.output):
                if verify_downloaded_file(args.output):
                    download_status = "Succès"
                    final_num_rows = len(df_data)
                else:
                    download_status = "Échec (Vérification échouée)"
                    error_message = f"Le fichier contient moins de {MIN_ROWS_THRESHOLD} lignes."
                    final_num_rows = len(df_data)
            else:
                download_status = "Échec (Sauvegarde échouée)"
                error_message = "Erreur lors de l'écriture du fichier CSV."
        else:
            download_status = "Échec (Aucune donnée)"
            error_message = "Aucune donnée n'a pu être récupérée ou le DataFrame est vide."

    except Exception as e:
        logger.exception("Une erreur non gérée est survenue dans le processus principal.") # Log l'exception complète
        download_status = "Échec (Erreur Inattendue)"
        error_message = str(e)

    # Écrire le rapport
    write_report(report_file_path, token_symbol, args.exchange, args.timeframe, args.start, args.end, download_status, final_num_rows, error_message)

    logger.info(f"Fin du processus de téléchargement. Statut: {download_status}")

    # Quitter avec un code d'erreur si échec
    if download_status != "Succès":
        # sys.exit(1) # Commenté pour permettre l'enchaînement des commandes même en cas d'échec partiel
        pass
