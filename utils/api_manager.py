import argparse
import ccxt
import numpy as np
import pandas as pd
import time
import os
import logging
import sys
from datetime import datetime
from typing import Dict, Any, Optional

# Configuration du Logging (existant)
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_file_path = os.path.join(LOG_DIR, 'api_manager.log')
report_file_path = os.path.join(LOG_DIR, 'data_download_report.txt') # Ajout pour le rapport

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler() # Afficher aussi dans la console
    ]
)
logger = logging.getLogger(__name__)

MIN_ROWS_THRESHOLD = 500 # Seuil minimum de lignes pour la vérification

class APIManager:
    """Wrapper pour l'interface API attendue par le workflow de trading"""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialise le manager API avec la configuration.

        Args:
            config: Configuration de l'API depuis config.yaml
        """
        self.config = config
        self.exchange = self._init_exchange()

    def _init_exchange(self):
        """Initialise la connexion à l'exchange"""
        logger.info(f"Initialisation de l'exchange via APIManager: {self.config.get('exchange', 'binance')}")
        try:
            exchange_class = getattr(ccxt, self.config.get('exchange', 'binance'))
            # Note: Pourrait nécessiter de charger les clés API depuis un fichier .env ou config
            # pour les opérations authentifiées (non fait ici pour fetch_ohlcv public)
            return exchange_class({
                # 'apiKey': self.config.get('api_key'), # Charger depuis secrets.env si besoin
                # 'secret': self.config.get('api_secret'), # Charger depuis secrets.env si besoin
                'timeout': self.config.get('timeout', 30000),
                # 'enableRateLimit': True, # ccxt gère le rate limit automatiquement
            })
        except AttributeError:
            logger.error(f"Exchange '{self.config.get('exchange')}' non trouvé par ccxt.")
            raise
        except Exception as e:
            logger.error(f"Erreur d'initialisation de l'exchange via APIManager: {e}")
            raise

    def get_market_data(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Récupère les données de marché au format attendu par le modèle.

        Returns:
            Dictionnaire avec:
            - technical: array numpy des features techniques
            - sentiment_embeddings: array numpy des embeddings LLM
            Ou None si erreur.
        """
        logger.info("Récupération des données de marché via APIManager...")
        try:
            pair = self.config.get('pair', 'BTC/USDT')
            timeframe = self.config.get('timeframe', '1h')
            limit = self.config.get('lookback', 100)

            logger.debug(f"Appel à fetch_ohlcv pour {pair}, {timeframe}, limit={limit}")
            ohlcv = self.exchange.fetch_ohlcv(pair, timeframe, limit=limit)

            if not ohlcv:
                logger.warning("Aucune donnée OHLCV retournée par l'exchange.")
                return None

            # Convertir en features techniques (simplifié)
            # Garder seulement O, H, L, C pour cet exemple
            technical_data = [[candle[1], candle[2], candle[3], candle[4]] for candle in ohlcv]
            technical = np.array(technical_data)

            # Embeddings factices - à remplacer par l'appel réel au LLM/autre source
            logger.debug("Génération d'embeddings factices.")
            embeddings = np.random.rand(768) # Taille exemple pour BERT base

            logger.info(f"Données techniques ({technical.shape}) et embeddings ({embeddings.shape}) récupérés.")
            return {
                'technical': technical,
                'sentiment_embeddings': embeddings
            }

        except ccxt.NetworkError as e:
            logger.error(f"Erreur réseau lors de la récupération des données: {e}")
        except ccxt.ExchangeError as e:
            logger.error(f"Erreur de l'exchange lors de la récupération des données: {e}")
        except Exception as e:
            logger.error(f"Erreur inattendue lors de la récupération des données: {e}")

        return None # Retourner None en cas d'erreur

    def execute_orders(self, decisions: Dict[str, Any]) -> bool:
        """
        Exécute les ordres de trading basés sur les décisions du modèle.

        Args:
            decisions: Dictionnaire de décisions de trading

        Returns:
            bool: True si l'exécution a réussi (simulation ici)
        """
        logger.info("Exécution des ordres via APIManager...")
        try:
            # Implémentation simplifiée - à adapter pour ordres réels
            # Nécessiterait des clés API chargées dans _init_exchange
            order_type = decisions.get('type') # 'buy', 'sell', 'close'
            symbol = decisions.get('symbol', self.config.get('pair', 'BTC/USDT'))
            amount = decisions.get('amount', 0.001) # Exemple de taille
            price = decisions.get('price') # Pour ordres limit

            if order_type in ['buy', 'sell']:
                order_action = 'buy' if order_type == 'buy' else 'sell'
                # Simuler un ordre market pour simplifier
                logger.info(f"SIMULATION: Création d'un ordre {order_action} market pour {amount} {symbol}")
                # ordre_result = self.exchange.create_market_order(symbol, order_action, amount)
                # logger.info(f"Résultat de l'ordre simulé: {ordre_result}")
                print(f"--- Ordre {order_action.upper()} simulé pour {amount} {symbol} ---") # Pour visibilité
                return True
            elif order_type == 'close':
                logger.info(f"SIMULATION: Fermeture de position pour {symbol}")
                # Logique de fermeture de position...
                return True
            else:
                logger.warning(f"Type de décision non reconnu: {order_type}")
                return False

        except ccxt.AuthenticationError:
            logger.error("Erreur d'authentification. Vérifiez les clés API.")
            return False
        except ccxt.InsufficientFunds:
            logger.error("Fonds insuffisants pour exécuter l'ordre.")
            return False
        except ccxt.ExchangeError as e:
            logger.error(f"Erreur de l'exchange lors de l'exécution de l'ordre: {e}")
            return False
        except Exception as e:
            logger.error(f"Erreur inattendue lors de l'exécution des ordres: {e}")
            return False

# --- Fonctions Standalone (utilisées par le script en __main__) ---

def format_symbol(token: str, exchange_id: str) -> str:
    """
    Formate le symbole du token selon les conventions ccxt (ex: BTC/USDT).
    Certains exchanges utilisent des formats sans '/', cette fonction tente de standardiser.
    """
    if '/' in token:
        return token.upper()
    # Essayer de deviner la quote currency (USDT, BUSD, BTC, ETH...)
    possible_quotes = ['USDT', 'BUSD', 'USDC', 'BTC', 'ETH']
    for quote in possible_quotes:
        if token.endswith(quote):
            base = token[:-len(quote)]
            formatted = f"{base}/{quote}"
            logger.info(f"Conversion du token '{token}' au format standard ccxt: '{formatted}'")
            return formatted.upper()
    # Si pas de quote trouvée, retourner tel quel (peut échouer sur l'exchange)
    logger.warning(f"Impossible de déterminer le format standard avec '/' pour '{token}'.")

    # Cas spécifique pour KuCoin qui utilise souvent '-'
    if exchange_id == 'kucoin' and '/' not in token:
        for quote in possible_quotes:
            if token.endswith(quote):
                base = token[:-len(quote)]
                formatted_dash = f"{base}-{quote}"
                logger.info(f"Tentative avec le format KuCoin: '{formatted_dash}'")
                return formatted_dash.upper()

    logger.warning(f"Utilisation du token '{token.upper()}' tel quel.")
    return token.upper()

def fetch_ohlcv_data(exchange_id: str, token: str, timeframe: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Récupère les données OHLCV depuis un exchange via ccxt pour une utilisation standalone.
    Gère la pagination et les erreurs spécifiques.
    """
    logger.info(f"Tentative de connexion à l'exchange: {exchange_id} (fonction standalone)")
    try:
        exchange_class = getattr(ccxt, exchange_id)
        # Initialisation sans clés API pour données publiques
        exchange = exchange_class({'enableRateLimit': True})
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
    # Utiliser une limite raisonnable, certains exchanges ont des max plus bas (ex: 500, 1000)
    limit = exchange.safe_integer(exchange.limits.get('fetchOHLCV', {}), 'max', 1000)
    logger.info(f"Utilisation de la limite de fetchOHLCV: {limit}")

    # Formater le symbole initialement
    symbol = format_symbol(token, exchange_id)

    while since < end_msec:
        try:
            logger.debug(f"Récupération des données pour {symbol} sur {exchange_id} depuis {exchange.iso8601(since)}...")
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since, limit)

            if not ohlcv: # Si aucune donnée n'est retournée pour cette période
                logger.warning(f"Aucune donnée retournée pour {symbol} depuis {exchange.iso8601(since)}. Vérifiez la disponibilité des données ou la fin de l'historique.")
                # Avancer le timestamp pour éviter boucle infinie si l'API retourne vide avant la fin
                since += limit * exchange.parse_timeframe(timeframe) * 1000
                if since >= end_msec:
                     logger.info("Timestamp 'since' a dépassé la date de fin après une réponse vide.")
                     break # Sortir si on dépasse la date de fin
                else:
                     logger.info(f"Avance du timestamp à {exchange.iso8601(since)} et nouvelle tentative.")
                     time.sleep(exchange.rateLimit / 1000) # Pause avant de réessayer
                     continue # Essayer la période suivante

            # Filtrer les données pour ne pas dépasser end_date (certains exchanges retournent plus)
            ohlcv = [candle for candle in ohlcv if candle[0] <= end_msec]
            if not ohlcv:
                 logger.debug("Toutes les bougies retournées sont après la date de fin.")
                 break # Si le filtrage vide la liste, on a dépassé la date de fin

            all_ohlcv.extend(ohlcv)
            last_timestamp = ohlcv[-1][0]
            logger.info(f"{len(ohlcv)} bougies récupérées. Dernière bougie: {exchange.iso8601(last_timestamp)}")
            since = last_timestamp + exchange.parse_timeframe(timeframe) * 1000 # Aller à la bougie suivante

            # Petite pause pour éviter le rate limiting (géré par ccxt avec enableRateLimit=True)
            # time.sleep(exchange.rateLimit / 1000) # Peut être redondant si enableRateLimit fonctionne bien

        except ccxt.RateLimitExceeded as e:
            logger.warning(f"Rate limit dépassé: {e}. Attente automatique gérée par ccxt...")
            # ccxt devrait gérer l'attente si enableRateLimit=True
            time.sleep(5) # Ajout d'une pause supplémentaire par sécurité
        except ccxt.NetworkError as e:
            logger.error(f"Erreur réseau lors de la récupération: {e}. Nouvelle tentative dans 10s...")
            time.sleep(10) # Attente plus longue en cas de problème réseau
        except ccxt.ExchangeNotAvailable as e:
            logger.error(f"Exchange {exchange_id} non disponible: {e}. Arrêt.")
            return None # Erreur probablement persistante
        except ccxt.AuthenticationError as e:
             logger.error(f"Erreur d'authentification pour {exchange_id}: {e}. Vérifiez si des clés API sont nécessaires.")
             return None
        except ccxt.ExchangeError as e:
            # Erreur spécifique de l'exchange (ex: symbole invalide, etc.)
            logger.error(f"Erreur de l'exchange {exchange_id} pour {symbol}: {e}.")
            # Tentative de reformatage pour KuCoin si l'erreur est BadSymbol et le format est avec '/'
            if exchange_id == 'kucoin' and isinstance(e, ccxt.BadSymbol) and '/' in symbol:
                new_symbol = symbol.replace('/', '-')
                logger.warning(f"Symbole '{symbol}' invalide pour KuCoin. Nouvelle tentative avec '{new_symbol}'...")
                symbol = new_symbol # Mettre à jour le symbole pour la prochaine itération/tentative
                continue # Recommencer la boucle while avec le nouveau symbole

            logger.error(f"Arrêt de la récupération pour {symbol} suite à une erreur de l'exchange non récupérable.")
            return None # Erreur fatale pour cette requête
        except Exception as e:
            logger.exception(f"Erreur inattendue lors de la récupération des données pour {symbol}: {e}. Arrêt.") # Log stack trace
            return None

    if not all_ohlcv:
        logger.warning(f"Aucune donnée OHLCV n'a pu être récupérée pour {symbol} sur la période spécifiée.")
        return None

    # Convertir en DataFrame pandas
    try:
        df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        # Trier et supprimer les doublons (certains exchanges peuvent en retourner)
        df = df.sort_values('timestamp').drop_duplicates(subset=['timestamp'], keep='first')
        # Mettre le timestamp en index pour cohérence avec data_preparation
        df = df.set_index('timestamp')
        logger.info(f"Total de {len(df)} bougies uniques récupérées pour {symbol}.")
        return df
    except Exception as e:
        logger.error(f"Erreur lors de la conversion en DataFrame: {e}")
        return None


def save_data(df: pd.DataFrame, output_path: str) -> bool:
    """Sauvegarde le DataFrame en CSV."""
    logger.info(f"Tentative de sauvegarde vers {output_path}")
    try:
        # Créer le répertoire parent si nécessaire
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir): # Vérifier si output_dir n'est pas vide
            os.makedirs(output_dir)
            logger.info(f"Répertoire créé: {output_dir}")

        df.to_csv(output_path, index=True) # Sauvegarder avec l'index timestamp
        logger.info(f"Données sauvegardées avec succès dans: {output_path}")
        return True
    except Exception as e:
        logger.error(f"Erreur lors de la sauvegarde du fichier {output_path}: {e}")
        return False

def verify_downloaded_file(file_path: str, min_rows: int = MIN_ROWS_THRESHOLD) -> bool:
    """Vérifie si le fichier existe et contient un nombre minimum de lignes."""
    logger.info(f"Vérification du fichier: {file_path} (seuil: {min_rows} lignes)")
    if not os.path.exists(file_path):
        logger.error(f"Vérification échouée: Le fichier {file_path} n'existe pas.")
        return False
    try:
        # Vérifier la taille du fichier d'abord (rapide)
        if os.path.getsize(file_path) < 50: # Taille arbitraire pour un fichier CSV presque vide
             logger.warning(f"Vérification échouée: Le fichier {file_path} semble vide ou très petit.")
             return False

        # Lire le fichier pour compter les lignes
        df = pd.read_csv(file_path, index_col='timestamp') # Lire avec index
        num_rows = len(df)
        if num_rows >= min_rows:
            logger.info(f"Vérification réussie: Le fichier {file_path} contient {num_rows} lignes (>= {min_rows}).")
            return True
        else:
            logger.warning(f"Vérification échouée: Le fichier {file_path} contient seulement {num_rows} lignes (< {min_rows}).")
            return False
    except pd.errors.EmptyDataError:
         logger.error(f"Vérification échouée: Le fichier {file_path} est vide.")
         return False
    except Exception as e:
        logger.error(f"Erreur lors de la lecture ou vérification du fichier {file_path}: {e}")
        return False

def write_report(report_path: str, token: str, exchange: str, timeframe: str, start: str, end: str, status: str, num_rows: Optional[int] = None, error_msg: Optional[str] = None):
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
                f.write(f"Message: {error_msg}\n")
            f.write("-" * 40 + "\n\n")
        logger.info(f"Rapport de téléchargement mis à jour: {report_path}")
    except Exception as e:
        logger.error(f"Impossible d'écrire le rapport {report_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Télécharge les données OHLCV depuis un exchange crypto.")
    parser.add_argument("--token", type=str, required=True, help="Symbole du token (ex: BTC/USDT, ETHUSDT).")
    parser.add_argument("--exchange", type=str, required=True, help="ID de l'exchange ccxt (ex: binance, kucoin, bitget).")
    parser.add_argument("--start", type=str, required=True, help="Date de début (format YYYY-MM-DD).")
    parser.add_argument("--end", type=str, required=True, help="Date de fin (format YYYY-MM-DD).")
    parser.add_argument("--timeframe", type=str, required=True, help="Timeframe (ex: 1m, 5m, 15m, 1h, 4h, 1d).")
    parser.add_argument("--output", type=str, required=True, help="Chemin du fichier CSV de sortie.")
    parser.add_argument("--min_rows", type=int, default=MIN_ROWS_THRESHOLD, help=f"Nombre minimum de lignes pour considérer le téléchargement réussi (défaut: {MIN_ROWS_THRESHOLD}).")

    args = parser.parse_args()

    # Utiliser la fonction de formatage
    token_symbol_formatted = format_symbol(args.token, args.exchange)

    logger.info(f"Début du téléchargement pour {args.token} ({token_symbol_formatted}) sur {args.exchange} [{args.timeframe}] de {args.start} à {args.end}")

    download_status = "Échec"
    final_num_rows = 0
    error_message = "Erreur inconnue." # Message par défaut

    try:
        df_data = fetch_ohlcv_data(args.exchange, token_symbol_formatted, args.timeframe, args.start, args.end)

        if df_data is not None and not df_data.empty:
            final_num_rows = len(df_data)
            if save_data(df_data, args.output):
                if verify_downloaded_file(args.output, args.min_rows):
                    download_status = "Succès"
                    error_message = None # Pas d'erreur si succès
                else:
                    download_status = "Échec (Vérification échouée)"
                    error_message = f"Le fichier contient {final_num_rows} lignes (< {args.min_rows} requis)."
            else:
                download_status = "Échec (Sauvegarde échouée)"
                error_message = f"Erreur lors de l'écriture du fichier CSV: {args.output}"
        elif df_data is None:
             download_status = "Échec (Erreur Fetch)"
             error_message = "La fonction fetch_ohlcv_data a retourné None (voir logs pour détails)."
             final_num_rows = 0
        else: # df_data is empty
            download_status = "Échec (Aucune donnée)"
            error_message = "Aucune donnée retournée par l'exchange pour cette période/paire."
            final_num_rows = 0

    except Exception as e:
        logger.exception("Une erreur non gérée est survenue dans le processus principal.") # Log l'exception complète
        download_status = "Échec (Erreur Inattendue)"
        error_message = f"Exception: {str(e)}"
        final_num_rows = 0 # Pas de données si exception majeure

    # Écrire le rapport
    write_report(report_file_path, args.token, args.exchange, args.timeframe, args.start, args.end, download_status, final_num_rows, error_message)

    logger.info(f"Fin du processus de téléchargement. Statut: {download_status}")

    # Quitter avec un code d'erreur si échec pour signaler au script appelant
    if download_status != "Succès":
        sys.exit(1)
    else:
        sys.exit(0)
