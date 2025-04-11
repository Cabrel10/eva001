import logging
import os
from pathlib import Path

def setup_logging():
    """Configuration centralisée des logs"""
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Format des logs
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configuration du logger principal
    logger = logging.getLogger('Morningstar')
    logger.setLevel(logging.DEBUG)

    # Handler pour fichier d'erreurs
    error_handler = logging.FileHandler('logs/error.log')
    error_handler.setLevel(logging.WARNING)
    error_handler.setFormatter(formatter)

    # Handler pour trading
    trading_handler = logging.FileHandler('logs/trading.log') 
    trading_handler.setLevel(logging.INFO)
    trading_handler.setFormatter(formatter)

    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    # Ajout des handlers
    logger.addHandler(error_handler)
    logger.addHandler(trading_handler)
    logger.addHandler(console_handler)

    # Configuration supplémentaire
    logging.captureWarnings(True)
    logging.getLogger('tensorflow').setLevel(logging.WARNING)
