"""Morningstar package - Crypto trading toolkit"""
from importlib.metadata import version
from Morningstar.configs.logging_config import setup_logging

# Initialisation des logs
setup_logging()

__version__ = version("Morningstar")

# Import des modules principaux
from .model.architecture import morningstar_model
from .workflows import morningstar_training, morningstar_trading
from .configs import morningstar_config

__all__ = [
    'morningstar_model',
    'morningstar_training', 
    'morningstar_trading',
    'morningstar_config'
]
