"""Morningstar package - Crypto trading toolkit"""
# Remplacer l'import de importlib.metadata par l'import depuis version.py
# from importlib.metadata import version 
from ..version import __version__
from Morningstar.configs.logging_config import setup_logging

# Initialisation des logs
setup_logging()

# La ligne __version__ = version("Morningstar") est supprimée car __version__ est importé

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
