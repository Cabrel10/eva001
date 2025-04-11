"""Morningstar package - Crypto trading toolkit"""
# Supprimer l'import relatif et définir la version statiquement
# from ..version import __version__
from Morningstar.configs.logging_config import setup_logging

# Initialisation des logs
setup_logging()

# Définir la version statiquement pour éviter les problèmes d'import dans Colab
__version__ = "0.1.0" 

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
