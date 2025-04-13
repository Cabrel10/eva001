import tensorflow as tf
from typing import Dict, Any
import os

class TFConfig:
    """Configuration complète pour TensorFlow avec support des données sociales"""
    
    # Paramètres de base
    BATCH_SIZE = 64
    EPOCHS = 100
    LEARNING_RATE = 0.001
    VALIDATION_SPLIT = 0.2
    
    # Configuration GPU
    GPU_CONFIG = {
        'memory_growth': True,
        'allow_growth': True,
        'visible_devices': '0' if tf.config.list_physical_devices('GPU') else None
    }
    
    # Clés API (à remplacer par les vraies valeurs ou variables d'environnement)
    TWITTER_KEYS = {
        'api_key': os.getenv('TWITTER_API_KEY', ''),
        'api_secret': os.getenv('TWITTER_API_SECRET', ''),
        'access_token': os.getenv('TWITTER_ACCESS_TOKEN', ''),
        'access_secret': os.getenv('TWITTER_ACCESS_SECRET', '')
    }
    
    REDDIT_KEYS = {
        'client_id': os.getenv('REDDIT_CLIENT_ID', ''),
        'client_secret': os.getenv('REDDIT_CLIENT_SECRET', ''),
        'username': os.getenv('REDDIT_USERNAME', ''),
        'password': os.getenv('REDDIT_PASSWORD', '')
    }

    @staticmethod
    def get_optimized_config() -> Dict[str, Any]:
        """Retourne la configuration optimisée pour TensorFlow"""
        return {
            'batch_size': TFConfig.BATCH_SIZE,
            'buffer_size': tf.data.AUTOTUNE,
            'prefetch_size': tf.data.AUTOTUNE,
            'parallel_calls': tf.data.AUTOTUNE,
            'optimizer': tf.keras.optimizers.Adam(learning_rate=TFConfig.LEARNING_RATE),
            'loss': 'mse',
            'metrics': ['mae', 'accuracy']
        }

    @staticmethod
    def get_trading_model_config() -> Dict[str, Any]:
        """Retourne la configuration optimisée pour le modèle de trading"""
        return {
            'batch_size': 32,
            'optimizer': 'adam',
            'loss': 'huber',
            'metrics': ['mae'],
            'learning_rate': 0.001,
            'input_shape': (50, 4)
        }

    @staticmethod
    def setup_gpu():
        """Configure les paramètres GPU"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"✅ GPU configuré: {gpus}")
            else:
                print("ℹ️ Aucun GPU détecté - Utilisation du CPU")
        except Exception as e:
            print(f"⚠️ Erreur lors de la configuration GPU: {str(e)}")
            print("ℹ️ Le système continuera avec le CPU")
