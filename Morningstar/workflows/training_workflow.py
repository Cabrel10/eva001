import tensorflow as tf
import asyncio
from datetime import datetime
from pathlib import Path
from Morningstar.utils.data_manager import ExchangeDataManager
from Morningstar.utils.social_scraper import SocialMediaScraper
from Morningstar.model.architecture.base_model import BaseTradingModel
from Morningstar.configs.tf_config import TFConfig

class TrainingWorkflow:
    """Workflow amélioré d'entraînement avec intégration des données sociales"""
    
    def __init__(self, pair="BTC/USDT", timeframe="1h"):
        self.pair = pair
        self.timeframe = timeframe
        self.data_manager = ExchangeDataManager("binance")
        self.social_scraper = SocialMediaScraper(
            twitter_keys=TFConfig.TWITTER_KEYS,
            reddit_keys=TFConfig.REDDIT_KEYS
        )
        self.model = BaseTradingModel()
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)

    async def run(self, epochs=50, include_social=False):
        """Exécute le workflow complet de manière asynchrone"""
        try:
            # 1. Chargement des données de marché
            market_data = await self.data_manager.load_data(
                self.pair, 
                self.timeframe
            )
            
            # 2. Chargement des données sociales (optionnel)
            social_data = None
            if include_social:
                social_data = await asyncio.gather(
                    self.social_scraper.get_twitter_sentiment(self.pair.split('/')[0]),
                    self.social_scraper.get_reddit_sentiment("CryptoCurrency")
                )
            
            # 3. Préparation du dataset
            dataset = self._prepare_dataset(market_data, social_data)
            
            # 4. Entraînement avec callbacks
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(self.models_dir/f"{self.pair.replace('/', '_')}_{self.timeframe}.h5"),
                    save_best_only=True,
                    monitor='val_loss'
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=f"logs/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
                )
            ]
            
            history = self.model.train(dataset, epochs=epochs, callbacks=callbacks)
            
            return history
            
        except Exception as e:
            print(f"Erreur dans le workflow: {e}")
            raise

    def _prepare_dataset(self, market_data, social_data=None):
        """Fusionne les données de marché et sociales"""
        # Conversion en TensorFlow Dataset
        dataset = tf.data.Dataset.from_tensor_slices(market_data)
        
        if social_data:
            # TODO: Implémenter la fusion avec les données sociales
            pass
            
        return dataset.batch(TFConfig.BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
