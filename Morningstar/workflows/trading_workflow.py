import tensorflow as tf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from Morningstar.utils import data_manager, social_scraper, custom_indicators
from Morningstar.model.architecture import base_model
from Morningstar.configs import tf_config

class TradingWorkflow:
    def __init__(self, config):
        # Initialisation des composants
        # Initialisation des managers d'échange
        self.exchange_managers = {
            name: data_manager.ExchangeDataManager(name) 
            for name in config['exchanges']
        }
        self.social_scraper = social_scraper.SocialMediaScraper(
            twitter_keys=config['social_keys']['twitter'],
            reddit_keys=config['social_keys']['reddit']
        )
        self.model = base_model.BaseTradingModel(
            input_shape=tf_config.TFConfig.get_trading_model_config()['input_shape']
        )
        self.indicators = custom_indicators
        
        # Configuration
        self.timeframe = config.get('timeframe', '1h')
        self.assets = config.get('assets', ['BTC/USDT'])
        self.sentiment_threshold = config.get('sentiment_threshold', 0.3)
        
    async def run_cycle(self):
        """Exécute un cycle complet de trading"""
        try:
            # 1. Récupération des données
            market_data = await self._get_market_data()
            social_data = await self._get_social_data()
            
            # 2. Préparation des features
            features = self._prepare_features(market_data, social_data)
            
            # 3. Prédiction du modèle
            predictions = self.model.predict(features)
            
            # 4. Prise de décision
            signals = self._generate_signals(predictions)
            
            # 5. Exécution des ordres
            await self._execute_orders(signals)
            
            return signals
            
        except Exception as e:
            print(f"Error in trading cycle: {str(e)}")
            raise

    async def _get_market_data(self):
        """Récupère les données de marché"""
        ohlcv = {}
        for asset in self.assets:
            # Utilise le premier exchange disponible
            exchange = next(iter(self.exchange_managers.values()))
            data = await exchange.load_data(
                asset,
                timeframe=self.timeframe,
                limit=100
            )
            # Calcul des indicateurs
            data['rsi'] = self.indicators.calculate_tf_indicators(
                data['close'].values, 'rsi')
            data['macd'], data['signal'] = self.indicators.calculate_tf_indicators(
                data['close'].values, 'macd')
            ohlcv[asset] = data
        return ohlcv

    async def _get_social_data(self):
        """Récupère les données sociales"""
        return {
            'twitter': await self.social_scraper.get_twitter_sentiment("bitcoin"),
            'reddit': await self.social_scraper.get_reddit_sentiment("cryptocurrency")
        }

    def _prepare_features(self, market_data, social_data):
        """Prépare les features pour le modèle"""
        features = []
        for asset, data in market_data.items():
            # Features techniques
            # Convertir en DataFrame pour les calculs de pourcentage
            temp_df = pd.DataFrame({
                'close': data['close'],
                'volume': data['volume'],
                'rsi': data['rsi'],
                'macd_diff': data['macd'] - data['signal']
            })
            
            tech_features = np.column_stack([
                temp_df['close'].pct_change().values[-50:],
                temp_df['volume'].pct_change().values[-50:],
                temp_df['rsi'].values[-50:],
                temp_df['macd_diff'].values[-50:]
            ])
            
            # Features de sentiment
            sentiment = np.array([
                social_data['twitter']['positive'] - social_data['twitter']['negative'],
                social_data['reddit']['positive'] - social_data['reddit']['negative']
            ])
            
            features.append({
                'asset': asset,
                'technical': tech_features,
                'sentiment': sentiment,
                'timestamp': datetime.now().isoformat()
            })
        return features

    def _generate_signals(self, predictions):
        """Génère les signaux de trading"""
        signals = []
        for pred in predictions:
            signal = 'HOLD'
            if pred['buy_prob'] > 0.7 and pred['sentiment'] > self.sentiment_threshold:
                signal = 'BUY'
            elif pred['sell_prob'] > 0.7 and pred['sentiment'] < -self.sentiment_threshold:
                signal = 'SELL'
                
            signals.append({
                'asset': pred['asset'],
                'signal': signal,
                'confidence': max(pred['buy_prob'], pred['sell_prob']),
                'timestamp': pred['timestamp']
            })
        return signals

    async def _execute_orders(self, signals):
        """Exécute les ordres sur les exchanges"""
        # Implémentation réelle à ajouter ici
        print(f"Executing orders: {signals}")
        return True
