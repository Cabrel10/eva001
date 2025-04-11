import tensorflow as tf
import numpy as np
from Morningstar.model.architecture.morningstar_model import MorningstarTradingModel
from Morningstar.configs.morningstar_config import MorningstarConfig, TradingStyle, RiskLevel
from Morningstar.utils.bt_analysis import BacktestAnalyzer, simple_backtest_analysis, get_metrics
import logging
import pandas as pd

class MorningstarTradingWorkflow:
    """Workflow d'exécution des trades avec le modèle Morningstar"""
    
    def __init__(self, model: MorningstarTradingModel, config: MorningstarConfig, initial_capital: float = 15):
        self.model = model
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.analyzer = BacktestAnalyzer()
        self.current_position = None
        self.portfolio = {
            'initial_capital': initial_capital,
            'balance': initial_capital, 
            'assets': 0,
            'peak': initial_capital,
            'n_trades': 0
        }
        self.trade_history = []
        
    async def run_live_trading(self, data_stream):
        """Exécution en temps réel"""
        self.logger.info("Démarrage du trading en temps réel")
        async for new_data in data_stream:
            prediction = self.predict(new_data)
            await self.execute_trade(prediction, new_data)
            
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Prédiction sur les nouvelles données"""
        processed_data = self._preprocess_data(data)
        return self.model.model.predict(processed_data)
    
    async def execute_trade(self, prediction: np.ndarray, market_data: pd.DataFrame):
        """Exécution des ordres basés sur la prédiction"""
        current_price = market_data['close'].iloc[-1]
        action = np.argmax(prediction)
        
        # Stratégie basée sur le style de trading et le risque
        if action == 0:  # Achat
            self._handle_buy_signal(current_price)
        elif action == 1:  # Vente
            self._handle_sell_signal(current_price)
            
        # Logging et monitoring
        self._log_portfolio_status()
        
    def _handle_buy_signal(self, price: float):
        """Gestion d'un signal d'achat avec suivi du capital"""
        if self.current_position != 'long':
            # Calcul dynamique basé sur la performance
            risk_factor = self._calculate_risk_factor()
            position_size = min(
                self._calculate_position_size(price),
                self.portfolio['balance'] * risk_factor / price
            )
            
            if position_size > 0:
                self.current_position = 'long'
                self.entry_price = price
                self.portfolio['assets'] = position_size
                self.portfolio['balance'] -= position_size * price
                self.logger.info(f"Achat à {price:.2f} - Taille: {position_size:.4f} - Capital: {self.portfolio['balance']:.2f}")
                
                # Enregistrement du trade
                self.trade_history.append({
                    'type': 'buy',
                    'price': price,
                    'size': position_size,
                    'timestamp': pd.Timestamp.now()
                })
    
    def _handle_sell_signal(self, price: float):
        """Gestion d'un signal de vente"""
        if self.current_position == 'long':
            self.portfolio['balance'] += self.portfolio['assets'] * price
            profit = (price * self.portfolio['assets']) - (self.portfolio['assets'] * self.entry_price)
            self.logger.info(f"Vente à {price:.2f} - Profit: {profit:.2f}")
            self.current_position = None
            self.portfolio['assets'] = 0
    
    def _calculate_risk_factor(self) -> float:
        """Calcule dynamiquement le facteur de risque basé sur la performance"""
        current_value = self.portfolio['balance'] + (self.portfolio['assets'] * self.current_price)
        drawdown = (self.portfolio['peak'] - current_value) / self.portfolio['peak']
        
        # Ajustement dynamique
        if drawdown > 0.1:  # Si drawdown > 10%
            return 0.5  # Réduit fortement le risque
        elif current_value > self.portfolio['peak']:
            self.portfolio['peak'] = current_value
            return 1.2  # Augmente le risque si nouveau peak
        return 1.0  # Risque normal

    def _calculate_position_size(self, price: float) -> float:
        """Calcul de la taille de position basée sur le risque"""
        base_risk = {
            RiskLevel.PRUDENT: 0.01,
            RiskLevel.MODERE: 0.03,
            RiskLevel.AGGRESSIF: 0.05
        }[self.config.risk_level]
        
        risk_factor = self._calculate_risk_factor()
        adjusted_risk = base_risk * risk_factor
        
        return (self.portfolio['balance'] * adjusted_risk) / price
    
    def _preprocess_data(self, data: pd.DataFrame) -> np.ndarray:
        """Prétraitement des données pour le modèle"""
        # Même normalisation que pendant l'entraînement
        return (data - data.mean()) / data.std()
    
    def _log_portfolio_status(self):
        """Journalisation du statut du portefeuille"""
        total = self.portfolio['balance'] + (self.portfolio['assets'] * self.current_price)
        self.logger.info(f"Portefeuille - Balance: {self.portfolio['balance']:.2f} | Actifs: {self.portfolio['assets']:.4f} | Total: {total:.2f}")
