#!/usr/bin/env python3
"""Script principal pour lancer le système Morningstar"""
import asyncio
import logging
from Morningstar.configs.morningstar_config import MorningstarConfig
from Morningstar.workflows.morningstar_training import MorningstarTrainingWorkflow
from Morningstar.workflows.morningstar_trading import MorningstarTradingWorkflow

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def main():
    """Workflow principal"""
    # Configuration
    config = MorningstarConfig(
        timeframe='1h',
        trading_style='swing',
        risk_level='modere'
    )
    
    # Entraînement du modèle
    trainer = MorningstarTrainingWorkflow(config)
    train_data, test_data = await trainer.prepare_data()
    model, history = trainer.train_model(train_data)
    
    # Initialisation du trading
    trader = MorningstarTradingWorkflow(model, config)
    
    # Backtest
    await trader.run_backtest(test_data)
    
    # Trading en live
    await trader.run_live_trading()

if __name__ == '__main__':
    asyncio.run(main())
