import pytest
import asyncio
import numpy as np
import pandas as pd
from unittest.mock import AsyncMock, MagicMock
from Morningstar.workflows import trading_workflow
from datetime import datetime

@pytest.fixture
def mock_config():
    return {
        'exchanges': {
            'binance': {
                'api_key': 'test',
                'api_secret': 'test'
            },
            'bitget': {
                'api_key': 'test',
                'api_secret': 'test'
            }
        },
        'social_keys': {
            'twitter': {
                'api_key': 'test',
                'api_secret': 'test',
                'access_token': 'test',
                'access_secret': 'test'
            },
            'reddit': {
                'client_id': 'test',
                'client_secret': 'test',
                'username': 'test',
                'password': 'test'
            }
        },
        'timeframe': '1h',
        'assets': ['BTC/USDT']
    }

@pytest.fixture
def mock_workflow(mock_config):
    workflow = trading_workflow.TradingWorkflow(mock_config)

    # Mock des composants
    # Mock ExchangeDataManager instances
    for manager in workflow.exchange_managers.values():
        manager.load_data = AsyncMock() # Mock load_data as an async function
        
    workflow.social_scraper = MagicMock()
    workflow.model = MagicMock()

    # Configurer les mocks
    mock_ohlcv_df = pd.DataFrame({
        'open': [50000] * 100,
        'high': [51000] * 100, 
        'low': [49000] * 100,
        'close': [50500] * 100,
        'volume': [100] * 100
    })
    
    # Set the return value for the mocked load_data
    for manager in workflow.exchange_managers.values():
        manager.load_data.return_value = mock_ohlcv_df
    
    mock_social_data = {
        'twitter': {
            'total': 100,
            'positive': 60,
            'negative': 20,
            'neutral': 20
        },
        'reddit': {
            'total': 50,
            'positive': 30,
            'negative': 10,
            'neutral': 10
        }
    }

    workflow.social_scraper.get_twitter_sentiment = AsyncMock(return_value=mock_social_data['twitter'])
    workflow.social_scraper.get_reddit_sentiment = AsyncMock(return_value=mock_social_data['reddit'])

    # Mock the model's predict method - adjust based on actual model output
    workflow.model.predict = MagicMock(return_value=[{
        'asset': 'BTC/USDT', # Assuming predict returns a list of dicts
        'buy_prob': 0.8,
        'sell_prob': 0.2,
        'sentiment': 0.5,
        'timestamp': datetime.now().isoformat()
    }]) # Added closing parenthesis here

    return workflow

@pytest.mark.asyncio
async def test_run_cycle(mock_workflow):
    """Test complet du workflow de trading"""
    signals = await mock_workflow.run_cycle()

    assert isinstance(signals, list)
    assert len(signals) == 1
    assert signals[0]['asset'] == 'BTC/USDT'
    assert signals[0]['signal'] in ['BUY', 'SELL', 'HOLD']
    assert 0 <= signals[0]['confidence'] <= 1

def test_feature_preparation(mock_workflow):
    """Test la préparation des features"""
    market_data = {
        'BTC/USDT': {
            'close': np.array([50000] * 100),
            'volume': np.array([100] * 100),
            'rsi': np.array([50] * 100),
            'macd': np.array([0] * 100),
            'signal': np.array([0] * 100)
        }
    }

    social_data = {
        'twitter': {'positive': 60, 'negative': 20},
        'reddit': {'positive': 30, 'negative': 10}
    }

    features = mock_workflow._prepare_features(market_data, social_data)

    assert len(features) == 1
    assert features[0]['asset'] == 'BTC/USDT'
    assert features[0]['technical'].shape == (50, 4)
    assert features[0]['sentiment'].shape == (2,)

@pytest.mark.asyncio
async def test_signal_generation(mock_workflow):
    """Test la génération des signaux"""
    predictions = [{
        'asset': 'BTC/USDT',
        'buy_prob': 0.8,
        'sell_prob': 0.2,
        'sentiment': 0.5,
        'timestamp': datetime.now().isoformat()
    }]

    signals = mock_workflow._generate_signals(predictions)

    assert signals[0]['signal'] == 'BUY'
    assert signals[0]['confidence'] == 0.8

@pytest.mark.asyncio
async def test_execute_orders(mock_workflow):
    """Test l'exécution des ordres"""
    signals = [{
        'asset': 'BTC/USDT',
        'signal': 'BUY',
        'confidence': 0.8,
        'timestamp': datetime.now().isoformat()
    }]

    result = await mock_workflow._execute_orders(signals)
    assert result is True
