import pytest
import pandas as pd
import numpy as np
from pathlib import Path

@pytest.fixture(scope="session")
def test_data_dir():
    return Path(__file__).parent / "test_data"

@pytest.fixture
def sample_price_data():
    """Retourne des données de prix BTC réalistes"""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=1000, freq="1h")
    base = 50000 + np.arange(1000) * 10
    noise = np.random.normal(0, 200, 1000)
    prices = base + noise
    
    return pd.DataFrame({
        'open': prices,
        'high': prices + np.random.uniform(50, 100, 1000),
        'low': prices - np.random.uniform(50, 100, 1000), 
        'close': prices,
        'volume': np.random.uniform(100, 1000, 1000)
    }, index=dates)

@pytest.fixture
def sample_trades():
    """Retourne des trades simulés réalistes"""
    return pd.DataFrame({
        'open_date': pd.date_range("2025-01-01", periods=100, freq="6h"),
        'close_date': pd.date_range("2025-01-01 01:00", periods=100, freq="6h"),
        'position': ['long'] * 50 + ['short'] * 50,
        'trade_result_pct': np.concatenate([
            np.random.normal(0.005, 0.002, 50),
            np.random.normal(-0.003, 0.002, 50)
        ])
    })
