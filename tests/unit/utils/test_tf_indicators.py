import pytest
import numpy as np
import tensorflow as tf
from Morningstar.utils.custom_indicators import calculate_tf_indicators

class TestTFIndicators:
    def test_tf_rsi(self):
        """Test le calcul du RSI avec TensorFlow"""
        closes = np.arange(100, dtype=np.float32)
        rsi_values = calculate_tf_indicators(closes, indicator='rsi')
        
        assert len(rsi_values) == len(closes)
        assert not np.isnan(rsi_values[-1])
        assert 0 <= rsi_values[-1] <= 100

    def test_tf_macd(self):
        """Test le calcul du MACD avec TensorFlow"""
        closes = np.sin(np.linspace(0, 10, 100)) * 100 + 100
        macd_line, signal = calculate_tf_indicators(closes, indicator='macd')
        
        assert len(macd_line) == len(closes)
        assert len(signal) == len(closes)
