import pytest
import pandas as pd
import numpy as np
from Morningstar.utils import bt_analysis

@pytest.fixture
def multi_quote_crypto_data():
    # Sample data with different crypto quote currencies (USDT, BTC)
    trades_data = {
        'pair': ['BTC/USDT', 'ETH/BTC'],
        'open_trade_size': [5000, 2.5],  # Trade size in quote currency (5000 USDT, 2.5 BTC)
        'close_trade_size': [5250, 2.4], # Trade size in quote currency (5250 USDT, 2.4 BTC)
        'open_fee': [5, 0.002], # Fees in quote currency (5 USDT, 0.002 BTC)
        'close_fee': [5.25, 0.002], # Fees in quote currency (5.25 USDT, 0.002 BTC)
        'quote_currency': ['USDT', 'BTC'], # Specify quote currency for each trade
        'wallet': [10000, 0.5], # Wallet value *before* trade in quote currency (10k USDT, 0.5 BTC)
        'open_date': pd.to_datetime(['2023-01-01', '2023-01-05']),
        'close_date': pd.to_datetime(['2023-01-03', '2023-01-08'])
    }
    df_trades = pd.DataFrame(trades_data)

    # Daily data, assuming wallet is tracked in USDT
    days_data = {
        'wallet': [10000, 10100, 10050, 10200, 10150, 10300, 10250, 10400],
        'price': [50000, 50500, 50200, 51000, 50800, 51500, 51200, 52000], # Price of the main asset (e.g., BTC price for B&H calc in USDT)
        'day': pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05', '2023-01-06', '2023-01-07', '2023-01-08']),
        'quote_currency': ['USDT'] * 8 # Assuming daily wallet tracked in USDT
    }
    df_days = pd.DataFrame(days_data).set_index('day')

    return df_trades, df_days

def test_multi_quote_crypto_analysis(multi_quote_crypto_data):
    df_trades, df_days = multi_quote_crypto_data

    # Conversion rates to USDT (example: 1 BTC = 50000 USDT)
    # We need the rate at the time of calculation, using an average/snapshot here for simplicity
    quote_conversion_rates = {'USDT': 1.0, 'BTC': 50000.0}

    # Test simple_backtest_analysis with conversion
    # Need to provide price column for buy_and_hold calculation
    if 'price' not in df_days.columns:
         df_days['price'] = 50000 # Add dummy price if missing for test setup

    # Use copies to avoid modifying fixture data during calculations within the function
    result_trades, result_days = bt_analysis.simple_backtest_analysis(
        df_trades.copy(), df_days.copy(), quote_conversion_rates=quote_conversion_rates, general_info=False)

    # --- Verification ---
    # Check if columns exist after processing
    assert 'trade_result' in result_trades.columns
    assert 'trade_result_pct' in result_trades.columns
    assert 'daily_return' in result_days.columns

    # Verify converted values (example checks)
    # First trade was in USDT, should remain unchanged (multiplied by 1.0)
    assert np.isclose(result_trades.iloc[0]['open_trade_size'], 5000 * quote_conversion_rates['USDT'])
    assert np.isclose(result_trades.iloc[0]['close_trade_size'], 5250 * quote_conversion_rates['USDT'])
    assert np.isclose(result_trades.iloc[0]['open_fee'], 5 * quote_conversion_rates['USDT'])
    assert np.isclose(result_trades.iloc[0]['close_fee'], 5.25 * quote_conversion_rates['USDT'])

    # Second trade was in BTC, should be converted to USDT
    assert np.isclose(result_trades.iloc[1]['open_trade_size'], 2.5 * quote_conversion_rates['BTC'])
    assert np.isclose(result_trades.iloc[1]['close_trade_size'], 2.4 * quote_conversion_rates['BTC'])
    assert np.isclose(result_trades.iloc[1]['open_fee'], 0.002 * quote_conversion_rates['BTC'])
    assert np.isclose(result_trades.iloc[1]['close_fee'], 0.002 * quote_conversion_rates['BTC'])

    # Verify wallet conversion in daily data (assuming original df_days was already in USDT)
    # The function converts based on 'quote_currency' column, which is USDT here.
    assert np.isclose(result_days.iloc[0]['wallet'], 10000 * quote_conversion_rates['USDT'])
    assert np.isclose(result_days.iloc[-1]['wallet'], 10400 * quote_conversion_rates['USDT'])

    # Verify trade result calculation after conversion
    # Trade 1 (USDT): 5250 - 5000 - 5 - 5.25 = 239.75
    expected_result_1 = (5250 * 1.0) - (5000 * 1.0) - (5 * 1.0) - (5.25 * 1.0)
    assert np.isclose(result_trades.iloc[0]['trade_result'], expected_result_1)
    # Trade 2 (BTC converted to USDT): (2.4 * 50000) - (2.5 * 50000) - (0.002 * 50000) - (0.002 * 50000)
    # = 120000 - 125000 - 100 - 100 = -5200
    expected_result_2 = (2.4 * 50000.0) - (2.5 * 50000.0) - (0.002 * 50000.0) - (0.002 * 50000.0)
    assert np.isclose(result_trades.iloc[1]['trade_result'], expected_result_2)


def test_backward_compatibility_crypto(multi_quote_crypto_data):
    df_trades, df_days = multi_quote_crypto_data

    # Test without quote conversion (should process trades as is, potentially mixing units if not careful)
    # Need to provide price column for buy_and_hold calculation
    if 'price' not in df_days.columns:
         df_days['price'] = 50000 # Add dummy price if missing for test setup

    # Use copies to avoid modifying fixture data
    result_trades, result_days = bt_analysis.simple_backtest_analysis(
        df_trades.copy(), df_days.copy(), general_info=False)

    # Basic checks for backward compatibility
    assert len(result_trades) == 2
    assert len(result_days) == 8 # Based on the fixture data length
    assert 'trade_result' in result_trades.columns # Ensure calculations still run
    assert 'daily_return' in result_days.columns

    # Check that original values were used (no conversion applied)
    # Trade 1 (USDT)
    expected_result_1_no_conv = 5250 - 5000 - 5 - 5.25
    assert np.isclose(result_trades.iloc[0]['trade_result'], expected_result_1_no_conv)
    # Trade 2 (BTC)
    expected_result_2_no_conv = 2.4 - 2.5 - 0.002 - 0.002
    assert np.isclose(result_trades.iloc[1]['trade_result'], expected_result_2_no_conv)
