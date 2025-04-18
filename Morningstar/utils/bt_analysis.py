import pandas as pd
from dataclasses import dataclass

@dataclass
class BacktestAnalyzer:
    """Analyse complète des résultats de backtest"""
    
    def analyze(self, trades: pd.DataFrame, days: pd.DataFrame) -> dict:
        """Analyse principale avec les métriques clés"""
        return get_metrics(trades, days)
        
    def full_report(self, trades: pd.DataFrame, days: pd.DataFrame):
        """Rapport détaillé avec toutes les statistiques"""
        return simple_backtest_analysis(
            trades,
            days,
            general_info=True,
            trades_info=True,
            days_info=True,
            long_short_info=True
        )

import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np

def get_metrics(df_trades, df_days, quote_conversion_rates=None):
    df_days_copy = df_days.copy()
    
    # Convert wallet to reference quote currency if rates provided
    if quote_conversion_rates is not None:
        # Assuming df_days has a 'quote_currency' column if conversion is needed
        if 'quote_currency' in df_days_copy.columns:
            df_days_copy['wallet'] = df_days_copy.apply(
                lambda x: x['wallet'] * quote_conversion_rates.get(x['quote_currency'], 1), axis=1)
    
    df_days_copy['evolution'] = df_days_copy['wallet'].diff()
    df_days_copy['daily_return'] = df_days_copy['evolution']/df_days_copy['wallet'].shift(1)
    sharpe_ratio = (365**0.5)*(df_days_copy['daily_return'].mean()/df_days_copy['daily_return'].std())
    
    df_days_copy['wallet_ath'] = df_days_copy['wallet'].cummax()
    df_days_copy['drawdown'] = df_days_copy['wallet_ath'] - df_days_copy['wallet']
    df_days_copy['drawdown_pct'] = df_days_copy['drawdown'] / df_days_copy['wallet_ath']
    max_drawdown = -df_days_copy['drawdown_pct'].max() * 100
    
    df_trades_copy = df_trades.copy()
    df_trades_copy['trade_result'] = df_trades_copy["close_trade_size"] - df_trades_copy["open_trade_size"] - df_trades_copy["open_fee"] - df_trades_copy["close_fee"]
    df_trades_copy['trade_result_pct'] = df_trades_copy['trade_result']/df_trades_copy["open_trade_size"]
    good_trades = df_trades_copy.loc[df_trades_copy['trade_result_pct'] > 0]
    win_rate = len(good_trades) / len(df_trades)
    avg_profit = df_trades_copy['trade_result_pct'].mean()
    
    # Calculate Sortino Ratio
    negative_returns = df_days_copy['daily_return'][df_days_copy['daily_return'] < 0]
    sortino_denominator = negative_returns.std()
    if sortino_denominator == 0 or pd.isna(sortino_denominator):
        sortino_ratio = np.nan # Avoid division by zero or NaN std dev
    else:
        sortino_ratio = (365**0.5) * (df_days_copy['daily_return'].mean() / sortino_denominator)

    # Calculate Calmar Ratio
    annual_return = df_days_copy['daily_return'].mean() * 365
    max_drawdown_abs = -max_drawdown / 100 # Convert percentage back to absolute value
    if max_drawdown_abs == 0 or pd.isna(max_drawdown_abs):
         calmar_ratio = np.nan # Avoid division by zero
    else:
        calmar_ratio = annual_return / max_drawdown_abs

    return {
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "calmar_ratio": calmar_ratio,
        "win_rate": win_rate,
        "avg_profit_pct": avg_profit * 100, # Return as percentage
        "total_trades": len(df_trades_copy),
        "max_drawdown_pct": max_drawdown, # Already in percentage
    }
            
def simple_backtest_analysis(
    trades,
    days,
    general_info=True,
    trades_info=False,
    days_info=False,
    long_short_info=False,
    entry_exit_info=False,
    indepedant_trade=True,
    quote_conversion_rates=None # Renamed parameter
):
    df_trades = trades.copy()
    # Get first trade date and ensure we have at least one day before it
    first_trade_date = df_trades['open_date'].min()
    df_days = days.copy().loc[first_trade_date:]
    
    # Convert trade sizes and wallet to reference quote currency if rates provided
    if quote_conversion_rates is not None:
        # Assuming df_trades has a 'quote_currency' column
        if 'quote_currency' in df_trades.columns:
            df_trades['open_trade_size'] = df_trades.apply(
                lambda x: x['open_trade_size'] * quote_conversion_rates.get(x['quote_currency'], 1), axis=1)
            df_trades['close_trade_size'] = df_trades.apply(
                lambda x: x['close_trade_size'] * quote_conversion_rates.get(x['quote_currency'], 1), axis=1)
            # Also convert fees if they exist and are in quote currency
            if 'open_fee' in df_trades.columns:
                 df_trades['open_fee'] = df_trades.apply(
                    lambda x: x['open_fee'] * quote_conversion_rates.get(x['quote_currency'], 1), axis=1)
            if 'close_fee' in df_trades.columns:
                 df_trades['close_fee'] = df_trades.apply(
                    lambda x: x['close_fee'] * quote_conversion_rates.get(x['quote_currency'], 1), axis=1)

        # Assuming df_days has a 'quote_currency' column
        if 'quote_currency' in df_days.columns:
            df_days['wallet'] = df_days.apply(
                lambda x: x['wallet'] * quote_conversion_rates.get(x['quote_currency'], 1), axis=1)
            # Convert price if it represents the quote currency value (e.g., for B&H calc)
            if 'price' in df_days.columns:
                 df_days['price'] = df_days.apply(
                    lambda x: x['price'] * quote_conversion_rates.get(x['quote_currency'], 1), axis=1)
    
    if df_trades.empty:
        raise Exception("No trades found")
    if df_days.empty:
        raise Exception("No days found")

    df_days['evolution'] = df_days['wallet'].diff()
    df_days['daily_return'] = df_days['evolution']/df_days['wallet'].shift(1)
    
    # Corrected trade_result calculation to include close_fee
    df_trades['trade_result'] = df_trades["close_trade_size"] - df_trades["open_trade_size"] - df_trades["open_fee"] - df_trades["close_fee"]
    df_trades['trade_result_pct'] = df_trades['trade_result']/df_trades["open_trade_size"]
    # Note: trade_result_pct_wallet calculation might need review depending on when 'wallet' is measured (before/after trade)
    # For now, keeping it as is, assuming 'wallet' column in trades_df represents wallet *before* the trade's impact.
    df_trades['trade_result_pct_wallet'] = df_trades['trade_result']/(df_trades["wallet"]) # Corrected denominator? Or should it be wallet before trade? Needs clarification. Let's assume wallet column is pre-trade for now.
    if indepedant_trade:
        result_to_use = "trade_result_pct"
    else:
        result_to_use = "trade_result_pct_wallet"

    df_trades['trades_duration'] = df_trades['close_date'] - df_trades['open_date']
    
    df_trades['wallet_ath'] = df_trades['wallet'].cummax()
    df_trades['drawdown'] = df_trades['wallet_ath'] - df_trades['wallet']
    df_trades['drawdown_pct'] = df_trades['drawdown'] / df_trades['wallet_ath']
    df_days['wallet_ath'] = df_days['wallet'].cummax()
    df_days['drawdown'] = df_days['wallet_ath'] - df_days['wallet']
    df_days['drawdown_pct'] = df_days['drawdown'] / df_days['wallet_ath']
    
    good_trades = df_trades.loc[df_trades['trade_result'] > 0]
    if good_trades.empty:
        print("!!! No good trades found")
    bad_trades = df_trades.loc[df_trades['trade_result'] < 0]
    if bad_trades.empty:
        print("!!! No bad trades found")
    
    initial_wallet = df_days.iloc[0]["wallet"]
    total_trades = len(df_trades)
    total_days = len(df_days)
    if good_trades.empty:
        total_good_trades = 0
    else:
        total_good_trades = len(good_trades)
        
    avg_profit = df_trades[result_to_use].mean()

    try:
        avg_profit_good_trades = good_trades[result_to_use].mean()
        avg_profit_bad_trades = bad_trades[result_to_use].mean()
        total_bad_trades = len(bad_trades)
        mean_good_trades_duration = good_trades['trades_duration'].mean()
        mean_bad_trades_duration = bad_trades['trades_duration'].mean()
    except Exception as e:
        pass

    global_win_rate = total_good_trades / total_trades
    max_trades_drawdown = df_trades['drawdown_pct'].max()
    max_days_drawdown = df_days['drawdown_pct'].max()
    final_wallet = df_days.iloc[-1]['wallet']
    buy_and_hold_pct = (df_days.iloc[-1]['price'] - df_days.iloc[0]['price']) / df_days.iloc[0]['price']
    buy_and_hold_wallet = initial_wallet + initial_wallet * buy_and_hold_pct
    vs_hold_pct = (final_wallet - buy_and_hold_wallet)/buy_and_hold_wallet
    vs_usd_pct = (final_wallet - initial_wallet)/initial_wallet
    sharpe_ratio = (365**0.5)*(df_days['daily_return'].mean()/df_days['daily_return'].std())
    mean_trades_duration = df_trades['trades_duration'].mean()
    mean_trades_per_days = total_trades/total_days

    best_trade = df_trades[result_to_use].max()
    best_trade_date1 = str(df_trades.loc[df_trades[result_to_use] == best_trade].iloc[0]['open_date'])
    best_trade_date2 = str(df_trades.loc[df_trades[result_to_use] == best_trade].iloc[0]['close_date'])
    worst_trade = df_trades[result_to_use].min()
    worst_trade_date1 = str(df_trades.loc[df_trades[result_to_use] == worst_trade].iloc[0]['open_date'])
    worst_trade_date2 = str(df_trades.loc[df_trades[result_to_use] == worst_trade].iloc[0]['close_date'])

    df_days["win_loose"] = 0
    df_days.loc[df_days['daily_return'] > 0, "win_loose"] = 1
    df_days.loc[df_days['daily_return'] < 0, "win_loose"] = -1
    trade_days = df_days.loc[df_days['win_loose'] != 0]
    grouper = (trade_days["win_loose"] != trade_days["win_loose"].shift()).cumsum()
    df_days['streak'] = trade_days["win_loose"].groupby(grouper).cumsum()
    df_days['streak'] = df_days['streak'].ffill(axis=0)

    best_day = df_days.loc[df_days['daily_return'] == df_days['daily_return'].max()].iloc[0]
    worst_day = df_days.loc[df_days['daily_return'] == df_days['daily_return'].min()].iloc[0]
    worst_day_return = worst_day['daily_return']
    best_day_return = best_day['daily_return']
    worst_day_date = str(worst_day.name)
    best_day_date = str(best_day.name)
    best_streak = df_days.loc[df_days['streak'] == df_days['streak'].max()].iloc[0]
    worst_streak = df_days.loc[df_days['streak'] == df_days['streak'].min()].iloc[0]
    best_streak_date = str(best_streak.name)
    worst_streak_date = str(worst_streak.name)
    best_streak_number = best_streak['streak']
    worst_streak_number = worst_streak['streak']
    win_days_number = len(df_days.loc[df_days['win_loose'] == 1])
    loose_days_number = len(df_days.loc[df_days['win_loose'] == -1])
    neutral_days_number = len(df_days.loc[df_days['win_loose'] == 0])

    if general_info:
        print(f"Period: [{str(df_days.index[0])}] -> [{str(df_days.index[-1])}]")
        print(f"Initial wallet: {round(initial_wallet,2)} $")
        
        print("\n--- General Information ---")
        print(f"Final wallet: {round(final_wallet,2)} $")
        print(f"Performance: {round(vs_usd_pct*100,2)} %")
        print(f"Sharpe Ratio: {round(sharpe_ratio,2)}")
        print(f"Worst Drawdown T|D: -{round(max_trades_drawdown*100, 2)}% | -{round(max_days_drawdown*100, 2)}%")
        print(f"Buy and hold performance: {round(buy_and_hold_pct*100,2)} %")
        print(f"Performance vs buy and hold: {round(vs_hold_pct*100,2)} %")
        print(f"Total trades on the period: {total_trades}")
        print(f"Average Profit: {round(avg_profit*100, 2)} %")
        print(f"Global Win rate: {round(global_win_rate*100, 2)} %")

    if trades_info:
        print("\n--- Trades Information ---")
        print(f"Mean Trades per day: {round(mean_trades_per_days, 2)}")
        print(f"Best trades: +{round(best_trade*100, 2)} % the {best_trade_date1} -> {best_trade_date2}")
        print(f"Worst trades: {round(worst_trade*100, 2)} % the {worst_trade_date1} -> {worst_trade_date2}")
        try:
            print(f"Total Good trades on the period: {total_good_trades}")
            print(f"Total Bad trades on the period: {total_bad_trades}")
            print(f"Average Good Trades result: {round(avg_profit_good_trades*100, 2)} %")
            print(f"Average Bad Trades result: {round(avg_profit_bad_trades*100, 2)} %")
            print(f"Mean Good Trades Duration: {mean_good_trades_duration}")
            print(f"Mean Bad Trades Duration: {mean_bad_trades_duration}")
        except Exception as e:
            pass

    if days_info:
        print("\n--- Days Information ---")
        print(f"Total: {len(df_days)} days recorded")
        print(f"Winning days: {win_days_number} days ({round(100*win_days_number/len(df_days), 2)}%)")
        print(f"Neutral days: {neutral_days_number} days ({round(100*neutral_days_number/len(df_days), 2)}%)")
        print(f"Loosing days: {loose_days_number} days ({round(100*loose_days_number/len(df_days), 2)}%)")
        print(f"Longest winning streak: {round(best_streak_number)} days ({best_streak_date})")
        print(f"Longest loosing streak: {round(-worst_streak_number)} days ({worst_streak_date})")
        print(f"Best day: {best_day_date} (+{round(best_day_return*100, 2)}%)")
        print(f"Worst day: {worst_day_date} ({round(worst_day_return*100, 2)}%)")

    if long_short_info:
        long_trades = df_trades.loc[df_trades['position'] == "LONG"]
        short_trades = df_trades.loc[df_trades['position'] == "SHORT"]
        if long_trades.empty or short_trades.empty:
            print("!!! No long or short trades found")
        else:
            total_long_trades = len(long_trades)
            total_short_trades = len(short_trades)
            good_long_trades = long_trades.loc[long_trades['trade_result'] > 0]
            good_short_trades = short_trades.loc[short_trades['trade_result'] > 0]
            if good_long_trades.empty:
                total_good_long_trades = 0
            else:
                total_good_long_trades = len(good_long_trades)
            if good_short_trades.empty:
                total_good_short_trades = 0
            else:
                total_good_short_trades = len(good_short_trades)
            long_win_rate = total_good_long_trades / total_long_trades
            short_win_rate = total_good_short_trades / total_short_trades
            long_average_profit = long_trades[result_to_use].mean()
            short_average_profit = short_trades[result_to_use].mean()
            print("\n--- " + "LONG informations" + " ---")
            print(f"Total LONG trades on the period: {total_long_trades}")
            print(f"LONG Win rate: {round(long_win_rate*100, 2)} %")
            print(f"Average LONG Profit: {round(long_average_profit*100, 2)} %")
            print("\n--- " + "SHORT informations" + " ---")
            print(f"Total SHORT trades on the period: {total_short_trades}")
            print(f"SHORT Win rate: {round(short_win_rate*100, 2)} %")
            print(f"Average SHORT Profit: {round(short_average_profit*100, 2)} %")
    
    if entry_exit_info:
        print("\n" + "-" * 16 + " Entries " + "-" * 16)
        total_entries = len(df_trades)
        open_dict = df_trades.groupby("position")["open_reason"].value_counts().to_dict()
        for entry in open_dict:
            print(
                "{:<25s}{:>15s}".format(
                    entry[0] + " - " + entry[1],
                    str(open_dict[entry])
                    + " ("
                    + str(round(100 * open_dict[entry] / total_entries, 1))
                    + "%)",
                )
            )
        print("-" * 17 + " Exits " + "-" * 17)
        total_exits = len(df_trades)
        close_dict = df_trades.groupby("position")["close_reason"].value_counts().to_dict()
        for entry in close_dict:
            print(
                "{:<25s}{:>15s}".format(
                    entry[0] + " - " + entry[1],
                    str(close_dict[entry])
                    + " ("
                    + str(round(100 * close_dict[entry] / total_entries, 1))
                    + "%)",
                )
            )
        print("-" * 40)

    return df_trades, df_days
