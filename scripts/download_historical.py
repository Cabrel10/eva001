#!/usr/bin/env python3
"""
Script avancé de téléchargement de données historiques depuis Binance
avec sauvegarde en Parquet et gestion des timezones.
"""
import os
import argparse
from datetime import datetime, timedelta
import pandas as pd
from binance.client import Client
import pytz
import pyarrow.parquet as pq
from tqdm import tqdm
import time

# Configuration
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
BINANCE_TZ = pytz.UTC  # Binance utilise UTC
MAX_CANDLES_PER_REQUEST = 1000  # Limite API Binance

class BinanceDataDownloader:
    def __init__(self):
        self.client = Client(timeout=30)  # Configuration simplifiée
        self.supported_intervals = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY
        }

    def download_data(self, symbol: str, interval: str, 
                     start_date: datetime, end_date: datetime):
        """
        Télécharge les données par batch et les agrège
        """
        if interval not in self.supported_intervals:
            raise ValueError(f"Intervalle non supporté: {interval}")

        binance_interval = self.supported_intervals[interval]
        all_klines = []
        
        current_start = start_date
        pbar = tqdm(desc=f"Téléchargement {symbol} {interval}", unit=" batch")

        while current_start < end_date:
            current_end = min(
                current_start + timedelta(days=30),  # Binance limite à 1 mois par requête
                end_date
            )
            
            try:
                klines = self._get_klines_with_retry(
                    symbol=symbol,
                    interval=binance_interval,
                    start_str=current_start.strftime('%Y-%m-%d'),
                    end_str=current_end.strftime('%Y-%m-%d'),
                    limit=MAX_CANDLES_PER_REQUEST
                )
            except Exception as e:
                print(f"\nErreur lors du téléchargement: {e}")
                print(f"Période en échec: {current_start} à {current_end}")
                continue  # Passer au batch suivant
            
            all_klines.extend(klines)
            current_start = current_end + timedelta(seconds=1)
            pbar.update(1)
        
        pbar.close()
        return self._format_data(all_klines, interval)

    def _get_klines_with_retry(self, max_retries: int = 3, **kwargs):
        """Tentative de récupération avec reprise sur erreur"""
        for attempt in range(max_retries):
            try:
                return self.client.get_historical_klines(**kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                wait_time = (attempt + 1) * 5  # Backoff exponentiel
                print(f"\nTentative {attempt + 1} échouée. Nouvel essai dans {wait_time}s...")
                time.sleep(wait_time)

    def _format_data(self, klines: list, interval: str) -> pd.DataFrame:
        """
        Formate les données dans le même format que data_manager.py
        """
        cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore']
        
        df = pd.DataFrame(klines, columns=cols)
        
        # Conversion des types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, axis=1)
        
        # Conversion des timestamps
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Filtrage des colonnes pour correspondre à data_manager.py
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('timestamp', inplace=True)
        
        return df

    def save_as_parquet(self, df: pd.DataFrame, symbol: str, interval: str):
        """
        Sauvegarde au format Parquet avec partitionnement optimal
        """
        os.makedirs(DATA_DIR, exist_ok=True)
        filename = f"{symbol.replace('/', '-')}_{interval}.parquet"
        filepath = os.path.join(DATA_DIR, filename)
        
        df.to_parquet(filepath, engine='pyarrow')
        print(f"\nDonnées sauvegardées dans {filepath}")

def parse_args():
    parser = argparse.ArgumentParser(description='Télécharge les données historiques Binance')
    parser.add_argument('symbol', type=str, help='Paire de trading (ex: BTCUSDT)')
    parser.add_argument('interval', type=str, 
                       choices=['1m', '5m', '15m', '1h', '4h', '1d'],
                       help='Intervalle temporel')
    parser.add_argument('--start', type=str, default='2020-01-01',
                       help='Date de début (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, 
                       default=datetime.now().strftime('%Y-%m-%d'),
                       help='Date de fin (YYYY-MM-DD)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    downloader = BinanceDataDownloader()
    
    start_date = datetime.strptime(args.start, '%Y-%m-%d').replace(tzinfo=BINANCE_TZ)
    end_date = datetime.strptime(args.end, '%Y-%m-%d').replace(tzinfo=BINANCE_TZ)
    
    print(f"\nTéléchargement des données {args.symbol} {args.interval} "
          f"de {start_date} à {end_date}")
    
    df = downloader.download_data(
        symbol=args.symbol,
        interval=args.interval,
        start_date=start_date,
        end_date=end_date
    )
    
    downloader.save_as_parquet(df, args.symbol, args.interval)

if __name__ == '__main__':
    main()
