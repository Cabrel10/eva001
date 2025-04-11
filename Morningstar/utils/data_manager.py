import ccxt.async_support as ccxt # Import the async version
import pandas as pd
import numpy as np
import asyncio
from typing import Dict, Optional, List
from datetime import datetime, timezone
import time # Needed for sleep

class ExchangeDataManager:
    """
    Version optimisée du Data Manager avec :
    - Cache mémoire
    - Gestion des erreurs améliorée
    - Support multi-thread
    - Support pour start_date et end_date
    """
    def __init__(self, exchange_name: str):
        # Ensure the async exchange has fetchOHLCV capability
        if not hasattr(ccxt, exchange_name):
            raise ValueError(f"Async exchange '{exchange_name}' not found in ccxt.async_support.")
        
        exchange_class = getattr(ccxt, exchange_name)
        # Instantiate the async exchange class with increased timeout
        self.exchange = exchange_class({
            'timeout': 30000, # 30 seconds timeout for requests
        }) 
        
        # Check capability on the async instance
        if not self.exchange.has['fetchOHLCV']:
            raise ccxt.NotSupported(f"{exchange_name} does not support fetchOHLCV.") # Use ccxt specific error
            
        # self.exchange.load_markets() # Removed from __init__, needs to be called asynchronously
        self._cache = {}
        self.lock = asyncio.Lock()
        # Default limit per fetch, adjust if needed based on exchange
        self.fetch_limit = 1000

    async def load_markets_async(self, reload: bool = False):
        """Loads markets asynchronously if they haven't been loaded yet or if reload is True."""
        # Check if markets are already loaded to avoid redundant calls
        if not reload and self.exchange.markets:
             # print("Markets already loaded.")
             return

        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"Loading markets asynchronously... (Attempt {attempt + 1}/{max_retries})")
                await self.exchange.load_markets(reload)
                print("Markets loaded successfully.")
                if not self.exchange.markets:
                     print("Warning: load_markets completed but self.exchange.markets is still empty.")
                return # Exit successfully
            except (ccxt.RequestTimeout, ccxt.NetworkError, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection) as e:
                print(f"Error loading markets (Attempt {attempt + 1}/{max_retries}): {e}")
                if attempt == max_retries - 1:
                    print("Max retries reached for loading markets. Aborting.")
                    raise # Re-raise the exception after max retries
                wait_time = (attempt + 1) * 5 # Exponential backoff (5s, 10s, 15s)
                print(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            except Exception as e:
                 print(f"Unexpected error loading markets: {e}")
                 # Re-raise the original exception 'e' instead of a bare 'raise'
                 raise e
    # Correction de l'indentation ici (aligné avec load_markets_async)
    async def load_data(self, pair: str, timeframe: str,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """
        Charge les données OHLCV pour une paire et une période données.
        Supporte le chargement sur une plage de dates ou les N dernières bougies.

        Args:
            pair: La paire de trading (ex: 'BTC/USDT').
            timeframe: L'intervalle de temps (ex: '1m', '1h', '1d').
            start_date: Date de début (format 'YYYY-MM-DD' ou 'YYYY-MM-DD HH:MM:SS').
            end_date: Date de fin (format 'YYYY-MM-DD' ou 'YYYY-MM-DD HH:MM:SS').
            limit: Nombre maximum de bougies à récupérer si start_date n'est pas fourni.

        Returns:
            Un DataFrame pandas avec les données OHLCV.
        """
        # Validate pair
        if pair not in self.exchange.markets:
             raise ValueError(f"Pair '{pair}' not available on {self.exchange.id}")

        # Determine cache key
        cache_key_parts = [pair, timeframe]
        if start_date:
            cache_key_parts.append(start_date)
        if end_date:
            cache_key_parts.append(end_date)
        if limit and not start_date: # Only use limit in key if no start_date
             cache_key_parts.append(str(limit))
        cache_key = "_".join(cache_key_parts)

        async with self.lock:
            if cache_key in self._cache:
                print(f"Cache hit for {cache_key}")
                return self._cache[cache_key].copy()

            print(f"Cache miss for {cache_key}. Fetching data...")
            try:
                if start_date:
                    # Fetch data between start_date and end_date
                    since_ms = self._parse_date_to_ms(start_date)
                    if since_ms is None: # Handle parsing error
                         return pd.DataFrame()
                    end_ms = self._parse_date_to_ms(end_date) if end_date else None
                    
                    all_ohlcv = await self._fetch_ohlcv_range(pair, timeframe, since_ms, end_ms)
                
                elif limit:
                     # Fetch last 'limit' candles
                     all_ohlcv = await self._fetch_with_retry(pair, timeframe, limit=limit)
                else:
                    # Default: Fetch last default limit candles if neither start_date nor limit is given
                    all_ohlcv = await self._fetch_with_retry(pair, timeframe, limit=self.fetch_limit)

                if not all_ohlcv:
                    print(f"No data returned for {pair} {timeframe}")
                    self._cache[cache_key] = pd.DataFrame() # Cache empty result
                    return pd.DataFrame()

                df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                # Ensure timestamp is int64 before conversion
                df['timestamp'] = df['timestamp'].astype(np.int64)
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True) # Ensure UTC
                df.set_index('datetime', inplace=True)
                df.drop(columns=['timestamp'], inplace=True) # Drop redundant timestamp column
                
                # Ensure data types
                df = df.astype({'open': 'float64', 'high': 'float64', 'low': 'float64', 'close': 'float64', 'volume': 'float64'})

                # Filter again just in case the loop fetched slightly outside the range
                if start_date:
                    start_dt_aware = pd.to_datetime(start_date).tz_localize('UTC') if pd.to_datetime(start_date).tzinfo is None else pd.to_datetime(start_date).tz_convert('UTC')
                    df = df[df.index >= start_dt_aware]
                if end_date:
                    end_dt_aware = pd.to_datetime(end_date).tz_localize('UTC') if pd.to_datetime(end_date).tzinfo is None else pd.to_datetime(end_date).tz_convert('UTC')
                    # Include candles starting *before* or *at* the end_date
                    df = df[df.index <= end_dt_aware] 

                # Sort index just in case
                df.sort_index(inplace=True)
                
                # Remove duplicates based on index (datetime)
                df = df[~df.index.duplicated(keep='first')]

                # Mise en cache
                self._cache[cache_key] = df
                print(f"Data fetched and cached for {cache_key}. Shape: {df.shape}")
                return df.copy()

            except ccxt.NetworkError as e:
                 print(f"Network error fetching {pair} {timeframe}: {e}")
                 return pd.DataFrame() # Return empty DataFrame on network errors
            except ccxt.ExchangeError as e:
                 print(f"Exchange error fetching {pair} {timeframe}: {e}")
                 return pd.DataFrame() # Return empty DataFrame on exchange errors
            except Exception as e:
                import traceback
                print(f"Unexpected error loading data for {pair} {timeframe}: {e}")
                traceback.print_exc() # Print traceback for unexpected errors
                return pd.DataFrame()

    def _parse_date_to_ms(self, date_str: str) -> Optional[int]:
        """Converts a date string to milliseconds timestamp (UTC)."""
        if not date_str:
            return None
        try:
            # Attempt parsing with time first
            dt = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            # Fallback to parsing date only
            try:
                 dt = datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                 print(f"Invalid date format: {date_str}. Use 'YYYY-MM-DD' or 'YYYY-MM-DD HH:MM:SS'.")
                 return None # Return None on parsing failure

        # Assume UTC if no timezone info, otherwise convert to UTC
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)

        return int(dt.timestamp() * 1000)


    async def _fetch_ohlcv_range(self, pair: str, timeframe: str, since_ms: int, end_ms: Optional[int] = None) -> List:
        """Fetches OHLCV data in chunks for a given date range with progress indication."""
        all_ohlcv = []
        timeframe_duration_ms = self.exchange.parse_timeframe(timeframe) * 1000
        current_since = since_ms
        max_fetch_attempts = 3 # Max attempts for the whole range fetch
        
        # Estimate total chunks for progress (approximate)
        total_duration_ms = (end_ms if end_ms else datetime.now(timezone.utc).timestamp() * 1000) - since_ms
        estimated_total_candles = total_duration_ms / timeframe_duration_ms
        estimated_total_chunks = int(np.ceil(estimated_total_candles / self.fetch_limit)) if estimated_total_candles > 0 else 1
        
        print(f"[{pair}] Estimated total chunks: {estimated_total_chunks}")
        chunk_num = 0

        for fetch_attempt in range(max_fetch_attempts):
            print(f"[{pair}] Starting fetch loop, attempt {fetch_attempt + 1}/{max_fetch_attempts}")
            all_ohlcv = [] # Reset data for this attempt
            current_since = since_ms # Reset start time for this attempt
            chunk_num = 0 # Reset chunk number for retry

            while True:
                chunk_num += 1
                print(f"[{pair}] Fetching chunk {chunk_num}/{estimated_total_chunks} since {datetime.fromtimestamp(current_since/1000, tz=timezone.utc)}")
                try:
                    ohlcv = await self._fetch_with_retry(pair, timeframe, limit=self.fetch_limit, since=current_since)
                    
                    if not ohlcv:
                        # print("No more data returned by exchange.")
                        break # No more data available from the exchange for this period

                    first_candle_time = ohlcv[0][0]
                    last_candle_time = ohlcv[-1][0]
                    # print(f"Fetched {len(ohlcv)} candles from {datetime.fromtimestamp(first_candle_time/1000, tz=timezone.utc)} to {datetime.fromtimestamp(last_candle_time/1000, tz=timezone.utc)}")

                    # Filter out candles before the requested start time (since)
                    # Some exchanges might return candles slightly before the 'since' parameter
                    ohlcv = [c for c in ohlcv if c[0] >= current_since]
                    if not ohlcv:
                         # print("All fetched candles were before 'since', stopping.")
                         break # Avoid infinite loop if exchange keeps returning old data

                    all_ohlcv.extend(ohlcv)
                    print(f"[{pair}] Fetched {len(ohlcv)} candles in chunk {chunk_num}. Total fetched: {len(all_ohlcv)}")


                    # Prepare 'since' for the next fetch: timestamp of the last candle + timeframe duration
                    next_since = ohlcv[-1][0] + timeframe_duration_ms

                    # Check if the *start* of the next fetch period is beyond the end date
                    if end_ms and next_since > end_ms:
                        # print(f"Next fetch start ({datetime.fromtimestamp(next_since/1000, tz=timezone.utc)}) is after end date ({datetime.fromtimestamp(end_ms/1000, tz=timezone.utc)}). Stopping fetch.")
                        break 
                    
                    # Avoid infinite loop if the exchange returns the same last candle repeatedly
                    if next_since <= current_since:
                         print(f"Warning: Next fetch time ({datetime.fromtimestamp(next_since/1000, tz=timezone.utc)}) is not after current time ({datetime.fromtimestamp(current_since/1000, tz=timezone.utc)}). Stopping fetch to prevent infinite loop.")
                         break

                    current_since = next_since

                    # Respect rate limits
                    await asyncio.sleep(self.exchange.rateLimit / 1000) 

                except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable) as e:
                     print(f"Network/Timeout error during chunked fetch for {pair} {timeframe}: {e}. Breaking inner loop, will retry outer loop if possible.")
                     await asyncio.sleep(5) # Wait before potentially retrying the outer loop
                     break # Break inner loop, retry outer
                except ccxt.ExchangeError as e:
                     print(f"Exchange error during chunked fetch for {pair} {timeframe}: {e}. Stopping fetch for this pair.")
                     return [] # Return empty list on unrecoverable exchange errors
                except Exception as e:
                    import traceback
                    print(f"Unexpected error during chunked fetch for {pair} {timeframe}: {e}")
                    traceback.print_exc()
                    return [] # Return empty list on other unexpected errors
            
            # If the inner loop completed without breaking due to retryable errors, break the outer loop
            print(f"Fetch loop attempt {fetch_attempt + 1} completed.")
            break # Success, exit outer loop
        else:
             # This else block executes if the outer loop finished without a 'break' (i.e., all retries failed)
             print(f"Failed to fetch data for {pair} {timeframe} after {max_fetch_attempts} attempts.")
             return [] # Return empty list if all retries failed


        # --- Post-processing after successful fetch loop ---
        if not all_ohlcv:
             return []

        # Remove duplicates based on timestamp (first column)
        seen_timestamps = set()
        unique_ohlcv = []
        for candle in all_ohlcv:
            # Ensure candle has the expected structure (list/tuple of length 6)
            if isinstance(candle, (list, tuple)) and len(candle) >= 6:
                timestamp = candle[0]
                if isinstance(timestamp, (int, float)) and timestamp not in seen_timestamps:
                    unique_ohlcv.append(candle)
                    seen_timestamps.add(timestamp)
            else:
                print(f"Warning: Skipping malformed candle data: {candle}")

        
        # Sort by timestamp
        unique_ohlcv.sort(key=lambda x: x[0])

        # Final filter based on start_ms and end_ms
        # Filter candles starting strictly >= since_ms
        unique_ohlcv = [c for c in unique_ohlcv if c[0] >= since_ms]
        if end_ms:
            # Filter candles starting strictly <= end_ms
            unique_ohlcv = [c for c in unique_ohlcv if c[0] <= end_ms]
            
        return unique_ohlcv


    async def _fetch_with_retry(self, pair: str, timeframe: str,
                              limit: int, since: Optional[int] = None, max_retries: int = 5) -> list:
        """
        Tentative de récupération avec reprise sur erreur et gestion de 'since'.
        """
        params = {} # Optional parameters for fetch_ohlcv

        for attempt in range(max_retries):
            try:
                # print(f"Attempt {attempt+1}: Fetching {pair} {timeframe} limit {limit} since {datetime.fromtimestamp(since/1000, tz=timezone.utc) if since else 'None'}")
                # Use await directly on the coroutine
                result = await self.exchange.fetch_ohlcv(pair, timeframe, since=since, limit=limit, params=params)
                return result # Return the fetched data on success
            
            except (ccxt.NetworkError, ccxt.RequestTimeout, ccxt.ExchangeNotAvailable, ccxt.DDoSProtection) as e:
                if attempt == max_retries - 1:
                    print(f"Max retries reached for {pair} {timeframe}. Last error: {e}")
                    # Decide whether to raise or return empty list
                    # Returning empty list might be safer for batch processing
                    return [] 
                wait_time = (attempt + 1) * 2 # Exponential backoff
                print(f"Retryable error fetching {pair} {timeframe} (Attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            
            except ccxt.ExchangeError as e:
                 # Non-retryable exchange errors (e.g., invalid symbol, authentication error)
                 print(f"Non-retryable Exchange error fetching {pair} {timeframe}: {e}. Aborting fetch for this request.")
                 return [] # Return empty list for non-retryable errors
            
            except Exception as e:
                 # Catch any other unexpected errors
                 import traceback
                 print(f"Unexpected error during fetch attempt {attempt+1} for {pair} {timeframe}: {e}")
                 traceback.print_exc()
                 if attempt == max_retries - 1:
                     print(f"Max retries reached for {pair} {timeframe} after unexpected error.")
                     return [] # Return empty list after max retries on unexpected errors
                 await asyncio.sleep(1 * (attempt + 1)) # Simple backoff

        # This point should ideally not be reached if logic is correct, but return empty list as fallback
        print(f"Warning: _fetch_with_retry completed all retries without success or explicit error handling for {pair} {timeframe}.")
        return [] 

    async def close(self):
        """Ferme la connexion à l'échange."""
        if self.exchange and hasattr(self.exchange, 'close'):
            try:
                await self.exchange.close()
                print("Exchange connection closed.")
            except Exception as e:
                print(f"Error closing exchange connection: {e}")

# Example usage (optional, for testing)
# async def test_fetch():
#     manager = ExchangeDataManager('binance')
#     try:
#         # Test fetching last N candles
#         # df_limit = await manager.load_data('BTC/USDT', '1h', limit=10)
#         # print("Fetched last 10 candles:")
#         # print(df_limit.head())
#         # print(df_limit.tail())

#         # Test fetching date range
#         df_range = await manager.load_data('BTC/USDT', '1d', start_date='2023-01-01', end_date='2023-01-10')
#         print("\nFetched 2023-01-01 to 2023-01-10:")
#         print(df_range)

#         # Test cache
#         df_range_cached = await manager.load_data('BTC/USDT', '1d', start_date='2023-01-01', end_date='2023-01-10')
#         print("\nFetched 2023-01-01 to 2023-01-10 (cached):")
#         print(df_range_cached)

#         # Test invalid date
#         df_invalid = await manager.load_data('BTC/USDT', '1d', start_date='invalid-date')
#         print("\nFetched with invalid date:")
#         print(df_invalid)


#     finally:
#         await manager.close()

# if __name__ == "__main__":
#     # Setup basic asyncio logging if running standalone
#     import logging
#     logging.basicConfig(level=logging.INFO)
#     asyncio.run(test_fetch())
