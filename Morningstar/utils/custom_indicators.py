import math
import numpy as np
import pandas as pd
import ta
import math
import requests
import tensorflow as tf

def calculate_tf_indicators(prices, indicator='rsi', window=14):
    """Calculate technical indicators using TensorFlow
    
    Args:
        prices (np.array): Array of price values
        indicator (str): Type of indicator to calculate ('rsi' or 'macd')
        window (int): Lookback window for indicators
        
    Returns:
        np.array: Calculated indicator values
    """
    prices = tf.convert_to_tensor(prices, dtype=tf.float32)
    
    if indicator == 'rsi':
        # Pad with initial value to maintain length
        padded_prices = tf.concat([[prices[0]], prices], axis=0)
        deltas = padded_prices[1:] - padded_prices[:-1]
        gain = tf.where(deltas > 0, deltas, 0)
        loss = tf.where(deltas < 0, -deltas, 0)
        
        # Initialize with first window
        avg_gain = tf.reduce_mean(gain[:window])
        avg_loss = tf.reduce_mean(loss[:window])
        rsi_values = [50.0] * window  # Default value for first window
        
        # Calculate remaining values
        for i in range(window, len(gain)):
            avg_gain = (avg_gain * (window-1) + gain[i]) / window
            avg_loss = (avg_loss * (window-1) + loss[i]) / window
            
            if avg_loss == 0:
                rsi = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            rsi_values.append(float(rsi))
        
        return np.array(rsi_values, dtype=np.float32)
        
    elif indicator == 'macd':
        # Custom EMA implementation
        def calculate_ema(values, window):
            ema = tf.Variable(values[0], dtype=tf.float32)
            ema_values = [ema.numpy()]
            alpha = 2 / (window + 1)
            
            for value in values[1:]:
                ema.assign(alpha * value + (1 - alpha) * ema)
                ema_values.append(ema.numpy())
            
            return tf.convert_to_tensor(ema_values, dtype=tf.float32)

        ema12 = calculate_ema(prices, 12)
        ema26 = calculate_ema(prices, 26)
        macd_line = ema12 - ema26
        signal_line = calculate_ema(macd_line, 9)
        return macd_line.numpy(), signal_line.numpy()

def get_n_columns(df, columns, n=1):
    dt = df.copy()
    for col in columns:
        dt["n"+str(n)+"_"+col] = dt[col].shift(n)
    return dt

def chop(high, low, close, window=14):
    ''' Choppiness indicator
    '''
    tr1 = pd.DataFrame(high - low).rename(columns={0: 'tr1'})
    tr2 = pd.DataFrame(abs(high - close.shift(1))
                       ).rename(columns={0: 'tr2'})
    tr3 = pd.DataFrame(abs(low - close.shift(1))
                       ).rename(columns={0: 'tr3'})
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').dropna().max(axis=1)
    atr = tr.rolling(1).mean()
    highh = high.rolling(window).max()
    lowl = low.rolling(window).min()
    chop_serie = 100 * np.log10((atr.rolling(window).sum()) /
                          (highh - lowl)) / np.log10(window)
    return pd.Series(chop_serie, name="CHOP")

def fear_and_greed(close):
    ''' Fear and greed indicator
    '''
    response = requests.get("https://api.alternative.me/fng/?limit=0&format=json")
    dataResponse = response.json()['data']
    fear = pd.DataFrame(dataResponse, columns = ['timestamp', 'value'])

    fear = fear.set_index(fear['timestamp'])
    fear.index = pd.to_datetime(fear.index, unit='s')
    del fear['timestamp']
    df = pd.DataFrame(close, columns = ['close'])
    df['fearResult'] = fear['value']
    df['FEAR'] = df['fearResult'].ffill()
    df['FEAR'] = df.FEAR.astype(float)
    return pd.Series(df['FEAR'], name="FEAR")


class Trix():
    """ Trix indicator

        Args:
            close(pd.Series): dataframe 'close' columns,
            trix_length(int): the window length for each mooving average of the trix,
            trix_signal_length(int): the window length for the signal line
    """

    def __init__(
        self,
        close: pd.Series,
        trix_length: int = 9,
        trix_signal_length: int = 21,
        trix_signal_type: str = "sma" # or ema
    ):
        self.close = close
        self.trix_length = trix_length
        self.trix_signal_length = trix_signal_length
        self.trix_signal_type = trix_signal_type
        self._run()

    def _run(self):
        self.trix_line = ta.trend.ema_indicator(
            ta.trend.ema_indicator(
                ta.trend.ema_indicator(
                    close=self.close, window=self.trix_length),
                window=self.trix_length), window=self.trix_length)
        
        self.trix_pct_line = self.trix_line.pct_change()*100

        if self.trix_signal_type == "sma":
            self.trix_signal_line = ta.trend.sma_indicator(
                close=self.trix_pct_line, window=self.trix_signal_length)
        elif self.trix_signal_type == "ema":
            self.trix_signal_line = ta.trend.ema_indicator(
                close=self.trix_pct_line, window=self.trix_signal_length)
            
        self.trix_histo = self.trix_pct_line - self.trix_signal_line

    def get_trix_line(self) -> pd.Series:
        return pd.Series(self.trix_line, name="trix_line")

    def get_trix_pct_line(self) -> pd.Series:
        return pd.Series(self.trix_pct_line, name="trix_pct_line")

    def get_trix_signal_line(self) -> pd.Series:
        return pd.Series(self.trix_signal_line, name="trix_signal_line")

    def get_trix_histo(self) -> pd.Series:
        return pd.Series(self.trix_histo, name="trix_histo")


class VMC():
    """ VuManChu Cipher B + Divergences 

        Args:
            high(pandas.Series): dataset 'High' column.
            low(pandas.Series): dataset 'Low' column.
            close(pandas.Series): dataset 'Close' column.
            wtChannelLen(int): n period.
            wtAverageLen(int): n period.
            wtMALen(int): n period.
            rsiMFIperiod(int): n period.
            rsiMFIMultiplier(int): n period.
            rsiMFIPosY(int): n period.
    """

    def __init__(
        self: pd.Series,
        open: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        wtChannelLen: int = 9,
        wtAverageLen: int = 12,
        wtMALen: int = 3,
        rsiMFIperiod: int = 60,
        rsiMFIMultiplier: int = 150,
        rsiMFIPosY: int = 2.5
    ) -> None:
        self._high = high
        self._low = low
        self._close = close
        self._open = open
        self._wtChannelLen = wtChannelLen
        self._wtAverageLen = wtAverageLen
        self._wtMALen = wtMALen
        self._rsiMFIperiod = rsiMFIperiod
        self._rsiMFIMultiplier = rsiMFIMultiplier
        self._rsiMFIPosY = rsiMFIPosY

        self._run()
        self.wave_1()

    def _run(self) -> None:
        self.hlc3 = (self._close + self._high + self._low)
        self._esa = ta.trend.ema_indicator(
            close=self.hlc3, window=self._wtChannelLen)
        self._de = ta.trend.ema_indicator(
            close=abs(self.hlc3 - self._esa), window=self._wtChannelLen)
        self._rsi = ta.trend.sma_indicator(self._close, self._rsiMFIperiod)
        self._ci = (self.hlc3 - self._esa) / (0.015 * self._de)

    def wave_1(self) -> pd.Series:
        """VMC Wave 1 

        Returns:
            pandas.Series: New feature generated.
        """
        wt1 = ta.trend.ema_indicator(self._ci, self._wtAverageLen)
        return pd.Series(wt1, name="wt1")

    def wave_2(self) -> pd.Series:
        """VMC Wave 2

        Returns:
            pandas.Series: New feature generated.
        """
        wt2 = ta.trend.sma_indicator(self.wave_1(), self._wtMALen)
        return pd.Series(wt2, name="wt2")

    def money_flow(self) -> pd.Series:
        """VMC Money Flow

        Returns:
            pandas.Series: New feature generated.
        """
        mfi = ((self._close - self._open) /
               (self._high - self._low)) * self._rsiMFIMultiplier
        rsi = ta.trend.sma_indicator(mfi, self._rsiMFIperiod)
        money_flow = rsi - self._rsiMFIPosY
        return pd.Series(money_flow, name="money_flow")


def heikinAshiDf(df):
    df['HA_Close'] = (df.open + df.high + df.low + df.close)/4
    ha_open = [(df.open[0] + df.close[0]) / 2]
    [ha_open.append((ha_open[i] + df.HA_Close.values[i]) / 2)
     for i in range(0, len(df)-1)]
    df['HA_Open'] = ha_open
    df['HA_High'] = df[['HA_Open', 'HA_Close', 'high']].max(axis=1)
    df['HA_Low'] = df[['HA_Open', 'HA_Close', 'low']].min(axis=1)
    return df

class SmoothedHeikinAshi():
    def __init__(self, open, high, low, close, smooth1=5, smooth2=3):
        self.open = open.copy()
        self.high = high.copy()
        self.low = low.copy()
        self.close = close.copy()
        self.smooth1 = smooth1
        self.smooth2 = smooth2
        self._run()

    def _calculate_ha_open(self):
        ha_open = pd.Series(np.nan, index=self.open.index)
        start = 0
        for i in range(1, len(ha_open)):
            if np.isnan(self.smooth_open.iloc[i]):
                continue
            else:
                ha_open.iloc[i] = (self.smooth_open.iloc[i] + self.smooth_close.iloc[i]) / 2
                start = i
                break

        for i in range(start + 1, len(ha_open)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + self.ha_close.iloc[i-1]) / 2

        return ha_open

    def _run(self):
        self.smooth_open = ta.trend.ema_indicator(self.open, self.smooth1)
        self.smooth_high = ta.trend.ema_indicator(self.high, self.smooth1)
        self.smooth_low = ta.trend.ema_indicator(self.low, self.smooth1)
        self.smooth_close = ta.trend.ema_indicator(self.close, self.smooth1)

        self.ha_close = (self.smooth_open + self.smooth_high + self.smooth_low + self.smooth_close) / 4
        self.ha_open = self._calculate_ha_open()
        

        self.smooth_ha_close = ta.trend.ema_indicator(self.ha_close, self.smooth2)
        self.smooth_ha_open = ta.trend.ema_indicator(self.ha_open, self.smooth2)
    
    def smoothed_ha_close(self):
        return self.smooth_ha_close
    def smoothed_ha_open(self):
        return self.smooth_ha_open


def add_technical_indicators(df):
    """Add multiple technical indicators to dataframe
    
    Args:
        df: DataFrame with OHLCV columns
        
    Returns:
        DataFrame with added technical indicators
    """
    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_hist'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bb_upper'] = bollinger.bollinger_hband()
    df['bb_middle'] = bollinger.bollinger_mavg()
    df['bb_lower'] = bollinger.bollinger_lband()
    
    # Volume indicators
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_anomaly'] = volume_anomality(df)
    # ATR (Average True Range)
    df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # OBV (On-Balance Volume)
    df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
    
    # SuperTrend (using the existing class)
    st = SuperTrend(high=df['high'], low=df['low'], close=df['close'], atr_window=10, atr_multi=3)
    df['supertrend_dir'] = st.super_trend_direction().astype(int) # True=1, False=0
    df['supertrend_lower'] = st.super_trend_lower()
    df['supertrend_upper'] = st.super_trend_upper()
    # Forward fill SuperTrend bands as they contain NaNs based on direction
    df['supertrend_lower'] = df['supertrend_lower'].ffill()
    df['supertrend_upper'] = df['supertrend_upper'].ffill()
    
    # --- Existing Volume Indicators ---
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_anomaly'] = volume_anomality(df) # Ensure this function is called after volume_ma if needed
    
    # Clean up potential NaNs introduced by indicators
    # It's often better to handle NaNs globally during the final cleaning step,
    # but we can do a basic ffill here for indicators that rely on previous values.
    df = df.fillna(method='ffill') 
    
    return df

def volume_anomality(df, volume_window=10):
    dfInd = df.copy()
    dfInd["VolAnomaly"] = 0
    dfInd["PreviousClose"] = dfInd["close"].shift(1)
    dfInd['MeanVolume'] = dfInd['volume'].rolling(volume_window).mean()
    dfInd['MaxVolume'] = dfInd['volume'].rolling(volume_window).max()
    dfInd.loc[dfInd['volume'] > 1.5 * dfInd['MeanVolume'], "VolAnomaly"] = 1
    dfInd.loc[dfInd['volume'] > 2 * dfInd['MeanVolume'], "VolAnomaly"] = 2
    dfInd.loc[dfInd['volume'] >= dfInd['MaxVolume'], "VolAnomaly"] = 3
    dfInd.loc[dfInd['PreviousClose'] > dfInd['close'],
              "VolAnomaly"] = (-1) * dfInd["VolAnomaly"]
    return dfInd["VolAnomaly"]

class SuperTrend():
    def __init__(
        self,
        high,
        low,
        close,
        atr_window=10,
        atr_multi=3
    ):
        self.high = high
        self.low = low
        self.close = close
        self.atr_window = atr_window
        self.atr_multi = atr_multi
        self._run()
        
    def _run(self):
        # calculate ATR
        price_diffs = [self.high - self.low, 
                    self.high - self.close.shift(), 
                    self.close.shift() - self.low]
        true_range = pd.concat(price_diffs, axis=1)
        true_range = true_range.abs().max(axis=1)
        # default ATR calculation in supertrend indicator
        atr = true_range.ewm(alpha=1/self.atr_window,min_periods=self.atr_window).mean() 
        # atr = ta.volatility.average_true_range(high, low, close, atr_period)
        # df['atr'] = df['tr'].rolling(atr_period).mean()
        
        # HL2 is simply the average of high and low prices
        hl2 = (self.high + self.low) / 2
        # upperband and lowerband calculation
        # notice that final bands are set to be equal to the respective bands
        final_upperband = upperband = hl2 + (self.atr_multi * atr)
        final_lowerband = lowerband = hl2 - (self.atr_multi * atr)
        
        # initialize Supertrend column to True
        supertrend = [True] * len(self.close)
        
        for i in range(1, len(self.close)):
            curr, prev = i, i-1
            
            # if current close price crosses above upperband
            if self.close[curr] > final_upperband[prev]:
                supertrend[curr] = True
            # if current close price crosses below lowerband
            elif self.close[curr] < final_lowerband[prev]:
                supertrend[curr] = False
            # else, the trend continues
            else:
                supertrend[curr] = supertrend[prev]
                
                # adjustment to the final bands
                if supertrend[curr] == True and final_lowerband[curr] < final_lowerband[prev]:
                    final_lowerband[curr] = final_lowerband[prev]
                if supertrend[curr] == False and final_upperband[curr] > final_upperband[prev]:
                    final_upperband[curr] = final_upperband[prev]

            # to remove bands according to the trend direction
            if supertrend[curr] == True:
                final_upperband[curr] = np.nan
            else:
                final_lowerband[curr] = np.nan
                
        self.st = pd.DataFrame({
            'Supertrend': supertrend,
            'Final Lowerband': final_lowerband,
            'Final Upperband': final_upperband
        })
        
    def super_trend_upper(self):
        return self.st['Final Upperband']
        
    def super_trend_lower(self):
        return self.st['Final Lowerband']
        
    def super_trend_direction(self):
        return self.st['Supertrend']
    
class MaSlope():
    """ Slope adaptative moving average
    """

    def __init__(
        self,
        close: pd.Series,
        high: pd.Series,
        low: pd.Series,
        long_ma: int = 200,
        major_length: int = 14,
        minor_length: int = 6,
        slope_period: int = 34,
        slope_ir: int = 25
    ):
        self.close = close
        self.high = high
        self.low = low
        self.long_ma = long_ma
        self.major_length = major_length
        self.minor_length = minor_length
        self.slope_period = slope_period
        self.slope_ir = slope_ir
        self._run()

    def _run(self):
        minAlpha = 2 / (self.minor_length + 1)
        majAlpha = 2 / (self.major_length + 1)
        # df = pd.DataFrame(data = [self.close, self.high, self.low], columns = ['close','high','low'])
        df = pd.DataFrame(data = {"close": self.close, "high": self.high, "low":self.low})
        df['hh'] = df['high'].rolling(window=self.long_ma+1).max()
        df['ll'] = df['low'].rolling(window=self.long_ma+1).min()
        df = df.fillna(0)
        df.loc[df['hh'] == df['ll'],'mult'] = 0
        df.loc[df['hh'] != df['ll'],'mult'] = abs(2 * df['close'] - df['ll'] - df['hh']) / (df['hh'] - df['ll'])
        df['final'] = df['mult'] * (minAlpha - majAlpha) + majAlpha

        ma_first = (df.iloc[0]['final']**2) * df.iloc[0]['close']

        col_ma = [ma_first]
        for i in range(1, len(df)):
            ma1 = col_ma[i-1]
            col_ma.append(ma1 + (df.iloc[i]['final']**2) * (df.iloc[i]['close'] - ma1))

        df['ma'] = col_ma
        pi = math.atan(1) * 4
        df['hh1'] = df['high'].rolling(window=self.slope_period).max()
        df['ll1'] = df['low'].rolling(window=self.slope_period).min()
        df['slope_range'] = self.slope_ir / (df['hh1'] - df['ll1']) * df['ll1']
        df['dt'] = (df['ma'].shift(2) - df['ma']) / df['close'] * df['slope_range'] 
        df['c'] = (1+df['dt']*df['dt'])**0.5
        df['xangle'] = round(180*np.arccos(1/df['c']) / pi)
        df.loc[df['dt']>0,"xangle"] = - df['xangle']
        self.df = df
        # print(df)

    def ma_line(self) -> pd.Series:
        """ ma_line

            Returns:
                pd.Series: ma_line
        """
        return self.df['ma']

    def x_angle(self) -> pd.Series:
        """ x_angle

            Returns:
                pd.Series: x_angle
        """
        return self.df['xangle']
