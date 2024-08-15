import pandas as pd
import pandas_ta as ta
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import List



# def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
#     """
#     Calculate technical indicators for OHLCV data.
    
#     Parameters:
#     data (pd.DataFrame): DataFrame containing OHLCV data with columns 'open', 'high', 'low', 'close', 'volume'.
    
#     Returns:
#     pd.DataFrame: DataFrame with additional columns for each calculated indicator.
#     """
#     # Ensure the required columns are present
#     required_columns = ['open', 'high', 'low', 'close']
#     if not all(column in data.columns for column in required_columns):
#         raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
#     # Calculate moving averages
#     data['SMA_20'] = ta.sma(data['close'], length=20)
#     data['SMA_50'] = ta.sma(data['close'], length=50)

#     # Calcular Media Móvil Exponencial (EMA)
#     data['EMA_14'] = ta.ema(data['close'], length=14)
    
#     # Calculate Relative Strength Index (RSI)
#     data['RSI'] = ta.rsi(data['close'], length=14)

#     # Calcular Stochastic Oscillator
#     #data[['STOCH_K', 'STOCH_D']] = ta.stoch(data['high'], data['low'], data['close'], k=14, d=3)
        
#     # Calculate Moving Average Convergence Divergence (MACD)
#     macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
#     data['MACD'] = macd['MACD_12_26_9']
#     data['MACD_Signal'] = macd['MACDs_12_26_9']
#     data['MACD_Diff'] = macd['MACDh_12_26_9']
    
#     # # Calculate Bollinger Bands
#     # bollinger = ta.bbands(data['close'], length=20)
#     # data['Bollinger_High'] = bollinger['BBU_20_2.0']
#     # data['Bollinger_Low'] = bollinger['BBL_20_2.0']
#     # data['Bollinger_Mid'] = bollinger['BBM_20_2.0']
    
#     # Calculate Average True Range (ATR)
#     data['ATR'] = ta.atr(data['high'], data['low'], data['close'], length=14)
    
#     # Calculate Volume Weighted Average Price (VWAP)
#     #data['VWAP'] = ta.vwap(data['high'], data['low'], data['close'], data['volume'])
    
#     # Calculate Stochastic Oscillator
#     # stoch = ta.stoch(data['high'], data['low'], data['close'])
#     # data['Stoch'] = stoch['STOCHk_14_3_3']
#     # data['Stoch_Signal'] = stoch['STOCHd_14_3_3']

#     # Calcular Positive Volume Index
#     #data['PVI'] = ta.pvi(data['close'], data['volume'])

#     # Calcular Negative Volume Index
#     #data['NVI'] = ta.nvi(data['close'], data['volume'])
    
#     # Calculate Commodity Channel Index (CCI)
#     data['CCI'] = ta.cci(data['high'], data['low'], data['close'], length=20)

#     # Calcular Nubes de Ichimoku
#     ichimoku = ta.ichimoku(data['high'], data['low'], data['close'])
#     data = data.join(ichimoku)

#     # Calcular Parabólico SAR (Stop and Reverse)
#     data['sar'] = ta.psar(data['high'], data['low'], data['close'])

#     # Calcular Índice de Movimiento Direccional (DMI)
#     dmi = ta.dmi(data['high'], data['low'], data['close'])
#     data = data.join(dmi)
    
#     # Calculate On-Balance Volume (OBV)
#     #data['OBV'] = ta.obv(data['close'], data['volume'])
    
#     # Calculate Hurst Exponent
#     #data['Hurst'] = data['close'].rolling(window=100).apply(lambda x: calculate_hurst_exponent(x), raw=False)
    
#     data.dropna(inplace=True)
#     return data

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for OHLCV data.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing OHLCV data with columns 'open', 'high', 'low', 'close', 'volume'.
    
    Returns:
    pd.DataFrame: DataFrame with additional columns for each calculated indicator.
    """
    # Ensure the required columns are present
    required_columns = ['open', 'high', 'low', 'close']
    if not all(column in data.columns for column in required_columns):
        raise ValueError(f"DataFrame must contain the following columns: {required_columns}")
    
    # Calculate moving averages
    data['SMA_20'] = ta.sma(data['close'], length=20)  # Simple Moving Average (20-period)
    data['SMA_50'] = ta.sma(data['close'], length=50)  # Simple Moving Average (50-period)

    # Calculate Exponential Moving Average (EMA)
    data['EMA_14'] = ta.ema(data['close'], length=14)  # Exponential Moving Average (14-period)

    # Calcular Bandas de Bollinger
    bb = ta.bbands(data['close'], length=20, std=2)
    data = data.join(bb)
    
    # Calculate Relative Strength Index (RSI)
    data['RSI'] = ta.rsi(data['close'], length=14)  # RSI (14-period)

    # Calculate Moving Average Convergence Divergence (MACD)
    macd = ta.macd(data['close'], fast=12, slow=26, signal=9)  # MACD (12,26,9)
    data['MACD'] = macd['MACD_12_26_9']
    data['MACD_Signal'] = macd['MACDs_12_26_9']
    data['MACD_Diff'] = macd['MACDh_12_26_9']
    
    # Calculate Average True Range (ATR)
    data['ATR'] = ta.atr(data['high'], data['low'], data['close'], length=14)  # ATR (14-period)
    
    # Calculate Commodity Channel Index (CCI)
    data['CCI'] = ta.cci(data['high'], data['low'], data['close'], length=20)  # CCI (20-period)

    # Calculate Nubes de Ichimoku
    # ichimoku = ta.ichimoku(data['high'], data['low'], data['close'])  # Ichimoku Cloud
    # data['ISA_9'] = ichimoku['ISA_9']
    # data['ISB_26'] = ichimoku['ISB_26']
    # data['ITS_9'] = ichimoku['ITS_9']
    # data['IKS_26'] = ichimoku['IKS_26']
    # data['ICS_26'] = ichimoku['ICS_26']

    # Calculate Parabólico SAR (Stop and Reverse)
    #data['SAR'] = ta.psar(data['high'], data['low'], data['close'])  # Parabolic SAR

    # Calculate Índice de Movimiento Direccional (DMI)
    # dmi = ta.dmi(data['high'], data['low'], data['close'])  # DMI
    # data = data.join(dmi)
    
    data.dropna(inplace=True)  # Drop rows with NaN values
    return data


def normalize_data(data: pd.DataFrame, exclude_columns: List[str] = []) -> pd.DataFrame:
    """
    Normalize the DataFrame columns, except for the specified columns.

    Parameters:
    data (pd.DataFrame): DataFrame with columns to normalize.
    exclude_columns (List[str]): List of column names to exclude from normalization.

    Returns:
    pd.DataFrame: Normalized DataFrame.
    """
    scaler = MinMaxScaler()

    # Select only numeric columns
    numeric_columns = data.select_dtypes(include=['float64']).columns

    # Exclude specified columns from normalization
    columns_to_normalize = [col for col in numeric_columns if col not in exclude_columns]

    # Apply scaling to the selected columns
    data[columns_to_normalize] = scaler.fit_transform(data[columns_to_normalize])

    return data

def filter_by_date(data: pd.DataFrame, cutoff_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter the DataFrame into two parts based on the cutoff date.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing a 'date' column.
    cutoff_date (str): The cutoff date in 'YYYY-MM-DD' format.
    
    Returns:
    tuple: Two DataFrames, one with dates before the cutoff date and one with dates on or after the cutoff date.
    """
    # Ensure the 'date' column is in datetime format
    data['date'] = pd.to_datetime(data['date'])
    
    # Define the cutoff date
    cutoff_date = pd.to_datetime(cutoff_date)
    
    # Filter the DataFrame
    before_cutoff = data[data['date'] < cutoff_date]
    after_cutoff = data[data['date'] >= cutoff_date]
    
    return before_cutoff, after_cutoff