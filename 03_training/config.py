import re
from typing import Tuple


def extract_asset_and_data_type(file_path: str) -> Tuple[str, str]:
    """
    Extracts the asset and data type from a parquet file path.
    
    Parameters:
    file_path (str): The path of the parquet file.
    
    Returns:
    Tuple[str, str]: A tuple containing the asset ('BTC' or 'SPY') and data type ('original', 'volume', 'dollar').
    """
    # Regular expressions to match the asset and data type
    asset_pattern = re.compile(r'(BTC|SPY)')
    data_type_pattern = re.compile(r'(original|volume|dollar)')
    
    # Extract the asset
    asset_match = asset_pattern.search(file_path)
    asset = asset_match.group(1) if asset_match else None
    
    # Extract the data type
    data_type_match = data_type_pattern.search(file_path)
    data_type = data_type_match.group(1) if data_type_match else None
    
    if asset is None or data_type is None:
        raise ValueError("Unable to extract asset and data type from the file path.")
    
    return asset, data_type

# dantas and silva reward alpha
ALPHA = 1.5

MODEL = 'PPO' # A2C

# SPY (IN_SAMPLE)
SPY_ORIGINAL = r'C:\Users\adelapuente\Desktop\math_tfm\02_training_data\SPY_original_processed_in_sample.parquet'
SPY_VOLUME = r'C:\Users\adelapuente\Desktop\math_tfm\02_training_data\SPY_volume_processed_in_sample.parquet'
SPY_DOLLAR = r'C:\Users\adelapuente\Desktop\math_tfm\02_training_data\SPY_dollar_processed_in_sample.parquet'

# BTC (IN-SAMPLE)
BTC_ORIGINAL = r'C:\Users\adelapuente\Desktop\math_tfm\02_training_data\BTC_original_processed_in_sample.parquet'
BTC_VOLUME = r'C:\Users\adelapuente\Desktop\math_tfm\02_training_data\BTC_volume_processed_in_sample.parquet'
BTC_DOLLAR = r'C:\Users\adelapuente\Desktop\math_tfm\02_training_data\BTC_dollar_processed_in_sample.parquet'

# SPY (IN_SAMPLE)-SIN OHLC
SPY_ORIGINAL_SIN_OHLC = r'C:\Users\adelapuente\Desktop\math_tfm\02_training_data\SPY_original_processed_in_sample_sin_ohlc.parquet'
SPY_VOLUME_SIN_OHLC = r'C:\Users\adelapuente\Desktop\math_tfm\02_training_data\SPY_volume_processed_in_sample_sin_ohlc.parquet'
SPY_DOLLAR_SIN_OHLC = r'C:\Users\adelapuente\Desktop\math_tfm\02_training_data\SPY_dollar_processed_in_sample_sin_ohlc.parquet'

# BTC (IN-SAMPLE)-SIN OHLC
BTC_ORIGINAL_SIN_OHLC = r'C:\Users\adelapuente\Desktop\math_tfm\02_training_data\BTC_original_processed_in_sample_sin_ohlc.parquet'
BTC_VOLUME_SIN_OHLC = r'C:\Users\adelapuente\Desktop\math_tfm\02_training_data\BTC_volume_processed_in_sample_sin_ohlc.parquet'
BTC_DOLLAR_SIN_OHLC = r'C:\Users\adelapuente\Desktop\math_tfm\02_training_data\BTC_dollar_processed_in_sample_sin_ohlc.parquet'

TRAINING_DATA = BTC_DOLLAR

REWARD_FUNCTION = 'dantas_and_silva_reward' # basic dantas_and_silva_reward sharpe_ratio
if REWARD_FUNCTION == 'dantas_and_silva_reward':
    ALPHA = 1.5
N_ENVS = 1
LEARNING_RATE = 1e-3
GAMMA = 0.99
N_EPOCHS = 100
TOTAL_STEPS = 1_500_000
WINDOW = 10
PORTFOLIO_INITIAL_VALUE = 100_000

ASSET, DATA_TYPE = extract_asset_and_data_type(TRAINING_DATA)