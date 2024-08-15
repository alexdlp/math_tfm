
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

TRAINING_DATA = SPY_VOLUME

REWARD_FUNCTION = 'sharpe_ratio' # dantas_and_silva_reward
if REWARD_FUNCTION == 'dantas_and_silva_reward':
    ALPHA = 1.5
N_ENVS = 1
LEARNING_RATE = 1e-3
GAMMA = 0.99
N_EPOCHS = 100
TOTAL_STEPS = 50_000_000
WINDOW = 10