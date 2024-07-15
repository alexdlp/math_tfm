
# dantas and silva reward alpha
ALPHA = 1.5

MODEL = 'PPO' # A2C

DATAFRAME_PATH = r'C:\Users\adelapuente\Desktop\rf_tfm\data\PROCESSED_BTC_1min_2018-01-01_to_2023-12-31.parquet'

REWARD_FUNCTION = 'sharpe_ratio' # dantas_and_silva_reward
if REWARD_FUNCTION == 'dantas_and_silva_reward':
    ALPHA = 1.5
N_ENVS = 1
LEARNING_RATE = 1e-3
GAMMA = 0.99
N_EPOCHS = 10
TOTAL_STEPS = 50_000
WINDOW = 5