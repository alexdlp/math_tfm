from stable_baselines3 import A2C, PPO

from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv

from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import os
import numpy as np
import pandas as pd

from stock_trading_env import reward_functions
from stock_trading_env.trading_env import TradingEnv
import config

from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from torch.utils.tensorboard import SummaryWriter

gym.envs.register(
    id='TradingEnv-v0',
    entry_point='stock_trading_env.trading_env:TradingEnv',
)


def create_envs(df_path: str, n_envs: int, monitor_dir: str) -> DummyVecEnv:
    loaded_df = pd.read_parquet(df_path)

    if config.REWARD_FUNCTION == 'sharpe_ratio':
        reward_function = reward_functions.sharpe_reward
    elif config.REWARD_FUNCTION == 'dantas_and_silva_reward':
        reward_function = reward_functions.dantas_and_silva_reward

    def make_env(rank: int) -> gym.Env:
        def _init() -> gym.Env:
            env = gym.make(
                "TradingEnv-v0",
                name="BTCUSD",
                df=loaded_df,
                windows=config.WINDOW,
                positions=[-1, 0, 1],  # -1 (=SHORT), +1 (=LONG)
                initial_position=0,  # Initial position
                trading_fees=0.01 / 100,  # 0.01% per stock buy/sell
                reward_function=reward_function,
                portfolio_initial_value=100000,  # in FIAT (here, USD)
                max_episode_duration=500000,
            )
            env = TimeLimit(env, max_episode_steps=500000)  # Asegura el número máximo de pasos por episodio
            return env

        return _init

    # Crear los entornos vectorizados
    envs = DummyVecEnv([make_env(i) for i in range(n_envs)])

    # Crear el directorio de monitor si no existe
    os.makedirs(monitor_dir, exist_ok=True)
    # Asignar un único archivo de monitor
    monitor_file = os.path.join(monitor_dir, "monitor.csv")
    envs = VecMonitor(envs, filename=monitor_file)  # Monitoriza el entorno vectorizado

    return envs


def create_envsII(df_path: str, n_envs: int, monitor_dir: str) -> DummyVecEnv:

    loaded_df = pd.read_parquet(df_path)

    if config.REWARD_FUNCTION == 'sharpe_ratio':
        reward_function = reward_functions.sharpe_reward
    elif config.REWARD_FUNCTION == 'dantas_and_silva_reward':
        reward_function = reward_functions.dantas_and_silva_reward

    envs =  make_vec_env(lambda: TradingEnv(df=loaded_df,
                windows=config.WINDOW,
                positions=[-1, 0, 1],  # -1 (=SHORT), +1 (=LONG)
                initial_position=0,  # Initial position
                trading_fees=0.01 / 100,  # 0.01% per stock buy/sell
                reward_function=reward_function,
                portfolio_initial_value=100000,  # in FIAT (here, USD)
                max_episode_duration=500000), n_envs=n_envs)
    
    # Crear el directorio de monitor si no existe
    os.makedirs(monitor_dir, exist_ok=True)
    # Asignar un único archivo de monitor
    monitor_file = os.path.join(monitor_dir, "monitor.csv")
    envs = VecMonitor(envs, filename=monitor_file)  # Monitoriza el entorno vectorizado

    return envs


# Directorios para guardar los modelos y logs
model_save_path = "./logs/best_model"
log_path = "./logs/eval"
tensorboard_log = './tb_logs'

# Crear los directorios si no existen
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)
os.makedirs(tensorboard_log, exist_ok=True)


    
if __name__ == "__main__":

    monitor_dir = './monitor_logs'
      
    # Define the environment
    train_env = create_envs(df_path=config.TRAINING_DATA, n_envs=config.N_ENVS, monitor_dir=monitor_dir)

    if config.MODEL == 'A2C':

        # Define the model
        model = A2C(
            "MlpPolicy", train_env,
            learning_rate=config.LEARNING_RATE,
            gamma=config.GAMMA,
            #n_epochs=config.N_EPOCHS,
            verbose=1,
            tensorboard_log=tensorboard_log,
            device = 'cuda'
        )
    elif config.MODEL == 'PPO':
        model = PPO("MlpPolicy", train_env, 
            learning_rate=config.LEARNING_RATE,
            gamma=config.GAMMA,
            # n_steps = n_steps,
            # batch_size = batch_size,
            # ent_coef = ent_coef,
            # clip_range = clip_range,
            n_epochs = config.N_EPOCHS,
            verbose=1, 
            tensorboard_log = tensorboard_log,
            device = 'cuda'
            )

    # Callback for evaluation and early stopping
    # eval_env = create_envsII(df_path=config.DATAFRAME_PATH, n_envs=config.N_ENVS)
    # eval_callback = EvalCallback(
    #     eval_env,
    #     log_path=log_path,
    #     eval_freq=500,
    #     deterministic=True,
    #     render=False,
    #     n_eval_episodes=5
    # )

    # Entrena el modelo
    model.learn(total_timesteps=config.TOTAL_STEPS, 
                callback=[TensorboardCallback(), HParamCallback()],
                #callback=[eval_callback, EarlyStoppingCallback(check_freq = 350, patience = 10)],
                #progress_bar=True
                )
    
    model.save("ppo_cartpole")



