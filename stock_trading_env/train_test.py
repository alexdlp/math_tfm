

import optuna
from stable_baselines3 import A2C, PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor, DummyVecEnv

from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
import os
import numpy as np
import pandas as pd

from trading_env2_alex import TradingEnv
import config

from reward_functions import sharpe_reward, dantas_and_silva_reward
from gymnasium.wrappers import RecordEpisodeStatistics, TimeLimit
from torch.utils.tensorboard import SummaryWriter

from trading_env2_alex import TradingEnv
gym.envs.register(
    id='TradingEnv-v0',
    entry_point='trading_env2_alex:TradingEnv',
)

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_rewards = []
        self.episode_portfolio_valuations = []
        self.episode_strategy_returns = []



    def _on_step(self) -> bool:

        # print(self.locals)
        # import time
        # time.sleep(30)
        info = self.locals['infos'][0]
        reward = info.get('reward', 0.0)
        portfolio_valuation = info.get('portfolio_valuation', 0.0)
        strategy_returns = info.get('strategy_returns', 0.0)
        action_taken = info.get('action_taken')
        self.logger.record('action_taken', action_taken)

        # Gather episode rewards and other metrics
        self.episode_rewards.append(reward)
        self.episode_portfolio_valuations.append(portfolio_valuation)
        self.episode_strategy_returns.append(strategy_returns)

        # Log metrics every 1000 steps
        if self.n_calls % 1000 == 0:
            mean_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0
            mean_portfolio_valuation = np.mean(self.episode_portfolio_valuations) if self.episode_portfolio_valuations else 0
            mean_strategy_returns = np.mean(self.episode_strategy_returns) if self.episode_strategy_returns else 0

            self.logger.record('reward', mean_reward)
            self.logger.record('portfolio_valuation', mean_portfolio_valuation)
            self.logger.record('strategy_returns', mean_strategy_returns)

            # Calculate and log rolling metrics
            if len(self.episode_rewards) >= 100:
                rolling_reward = np.mean(self.episode_rewards[-100:])
                self.logger.record('rolling_reward', rolling_reward, self.num_timesteps)

            if len(self.episode_portfolio_valuations) >= 100:
                rolling_portfolio_valuation = np.mean(self.episode_portfolio_valuations[-100:])
                self.logger.record('rolling_portfolio_valuation', rolling_portfolio_valuation, self.num_timesteps)

            if len(self.episode_strategy_returns) >= 100:
                rolling_strategy_returns = np.mean(self.episode_strategy_returns[-100:])
                self.logger.record('rolling_strategy_returns', rolling_strategy_returns, self.num_timesteps)

            # Reset the lists of episode metrics
            self.episode_rewards = []
            self.episode_portfolio_valuations = []
            self.episode_strategy_returns = []

        return True




class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "n_envs": self.model.get_env().num_envs,
            'reward_function': config.REWARD_FUNCTION,
            'total_timesteps': config.TOTAL_STEPS
            #"n_epochs": self.model.n_epochs,
        }
        if config.REWARD_FUNCTION == 'dantas_and_silva_reward':
            hparam_dict.update({'alpha':config.ALPHA})
        # define the metrics that will appear in the `HPARAMS` Tensorboard tab by referencing their tag
        # Tensorbaord will find & display metrics from the `SCALARS` tab
        metric_dict = {
            "rollout/ep_len_mean": 0,
            "train/value_loss": 0.0,
        }
        self.logger.record(
            "hparams",
            HParam(hparam_dict, metric_dict),
            exclude=("stdout", "log", "json", "csv"),
        )

    def _on_step(self) -> bool:
        return True


class EarlyStoppingCallback(BaseCallback):
    def __init__(self, check_freq, patience, verbose=1):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.patience = patience
        self.best_mean_reward = -float('inf')
        self.patience_counter = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Suponiendo que estamos usando un entorno con un método de recompensa
            # Esto puede variar según el entorno y el problema
            mean_reward = np.mean([self.model.rollout_buffer.rewards[i] for i in range(self.model.rollout_buffer.size())])
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if self.patience_counter > self.patience:
                print("Early stopping: no improvement for {} checks".format(self.patience))
                return False  # Retorna False para detener el entrenamiento
        return True  # Retorna True para continuar el entrenamiento



# def create_envs(df_path: str, n_envs: int, monitor_dir: str) -> gym.Env:

#     loaded_df = pd.read_parquet(df_path)

    

#     env = gym.make(
#         "TradingEnv-v0",
#         name= "BTCUSD",
#         df = loaded_df,
#         windows= 5,
#         positions = [-1, 0, 1],  # -1 (=SHORT), +1 (=LONG)
#         initial_position = 0, #Initial position
#         trading_fees = 0.01/100, # 0.01% per stock buy / sell
#         #borrow_interest_rate = None,
#         #borrow_interest_rate= 0.0003/100, #per timestep (= 1h here)
#         reward_function = sharpe_reward,
#         portfolio_initial_value = 100000, # in FIAT (here, USD)
#         max_episode_duration = config.MAX_EPISODE_LENGTH,
#     )

#     # return make_vec_env(lambda: TradingEnv(ticker='AAPL',
#     #                     start="1980-12-12", 
#     #                     end="2021-12-31",
#     #                     trading_days=trading_days,
#     #                     trading_cost_bps=trading_cost_bps,
#     #                     time_cost_bps=time_cost_bps), n_envs=n_envs)
#     env = RecordEpisodeStatistics(env)
#     env = TimeLimit(env, max_episode_steps=500000)  # Ensure to set max_episode_steps

#     return env


def create_envs(df_path: str, n_envs: int, monitor_dir: str) -> DummyVecEnv:
    loaded_df = pd.read_parquet(df_path)

    if config.REWARD_FUNCTION == 'sharpe_ratio':
        reward_function = sharpe_reward
    elif config.REWARD_FUNCTION == 'dantas_and_silva_reward':
        reward_function = dantas_and_silva_reward

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
        reward_function = sharpe_reward
    elif config.REWARD_FUNCTION == 'dantas_and_silva_reward':
        reward_function = dantas_and_silva_reward

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
    train_env = create_envs(df_path=config.DATAFRAME_PATH, n_envs=config.N_ENVS, monitor_dir=monitor_dir)

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



