
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam

import numpy as np
import config 

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

asset, data_type = extract_asset_and_data_type(config.TRAINING_DATA)
class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            'asset': asset,
            'data_type': data_type,
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
