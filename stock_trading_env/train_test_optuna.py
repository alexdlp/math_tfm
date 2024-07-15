

import optuna
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.env_util import make_vec_env
import gymnasium as gym
import os
import numpy as np
import pandas as pd

from trading_env2_alex import TradingEnv

from reward_functions import sharpe_reward, dantas_and_silva_reward

# class TensorboardLoggingCallback(BaseCallback):
#     def __init__(self, trial, log_dir, verbose=0):
#         super().__init__(verbose)
#         self.trial = trial
#         self.log_dir = log_dir
#         self.writer = SummaryWriter(log_dir=self.log_dir)

#     def _on_step(self) -> bool:
#         # Registro de recompensas
#         self.writer.add_scalar("reward", self.locals["reward"], self.num_timesteps)
        
#         # Ejemplo para registrar pérdida (si está disponible)
#         if 'loss' in self.locals:
#             self.writer.add_scalar("loss", self.locals["loss"], self.num_timesteps)

#         # Añadir aquí más registros según las métricas disponibles

#         return True

#     def _on_training_end(self):
#         self.writer.close()


class HParamCallback(BaseCallback):
    """
    Saves the hyperparameters and metrics at the start of the training, and logs them to TensorBoard.
    """

    def _on_training_start(self) -> None:
        hparam_dict = {
            "algorithm": self.model.__class__.__name__,
            "learning rate": self.model.learning_rate,
            "gamma": self.model.gamma,
            "n_envs": self.model.n_envs,
            "n_epochs": self.model.n_epochs,
   
        }
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

trading_days = 252 * 36
trading_cost_bps = 1e-3
time_cost_bps = 1e-4

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



def create_envs(df_path:str, n_envs):

    loaded_df = pd.read_parquet(df_path)

    env = gym.make(
        "TradingEnv",
        name= "BTCUSD",
        df = loaded_df,
        windows= 5,
        positions = [-1, 0, 1],  # -1 (=SHORT), +1 (=LONG)
        initial_position = 0, #Initial position
        trading_fees = 0.01/100, # 0.01% per stock buy / sell
        borrow_interest_rate = None,
        #borrow_interest_rate= 0.0003/100, #per timestep (= 1h here)
        reward_function = sharpe_reward,
        portfolio_initial_value = 10000, # in FIAT (here, USD)
        max_episode_duration = 500000,
    )


    # return make_vec_env(lambda: TradingEnv(ticker='AAPL',
    #                     start="1980-12-12", 
    #                     end="2021-12-31",
    #                     trading_days=trading_days,
    #                     trading_cost_bps=trading_cost_bps,
    #                     time_cost_bps=time_cost_bps), n_envs=n_envs)
    return env


# Directorios para guardar los modelos y logs
model_save_path = "./logs/best_model"
log_path = "./logs/eval"
tensorboard_log = './tb_logs'

# Crear los directorios si no existen
os.makedirs(model_save_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)
os.makedirs(tensorboard_log, exist_ok=True)

# Función objetivo para la optimización con Optuna
def objective(trial):
    # Define el espacio de búsqueda
    #batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    # n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 2048*2])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log = True)
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    n_envs = trial.suggest_int("n_envs", 1, 16)
    # ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    # clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    
    
    # Define el entorno
    train_env = create_envs(n_envs= n_envs)

    # Define el modelo
    model = A2C("MlpPolicy", train_env, 
                learning_rate = learning_rate, 
                gamma = gamma,
                # n_steps = n_steps,
                # batch_size = batch_size,
                # ent_coef = ent_coef,
                # clip_range = clip_range,
                n_epochs = n_epochs,
                verbose=1, 
                tensorboard_log = tensorboard_log)

    # Callback para evaluación y Early Stopping
    eval_env = create_envs(n_envs= 1)
    eval_callback = EvalCallback(eval_env, 
                                log_path=log_path, eval_freq=500,
                                deterministic=True, render=False,
                                n_eval_episodes=5)


    # Entrena el modelo
    model.learn(total_timesteps=35_000, callback=[eval_callback, EarlyStoppingCallback(check_freq = 350, patience = 10)])

    # Evaluar y devolver la métrica objetivo
    total_rewards = eval_callback.best_mean_reward
    return total_rewards


if __name__ == "__main__":

    # Crear los directorios si no existen
    os.makedirs(model_save_path, exist_ok=True)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(tensorboard_log, exist_ok=True)
    # Configuración del estudio de Optuna con pruner
    study = optuna.create_study(direction='maximize', pruner=optuna.pruners.SuccessiveHalvingPruner())


    study.optimize(objective, n_trials=100)

    # Resultados
    print('Número de trials terminados: ', len(study.trials))
    print('Mejor trial:')
    trial = study.best_trial

    print('  Valor: ', trial.value)
    print('  Parámetros: ')
    for key, value in trial.params.items():
        print(f'    {key}: {value}')
