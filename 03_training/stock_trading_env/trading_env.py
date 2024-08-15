import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import numpy as np

from .history import History
from .simple_portfolio import SimplePortfolio

import warnings
warnings.filterwarnings("error")

def basic_reward_function(history : History):
    return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])

def dynamic_feature_last_action_taken(history):
    return history['agent_action', -1]

def dynamic_feature_asset_quantity(history):
    return history['portfolio_asset_quantity', -1]

def dynamic_feature_real_action_taken(history):
    return history['real_action_taken', -1]

class TradingEnv(gym.Env):
    """
    An easy trading environment for OpenAI gym. It is recommended to use it this way :

    .. code-block:: python

        import gymnasium as gym
        import gym_trading_env
        env = gym.make('TradingEnv', ...)


    :param df: The market DataFrame. It must contain 'open', 'high', 'low', 'close'. Index must be DatetimeIndex. Your desired inputs need to contain 'feature' in their column name : this way, they will be returned as observation at each step.
    :type df: pandas.DataFrame

    :param positions: List of the positions allowed by the environment.
    :type positions: optional - list[int or float]

    :param dynamic_feature_functions: The list of the dynamic features functions. By default, two dynamic features are added :
    
        * the last position taken by the agent.
        * the real position of the portfolio (that varies according to the price fluctuations)

    :type dynamic_feature_functions: optional - list   

    :param reward_function: Take the History object of the environment and must return a float.
    :type reward_function: optional - function<History->float>

    :param windows: Default is None. If it is set to an int: N, every step observation will return the past N observations. It is recommended for Recurrent Neural Network based Agents.
    :type windows: optional - None or int

    :param trading_fees: Transaction trading fees (buy and sell operations). eg: 0.01 corresponds to 1% fees
    :type trading_fees: optional - float

    :param borrow_interest_rate: Borrow interest rate per step (only when position < 0 or position > 1). eg: 0.01 corresponds to 1% borrow interest rate per STEP ; if your know that your borrow interest rate is 0.05% per day and that your timestep is 1 hour, you need to divide it by 24 -> 0.05/100/24.
    :type borrow_interest_rate: optional - float

    :param portfolio_initial_value: Initial valuation of the portfolio.
    :type portfolio_initial_value: float or int

    :param initial_position: You can specify the initial position of the environment or set it to 'random'. It must contained in the list parameter 'positions'.
    :type initial_position: optional - float or int

    :param max_episode_duration: If a integer value is used, each episode will be truncated after reaching the desired max duration in steps (by returning `truncated` as `True`). When using a max duration, each episode will start at a random starting point.
    :type max_episode_duration: optional - int or 'max'

    :param verbose: If 0, no log is outputted. If 1, the env send episode result logs.
    :type verbose: optional - int
    
    :param name: The name of the environment (eg. 'BTC/USDT')
    :type name: optional - str
    
    """
    # vamos a ir linea por linea
    # render ni puta idea para que es esto
    metadata = {'render_modes': ['logs']}
    # constructor
    def __init__(self,
                 # entrada del dataframe, el espacio que el agente debera recorrer.
                df : pd.DataFrame,
                # las posiciones no se si son lo mismo que las acctiones. ental caso, habra que modificar esto.
                positions : list = [0, 1],
                # funciones dinámicas, cogen el objeto historia y computan algo con el 
                dynamic_feature_functions = [dynamic_feature_last_action_taken, dynamic_feature_real_action_taken, dynamic_feature_asset_quantity],
                # funcion de recompensa. lo mismo pero para el reward
                reward_function = basic_reward_function,
                # window, cada step devolvera un array de window x col_length
                windows = None,
                # costes de la operacion
                trading_fees = 0,
                # valor inicial del portfolio. ya veremos esto como lo hacemos
                portfolio_initial_value = 1000,
                # la posición inicial seguramente la cambiemos; sera siempre 0 (sin ningun activo)
                initial_position ='random',
                # cuantos steps maximos dura un episodio
                max_episode_duration = 'max',
                # pues el verbose
                verbose = 1,
                name = "Stock",
                render_mode= "logs"
                ):
        
        # asignamos las variables a 
        self.max_episode_duration = max_episode_duration
        self.name = name
        self.verbose = verbose

        self.positions = positions
        self.dynamic_feature_functions = dynamic_feature_functions
        self.reward_function = reward_function
        self.windows = windows
        self.trading_fees = trading_fees
       
        self.portfolio_initial_value = float(portfolio_initial_value)
        self.initial_position = initial_position

        # esto tambien lo podemos cambiar en el futuro
        assert self.initial_position in self.positions or self.initial_position == 'random', "The 'initial_position' parameter must be 'random' or a position mentionned in the 'position' (default is [0, 1]) parameter."
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.max_episode_duration = max_episode_duration
        self.render_mode = render_mode
        self._set_df(df)
        
        # definir el espacio de acciones
        self.action_space = spaces.Discrete(len(positions))

        # definir el espacio de observaciones
        self.observation_space = spaces.Box(-np.inf, np.inf,shape = [self._nb_features])
        if self.windows is not None:
            self.observation_space = spaces.Box(
                -np.inf,
                np.inf,
                shape = [self.windows, self._nb_features]
            )
        

        self.log_metrics = []

    # metodo para inicializar el dataframe de entrada
    def _set_df(self, df):

        # copia el dataframe
        df = df.copy()

        # esto esta pensado para las features definidas por el usuario. no se si lo voy a dejar
        self._features_columns = [col for col in df.columns if "feature" in col]

        # Aquí se crea una lista de columnas que incluye todas las columnas del DataFrame df 
        # más la columna "close", excluyendo las columnas que ya están en self._features_columns. 
        # El resultado se asigna a self._info_columns
        self._info_columns = list(set(list(df.columns) + ["close"]) - set(self._features_columns))

        # cuantas características tengo
        self._nb_features = len(self._features_columns)
        self._nb_static_features = self._nb_features

        # aa vale. esto son las features dinámicas que se van creando a medida que el agente recorre el espacio.
        # Este bucle itera sobre el rango del número de funciones dinámicas (self.dynamic_feature_functions). 
        # Para cada iteración, agrega una nueva columna al DataFrame df llamada dynamic_feature__{i}, inicializándola con ceros. 
        # Luego, agrega el nombre de esta columna a self._features_columns y aumenta el contador de características (self._nb_features) en 1.
        for i  in range(len(self.dynamic_feature_functions)):
            df[f"dynamic_feature__{i}"] = 0
            self._features_columns.append(f"dynamic_feature__{i}")
            self._nb_features += 1

        self.df = df
        self._obs_array = np.array(self.df[self._features_columns], dtype= np.float32)
        self._info_array = np.array(self.df[self._info_columns])
        self._price_array = np.array(self.df["close"])


    ## NO SE SI LOS UTLIZAREMOS
    def _get_ticker(self, delta = 0):
        return self.df.iloc[self._idx + delta]
    ## NO SE SI LOS UTLIZAREMOS
    def _get_price(self, delta = 0):
        return self._price_array[self._idx + delta]
    
    def _get_obs(self):

        # SOLO DEVUELVE LOS INDICADORES CREADOS + LOS DINAMICOS
        
        # CALCULA LAS POSICIONES DINAMICAS PARA CADA STEP
        for i, dynamic_feature_function in enumerate(self.dynamic_feature_functions):
            self._obs_array[self._idx, self._nb_static_features + i] = dynamic_feature_function(self.historical_info)

        # if self.windows is None:
        #     _step_index = self._idx
        # else: 
        #     _step_index = np.arange(self._idx + 1 - self.windows , self._idx + 1)
        # return self._obs_array[_step_index]
    
        if self.windows:
            _step_index = np.arange(self._idx + 1 - self.windows, self._idx + 1)

        else:
            _step_index = self._idx 

        return self._obs_array[_step_index]

    def _set_start_index(self):
        """
        Initialize the starting index for the episode based on the provided parameters.
        """
        self._idx = 0

        # Set initial index based on windows if provided
        if self.windows is not None:
            self._idx = self.windows - 1

        # Adjust the index randomly if max_episode_duration is not 'max'
        if self.max_episode_duration != 'max':
            # Calculate the high limit for the random index to avoid exceeding DataFrame bounds
            max_index = len(self.df) - self.max_episode_duration
            if max_index > self._idx:
                self._idx = np.random.randint(
                    low=self._idx, 
                    high=max_index
                )
            else:
                # Handle the case where the max_index is not greater than _idx
                raise ValueError("max_episode_duration is too large for the DataFrame length.")

    def reset(self, seed = None, options=None):
        super().reset(seed = seed)
        
        self._step = 0
        self._set_start_index()
        self._agent_action = np.random.choice(self.positions) if self.initial_position == 'random' else self.initial_position
    

        self._portfolio  = SimplePortfolio(
            initial_cash = self.portfolio_initial_value,
            trading_fees= self.trading_fees
        )
        
        self.historical_info = History(max_size= len(self.df))

        self.historical_info.set(
            idx = self._idx,
            step = self._step,
            date = self.df.index.values[self._idx],
            agent_action = self._agent_action,
            real_action_taken = self._agent_action,
            portfolio_asset_quantity = 0,
            data =  dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation = self.portfolio_initial_value,
            portfolio_distribution = self._portfolio.get_portfolio_distribution(),
            reward = 0,
            strategy_returns = 0.0,
        )


        return self._get_obs(), self.historical_info[0]

    def render(self):
        pass

    def _trade(self, action, price = None):

        step_return, real_action_taken, asset = self._portfolio.trade(
            action = action, 
            price = self._get_price() if price is None else price, 
        )

        # actualizo la accion del agente y la accion real tomada.
        self._agent_action = action
        self._real_action_taken = real_action_taken

        # actualizamos step return y el asset
        self._step_return = step_return
        self._portfolio_asset_quantity = asset
        

    # def _take_action(self, action):
    #     # si la posicion anterior no es la misma, ejecutamos trade
    #     # para nosotros, la posicion sera estar comprado o vendido

    #     # esto se gestiona actualmente en el portfolio. se podria quitar.
    #     if action != self._position:
    #         step_ret = self._trade(action)


    def step(self, position_index = None):

        # vale esto está pensado para que, en caso de que la posicion no sea 0, el entorno haga algo.
        
        if position_index is not None: 
            self._trade(self.positions[position_index])

        self._idx += 1
        self._step += 1

        stock_price = self._get_price()
        portfolio_value = self._portfolio.get_portfolio_valuation(stock_price)
        portfolio_distribution = self._portfolio.get_portfolio_distribution()

        done, truncated = False, False

        if portfolio_value <= 0:
            done = True
        if self._idx >= len(self.df) - 1:
            truncated = True
        if isinstance(self.max_episode_duration,int) and self._step >= self.max_episode_duration - 1:
            truncated = True

        self.historical_info.add(
            idx = self._idx,
            step = self._step,
            date = self.df.index.values[self._idx],
            agent_action = self._agent_action,
            real_action_taken = self._real_action_taken,
            portfolio_asset_quantity = self._portfolio_asset_quantity,
            #real_position = self._portfolio.get_portfolio_position(stock_price),
            data =  dict(zip(self._info_columns, self._info_array[self._idx])),
            portfolio_valuation = portfolio_value,
            portfolio_distribution = portfolio_distribution, 
            reward = 0, # se actualiza después
            strategy_returns = self._step_return
        )

        if not done:
            reward = self.reward_function(self.historical_info)
            self.historical_info["reward", -1] = reward

        if done or truncated:
            self.calculate_metrics()
            self.log()

        return self._get_obs(),  self.historical_info["reward", -1], done, truncated, self.historical_info[-1]

    def add_metric(self, name, function):
        self.log_metrics.append({
            'name': name,
            'function': function
        })
    def calculate_metrics(self):
        self.results_metrics = {
            "Market Return" : f"{100*(self.historical_info['data_close', -1] / self.historical_info['data_close', 0] -1):5.2f}%",
            "Portfolio Return" : f"{100*(self.historical_info['portfolio_valuation', -1] / self.historical_info['portfolio_valuation', 0] -1):5.2f}%",
        }

        for metric in self.log_metrics:
            self.results_metrics[metric['name']] = metric['function'](self.historical_info)

    def get_metrics(self):
        return self.results_metrics
    
    def log(self):
        if self.verbose > 0:
            text = ""
            for key, value in self.results_metrics.items():
                text += f"{key} : {value}   |   "
            print(text)

