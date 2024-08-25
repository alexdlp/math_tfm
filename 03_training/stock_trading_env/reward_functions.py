import numpy as np
import config
from .history import History


def sharpe_reward(history, risk_free_rate :float = 0.0):

    excess_returns = np.array(history['strategy_returns']) - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)

    if std_excess_return == 0:
        return 0.0
    else:
        return mean_excess_return / std_excess_return


def dantas_and_silva_reward(history, alpha: float = 1.5) -> float:
    """
    Calculate the reward for the reinforcement learning agent based on portfolio and stock price changes.

    Parameters:
    portfolio_value_t (float): Portfolio value at time t.
    portfolio_value_t_minus_1 (float): Portfolio value at time t-1.
    stock_price_t (float): Stock price at time t.
    stock_price_t_minus_1 (float): Stock price at time t-1.
    alpha (float): A constant that influences the reward.

    Returns:
    float: The calculated reward.
    """
    if len(history["portfolio_valuation"]) >= 2:
        alpha = config.ALPHA
        portfolio_value_t = history['portfolio_valuation', -1]
        portfolio_value_t_minus_1 = history['portfolio_valuation', -2]

        stock_price_t = history['stock_price', -1]
        stock_price_t_minus_1 = history['stock_price', -2]

        portfolio_change = (portfolio_value_t - portfolio_value_t_minus_1) / portfolio_value_t_minus_1
        stock_price_change = (stock_price_t - stock_price_t_minus_1) / stock_price_t_minus_1
        
        reward = alpha * portfolio_change - stock_price_change
        
        return reward
    else:
        return 0.0

def basic_reward_function(history:History, cash_penalty:float=-0.01):

    if len(history["portfolio_valuation"]>=2):

        # penalización por no operar
        if history['portfolio_exposition', -1] == 0:
            return cash_penalty
        else:
            return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])
    
    else:
        print('No hay suficientes muestras en la historia')
        return 0.0
    



def combined_penalty_reward(history: History, cash_penalty: float = -0.01, decay_rate: float = 0.005, opportunity_threshold: float = 0.02) -> float:
    """
    Reward function combining inactivity penalty and missed opportunity penalty.
    
    Parameters:
    history (History): The history of portfolio valuations and exposures.
    cash_penalty (float): Base penalty for not being exposed to the market.
    decay_rate (float): Incremental penalty for consecutive inactivity periods.
    opportunity_threshold (float): Threshold for significant market movements.

    Returns:
    float: The calculated reward.
    """
    if len(history["portfolio_valuation"]) >= 2:
        portfolio_exposition = history['portfolio_exposition', -1]
        stock_price_change = (history["stock_price", -1] - history["stock_price", -2]) / history["stock_price", -2]
        inactivity_periods = history.get('inactivity_periods', 0)

        if portfolio_exposition == 0:
            # Increase penalty over time for inactivity
            inactivity_periods += 1
            penalty = cash_penalty - (inactivity_periods * decay_rate)
            if abs(stock_price_change) > opportunity_threshold:
                # Additional penalty for missing a significant market movement
                penalty *= 2
            history['inactivity_periods'] = inactivity_periods
            return penalty
        else:
            # Reset inactivity periods if the agent takes action
            history['inactivity_periods'] = 0
            return np.log(history["portfolio_valuation", -1] / history["portfolio_valuation", -2])
    else:
        print('No hay suficientes muestras en la historia')
        return 0.0
    


def dantas_and_silva_reward_with_combined_penalty(history: dict, alpha: float, cash_penalty: float = -0.01, decay_rate: float = 0.005, position_holding_penalty: float = 0.01, reward_threshold: float = 0.001) -> float:
    """
    Reward function that calculates reward based on portfolio and stock price changes,
    with penalties for inactivity and prolonged position holding.
    
    Parameters:
    history (dict): Dictionary containing historical portfolio valuations and stock prices.
    alpha (float): A constant that influences the reward.
    cash_penalty (float): Penalty for not being exposed to the market.
    decay_rate (float): Incremental penalty for consecutive inactivity periods.
    position_holding_penalty (float): Penalty for holding a position without change.
    reward_threshold (float): Minimum return threshold above which holding penalty is not applied.

    Returns:
    float: The calculated reward.
    """
    if len(history["portfolio_valuation"]) >= 2:
        portfolio_value_t = history['portfolio_valuation'][-1]
        portfolio_value_t_minus_1 = history['portfolio_valuation'][-2]

        stock_price_t = history['stock_price'][-1]
        stock_price_t_minus_1 = history['stock_price'][-2]

        portfolio_change = (portfolio_value_t - portfolio_value_t_minus_1) / portfolio_value_t_minus_1
        stock_price_change = (stock_price_t - stock_price_t_minus_1) / stock_price_t_minus_1
        
        portfolio_exposition = history['portfolio_exposition'][-1]
        inactivity_periods = history.get('inactivity_periods', 0)
        position_holding_periods = history.get('position_holding_periods', 0)
        
        if portfolio_exposition == 0:
            # Penalización por inactividad (no operar)
            inactivity_periods += 1
            penalty = cash_penalty - (inactivity_periods * decay_rate)
            history['inactivity_periods'] = inactivity_periods
            return penalty
        else:
            # Calcular recompensa estándar
            reward = alpha * portfolio_change - stock_price_change

            # Penalizar si la posición se mantiene demasiado tiempo y no supera el umbral de recompensa
            if reward < reward_threshold:
                position_holding_periods += 1
                holding_penalty = -position_holding_periods * position_holding_penalty
            else:
                holding_penalty = 0  # No penalizar si la recompensa supera el umbral
            
            history['position_holding_periods'] = position_holding_periods

            # Resetear los periodos de inactividad si el agente toma acción
            history['inactivity_periods'] = 0

            # Combinar la recompensa con la penalización por mantener posición
            return reward + holding_penalty
    else:
        print('No hay suficientes muestras en la historia')
        return 0.0
