import numpy as np
import config

def sharpe_reward(history, risk_free_rate :float = 0.0):

    excess_returns = np.array(history['strategy_returns']) - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_excess_return = np.std(excess_returns)

    if std_excess_return == 0:
        return 0.0
    else:
        return mean_excess_return / std_excess_return


def dantas_and_silva_reward(history, alpha: float) -> float:
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
    alpha = config.ALPHA
    portfolio_value_t = history['portfolio_valuation', -1]
    portfolio_value_t_minus_1 = history['portfolio_valuation', -2]

    stock_price_t = history['stock_price', -1]
    stock_price_t_minus_1 = history['stock_price', -2]

    portfolio_change = (portfolio_value_t - portfolio_value_t_minus_1) / portfolio_value_t_minus_1
    stock_price_change = (stock_price_t - stock_price_t_minus_1) / stock_price_t_minus_1
    
    reward = alpha * portfolio_change - stock_price_change
    
    return reward