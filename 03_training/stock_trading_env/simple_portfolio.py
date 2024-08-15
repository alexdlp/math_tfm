import numpy as np

class SimplePortfolio:
    def __init__(self, initial_cash: float = 100000, trading_fees: float = 0.001, max_asset: int = 1):
        """
        Initialize the portfolio with given asset, cash amounts, and trading fees.

        :param asset: Amount of asset held in the portfolio.
        :param cash: Amount of cash held in the portfolio.
        :param trading_fees: Trading fees as a fraction of the trade amount.
        """
        self.asset = 0
        self.max_asset = max_asset
        self.cash = initial_cash
        self.initial_cash = initial_cash
        self.trading_fees = trading_fees
        self.portfolio_value_history = []
        self.pnl_history = []

    def get_portfolio_valuation(self, price: float) -> float:
        """
        Calculate the total value of the portfolio based on the current price of the asset.

        :param price: Current price of the asset.
        :return: Total value of the portfolio.
        """
        return self.asset * price + self.cash
    
    def _update_portfolio(self, price):

        # update portfolio valuation
        portfolio_value = self.get_portfolio_valuation(price)
        self.portfolio_value_history.append(portfolio_value)
        #print('PORT VALUE:', portfolio_value)
        # update portfolio PnL
        current_pnl = portfolio_value - self.initial_cash
        self.pnl_history.append(current_pnl)
        
        # Calcular el retorno del portafolio para este paso
        if len(self.portfolio_value_history) > 1:
            step_return = (self.portfolio_value_history[-1] - self.portfolio_value_history[-2]) / self.portfolio_value_history[-2]
            log_step_return = np.log(step_return+1)
        else:
            step_return = 0.0

        return step_return
    def trade(self, action: int, price: float) -> float:
        #print('PRICE',price)
        """
        Execute a trade based on the given action, update the portfolio's value and PnL, and return the portfolio's return.

        :param action: The trading action to perform. Action can be -1 (sell), 0 (hold), or 1 (buy).
        :param price: The current price of the asset.
        :return: The portfolio's return for this step.
        """
        print_buy = False
        print_sell = False
        # no se toma ninguna accion
        real_action_taken = 0
        # a menos que se cumplan estas historias.
        if action == 1 and self.asset < self.max_asset:  # Buy one unit
            
            cost = price * (1 + self.trading_fees)
            if self.cash >= cost:
                
                self.asset += 1
                self.cash -= cost
                real_action_taken = 1
                print_buy = True
                
        elif action == -1:  # Sell one unit
            if self.asset >= 1:
                
                revenue = price * (1 - self.trading_fees)
                self.asset -= 1
                self.cash += revenue
                real_action_taken = -1
                print_sell = True               
        
        step_return = self._update_portfolio(price)

        if print_buy:
            print(f'BUYING --> \tprice:{price} | \tcost: {cost} | port: {self.portfolio_value_history[-1]} | port-1: {self.portfolio_value_history[-2]} | asset : {self.asset}')
        if print_sell:
            print(f'SELLING --> \tprice:{price} | \tprofit: {revenue} | port: {self.portfolio_value_history[-1]} | port-1: {self.portfolio_value_history[-2]} | asset : {self.asset}')
        
        return step_return, real_action_taken, self.asset


    def __str__(self) -> str:
        """
        String representation of the portfolio.

        :return: String representation of the portfolio.
        """
        return f"SimplePortfolio(asset={self.asset}, cash={self.cash})"

    def describe(self, price: float):
        """
        Print the current value and position of the portfolio.

        :param price: Current price of the asset.
        """
        print("Value: ", self.valorisation(price), "Position: ", self.position())

    def get_portfolio_distribution(self) -> dict:
        """
        Get the current distribution of the portfolio in terms of assets and cash.

        :return: A dictionary with the current distribution of the portfolio.
        """
        return {
            "asset": self.asset,
            "cash": self.cash
        }
    
    def get_portfolio_position(self, price: float) -> float:
        """
        Calculate the position of the portfolio.

        :param price: Current price of the asset.
        :return: Position of the portfolio.
        """
        return self.asset * price / self.get_portfolio_valuation(price)
    
    def get_portfolio_value_history(self):
        return self.portfolio_value_history

    def get_pnl_history(self):
        return self.pnl_history

    def get_step_return(self):
        """
        Calculate the daily returns based on the portfolio value history.

        :return: Numpy array of daily returns.
        """
        portfolio_value_history = np.array(self.get_portfolio_value_history())
        step_returns = np.diff(portfolio_value_history) / portfolio_value_history[:-1]
        return step_returns
