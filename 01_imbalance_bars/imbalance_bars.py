import pandas as pd
from typing import List

class ImbalanceBarGenerator:
    def __init__(self, dataframe: pd.DataFrame, ewma_window: int, T_init: int = 10, imbalance_init: float = 0.0, threshold_value: float = None):
        """
        Initialize the Imbalance Bar Generator with dynamic threshold adjustment.
        
        Args:
            dataframe (pd.DataFrame): DataFrame containing the columns 'date', 'open', 'close', 'high', 'low', 'volume'.
            ewma_window (int): The window size for the Exponentially Weighted Moving Average (EWMA) of the imbalance.
            T_init (int): Initial value for T before any bars are created.
            imbalance_init (float): Initial value for imbalance before any bars are created.
            threshold_value (float): Valor fijo para clipear el umbral del desequilibrio.
        """
        self.dataframe = dataframe.copy() 
        self.threshold = threshold_value
        self.ewma_window = ewma_window
        self.ewma_T = T_init  # EWMA for E_0[T] initialized with T_init
        self.ewma_imbalance = imbalance_init  # EWMA for 2v_+ - E_0[v_t] initialized with imbalance_init
        self.alpha_T = 2 / (ewma_window + 1)  # Smoothing factor for EWMA
        self.bars = []
        self.cumulative_imbalance = 0.0
        self.imbalance_type = None  # To be set in fit method
        self._apply_tick_rule()

        self.cum_imbalance_series = []
        self.expected_imabalance_series = []
        self.expected_imbalance_per_bar = []


    def _apply_tick_rule(self):
        """Apply the tick rule to calculate the signed ticks."""
        self.dataframe['delta_p'] = self.dataframe['close'] - self.dataframe['close'].shift(1)  # Calcular Δp_t como close_t - close_{t-1}

        # Asignar valores iniciales basados en delta_p
        self.dataframe['bt'] = self.dataframe['delta_p'].apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
        # Asegurar que el primer valor sea 1
        self.dataframe.loc[0, 'bt'] = 1
        # Llenar valores donde delta_p == 0 con el valor anterior de bt
        self.dataframe['bt'] = self.dataframe['bt'].ffill().replace(0, 1)
        # Eliminar la columna delta_p
        self.dataframe.drop(columns=['delta_p'], inplace=True)

    def _reset_bar(self, first_bar:bool = False):
        """Reset values for the next bar."""

        self.expected_imbalance = self.ewma_T * abs(self.ewma_imbalance)
        self._clip_threshold()
        #print(self.expected_imbalance)
        self.expected_imbalance_per_bar.append(self.expected_imbalance)

        #print('EXPECTED IMABALANCE', self.expected_imbalance)
        self.cumulative_imbalance = 0.0
        self.current_high = -float('inf')
        self.current_low = float('inf')
        self.open_price = None
        self.T = 0  

    def _clip_threshold(self):
        """Ajusta el umbral a un valor fijo si está definido."""
        # print(self.expected_imbalance, self.threshold)
        # print(self.expected_imbalance >= self.threshold)
        if self.threshold is not None:
            #print('clipeado')
            self.expected_imbalance = self.threshold
    
    def _update_ewma_T(self, value: float, ewma: float) -> float:
        """Update EWMA value dynamically."""
        return (self.alpha_T * value) + ((1 - self.alpha_T) * ewma)

    def fit(self, imbalance_type: str = "volume"):
        """
        Fit the Imbalance Bar Generator to the data and calculate the imbalance bars.
        
        Args:
            imbalance_type (str): The type of imbalance to calculate, either "volume" or "dollar".
        """
        if imbalance_type not in ["volume", "dollar"]:
            raise ValueError("Imbalance type must be either 'volume' or 'dollar'")

        self.imbalance_type = imbalance_type
        vt_column = 'volume' if imbalance_type == 'volume' else 'dollar'
        if imbalance_type == 'dollar':
            self.dataframe['dollar'] = self.dataframe['close'] * self.dataframe['volume']

        self.dataframe['imbalance'] = self.dataframe['bt'] * self.dataframe[vt_column]
        self.dataframe['imbalance_ewma'] = self.dataframe['imbalance'].ewm(span = self.ewma_window).mean()
        self.ewma_imbalance = self.dataframe['imbalance_ewma'][0]
        #print(self.ewma_imbalance)
        self._reset_bar(first_bar=True)
        for _, row in self.dataframe.iterrows():

            self.T += 1

            self.ewma_imbalance = row['imbalance_ewma']
            self.cumulative_imbalance += row['imbalance']
            
            self.cum_imbalance_series.append(self.cumulative_imbalance)
            self.expected_imabalance_series.append(self.expected_imbalance)
            
            if self.open_price is None:
                self.open_price = row['open']

            self.current_high = max(self.current_high, row['high'])
            self.current_low = min(self.current_low, row['low'])
            self.close_price = row['close']

            if abs(self.cumulative_imbalance) >= self.expected_imbalance:
                              
                self.bars.append((row['date'], self.open_price, self.current_high, self.current_low, self.close_price))
 
                # Actualizar EWMA para E_0[T] y 2v_+ - E_0[v_t] con los valores de la barra recién formada
                #print('T',self.ewma_T)
                #print('imbalance',self.ewma_imbalance)
                self.ewma_T = self._update_ewma_T(self.T, self.ewma_T)

                # Reiniciar variables para la siguiente barra
                self._reset_bar()

        self.dataframe['cum_imbalance'] = self.cum_imbalance_series
        self.dataframe['expected_imbalance'] = self.expected_imabalance_series

    def get_bars(self) -> pd.DataFrame:
        """
        Get the DataFrame of generated bars.
        
        Returns:
            pd.DataFrame: DataFrame of bars with columns (date, open, high, low, close).
        """
        bars_df = pd.DataFrame(self.bars, columns=['date', 'open', 'high', 'low', 'close'])
        return bars_df

    def get_series(self, variable) -> List[float]:

        return self.dataframe[variable]
