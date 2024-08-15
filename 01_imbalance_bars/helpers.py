import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas_ta as ta


def calculate_log_returns(df:pd.DataFrame):
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df.dropna(inplace=True)
    return df
def plot_ohlc(ax, df, color_up='green', color_down='red', width=0.001):
    for idx, row in df.iterrows():
        if row['close'] >= row['open']:
            color = color_up
            lower = row['open']
            higher = row['close']
        else:
            color = color_down
            lower = row['close']
            higher = row['open']
        
        # Graficar la línea vertical para High-Low (sombra)
        ax.plot([mdates.date2num(row['date']), mdates.date2num(row['date'])], [row['low'], row['high']], color=color, linewidth=1)

        # Graficar el rectángulo para Open-Close (cuerpo)
        rect = plt.Rectangle(
            (mdates.date2num(row['date']) - width/2, lower), 
            width, 
            higher-lower, 
            color=color, 
            alpha=1
        )
        ax.add_patch(rect)
