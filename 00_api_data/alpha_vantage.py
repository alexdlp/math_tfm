import requests
import pandas as pd
from datetime import datetime, timedelta
import argparse
import json

# Configuración
ALPHA_KEY = 'QEON04VFFZ7PDMAD'

def convert_to_df(data, ascending = True):

    # Convertimos el diccionario en un DataFrame
    df = pd.DataFrame.from_dict(data, orient='index')

    # Renombramos las columnas para tener nombres más claros
    df.columns = ['open', 'high', 'low', 'close', 'volume']

    # Convertimos los tipos de datos de las columnas al formato correcto
    df = df.astype({
        'open': 'float',
        'high': 'float',
        'low': 'float',
        'close': 'float',
        'volume': 'int'
    })

    # Opcional: Convertimos el índice en un DateTimeIndex para manejar mejor las fechas
    df.index = pd.to_datetime(df.index)

    # Para ordenar el DataFrame en orden descendente (de más reciente a más antiguo -> ascending = False)
    df_sorted_desc = df.sort_index(ascending = ascending)

    return df_sorted_desc


def fetch_data(start_date, end_date, symbol, interval):

    # Convierte las fechas de string a objetos datetime
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')

    # DataFrame final para todos los datos
    all_data = pd.DataFrame()

    try:

        current_date = start_date
        while current_date <= end_date:
            
            year = current_date.strftime('%Y')
            month = current_date.strftime('%m')

            # Forma la URL de la API usando el año y el mes actuales del bucle
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval={interval}&month={year}-{month}&outputsize=full&apikey={ALPHA_KEY}'

            # Realiza la solicitud a la API
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                
                
                # guardo el jon por si acaso -> QUITAR EN EL FUTURO
                #filename = f'data_{year}_{month}.json'
                # Open a file and write the JSON data
                # with open(filename, 'w') as file:
                #     json.dump(data, file, indent=4)
                

                index = f'Time Series ({interval})'       
                df = convert_to_df(data[index])
                #print('convert to df successful')
          
                all_data = pd.concat([all_data, df])
                print(f'Data for {year}-{month} dowloaded successfully')

            else:
                print(f"Application error for {year}-{month}: {response.status_code}")

            # Calcula el próximo mes
            if current_date.month == 12:
                current_date = datetime(current_date.year + 1, 1, 1)
            else:
                current_date = datetime(current_date.year, current_date.month + 1, 1)

    except Exception as ex:
        print(f"The loop stopped due to the following exception: {ex}")

    finally:

        # Opcional: convertir el índice a datetime y ordenar
        all_data.index = pd.to_datetime(all_data.index)
        all_data.sort_index(inplace=True)

        return all_data


def main():

    parser = argparse.ArgumentParser(description='Fetch financial data from Alpha Vantage API.')
    parser.add_argument('--start_date', type=str, default = '2000-01-01', help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end_date', type=str, default = '2024-06-30', help='End date in YYYY-MM-DD format')
    parser.add_argument('--symbol', type=str, default = 'AAPL', help='Symbol for the stock (e.g., IBM)')
    parser.add_argument('--interval', type=str, default='1min',  help='Time interval between two consecutive data points in the time series. The following values are supported: 1min, 5min, 15min, 30min, 60min')
    
    args = parser.parse_args()

    df_data = fetch_data(args.start_date, args.end_date, args.symbol, args.interval)

    # Generar un nombre de archivo
    filename = f"Historical_Intraday_{args.symbol}_{args.interval}_{args.start_date}_to_{args.end_date}.csv"
    
    # Guardar el DataFrame en un archivo CSV
    df_data.to_csv(filename, index=True)
    
    print(f"Data saved to {filename}")
  

if __name__ == '__main__':
    main()