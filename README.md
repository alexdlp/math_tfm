# math_tfm
Repositorio del TFM del máster de matemáticas UPV/EHU.


# Carpeta api data
Contiene notebooks de acceso a las apis de mercado. Se han utilizado principalmente las apis de alpha vantage y financial modelling prep.

#### Notebook 00_Market_Api_data
Resumen de las apis disponibles y metodos para obtención de datos desde la api de FMP. Cada llamada a la api sol dos días (en caso de datos de 1min) de cotizaciónn, por lo que necesitan ser agregados después. 

#### Notebook 01_Merge_market_data
Los proveedores de datos ofrecen llamadas gratuítas a la api, pero limitadas. Por lo que para obtener series temporales de datos financieros de varios años, hay que hacer llamadas durante varios días y agregar los resultados. Este notebok contiene la agregacion de los distintos archivos (que permanecen separados para poder ser subidos a github)

#### Notebook 02_Process_market_data
Contiene la obtención de indicadores que después serán utilizados como observaciones en el entorno del agente.

#### alpha vantage.py
Similar al primer notebook, pero en formato script para la api de alpha vantage.


