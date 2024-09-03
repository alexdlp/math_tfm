# Math_tfm
Repositorio del TFM del máster de matemáticas UPV/EHU.

Contiene el código y los recursos utilizados en el tfm de trading algorítmico, centrado en la implementación de técnicas de *Deep Reinforcement Learning* (DRL) y técnicas del libro *Advances in Financial Machine Learnign* 
## Estructura de Carpetas

### 00_api_data
Esta carpeta contiene los scripts y los datos descargados a través de APIs financieras. Estos datos son la base para el análisis posterior y las simulaciones de trading. Incluye precios históricos, volúmenes y otros indicadores relevantes.

### 01_imbalance_bars
En esta carpeta se encuentran los scripts y resultados relacionados con la creación de barras de desequilibrio (*imbalance bars*). Las barras de desequilibrio son una técnica utilizada para estructurar los datos financieros, capturando eventos relevantes en lugar de basarse en intervalos de tiempo fijos. 

### 02_training_data
Aquí se generan los conjuntos de datos preparados para el entrenamiento de los modelos de *Deep Reinforcement Learning*. Estos datos han sido procesados y adaptados a partir de las barras de desequilibrio y otros métodos.

### 03_training
Esta carpeta contiene los scripts y modelos relacionados con el proceso de entrenamiento del agente de *Deep Reinforcement Learning*. Incluye configuraciones, parámetros de entrenamiento y resultados intermedios.

### 04_validation
En esta carpeta se encuentran los scripts y resultados del proceso de validación. Aquí se evalúa el desempeño del agente entrenado utilizando un conjunto de datos *out-of-sample*, lo que permite medir su capacidad de generalización y la robustez de las estrategias desarrolladas.

### 05_meta_labeling
Esta carpeta incluye los scripts y resultados relacionados con la implementación de *meta-labeling*. 

### tfm 
Contiene el tfm.

### tfmII
Mismo texto, distinta estructura.
