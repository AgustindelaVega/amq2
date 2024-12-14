# Implementación de Modelo de Predicción de Spotify
## AMq2 - CEIA Coh17 2024 - FIUBA
### Agustín De La Vega - Agustina Quirós - Florentino Arias

Para levantar el contenedor con todos los servicios en **Docker**, utilizar el comando ```docker-compose --profile all up --build```

Una vez levantado ir a la [UI de Airflow](http://localhost:8080) y activar el **DAG** `process_etl_spotify_data`.
Este DAG se encarga de extraer la info del archivo **playlist_data.csv** de **Google Drive**, realiza un preprocesamiento de los datos y crea 2 archivos separados en un **bucket s3** (simulado con **MinIO**): uno para entrenamiento (70%) y otro para test (30%).

Correr el Jupyter Notebook **initial_train_notebook/experiment_mlflow.ipynb** para generar la primera versión del modelo y registrarlo en **MlFlow**. Se puede ver el progreso del entrenamiento y el mejor modelo en la [UI de MlFlow](http://localhost:5001).

Para realizar nuevos entrenamientos se puede activar el **DAG** `retrain_the_model` desde Airflow.

Para probar las predicciones que realiza el modelo está disponible la [UI de FastAPI](http://localhost:8800/docs).

Para bajar **y borrar los volúmenes** de Docker se puede utilizar el comando ```docker-compose --profile all down -v```.