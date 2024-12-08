# Implementación modelo de predicción de Spotify
### AMq2 - CEIA - FIUBA

Comando para levantar ```docker-compose --profile all up --build```

Una vez levantado ir a la [UI de Airflow](http://localhost:8080) y activar DAG `process_etl_spotify_data`.

Correr notebook de initial_train_notebook/experiment_mlflow.ipynb para generar la primera versión del modelo y registrarlo en MlFlow. Se puede ver el progreso del entrenamiento y el mejor modelo en la [siguiente UI](http://localhost:5001).

Para realizar nuevos entrenamientos se puede utilizar el DAG `retrain_the_model`.

... Como usar API

Comando para bajar **y borrar volumes** ```docker-compose --profile all down -v```.