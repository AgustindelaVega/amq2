import datetime

from airflow.decorators import dag, task

markdown_text = """
### ETL Process for Spotify playlist Data

This DAG extracts information from the original CSV file stored in the following 
[Google Drive folder](https://drive.google.com/uc?export=download&id=1c1l3OMUFjQWcydV0srKe95LLEfkBMKia). 
It preprocesses the data by dropping outliers, creating bands and scaling numerical features.
    
After preprocessing, the data is saved back into a S3 bucket as two separate CSV files: one for training and one for 
testing. The split between the training and testing datasets is 70/30 and they are stratified.
"""


default_args = {
    'owner': "Agustina Quiros, Agustín de la Vega, Florentino Arias",
    'depends_on_past': False,
    'schedule_interval': None,
    'retries': 1,
    'retry_delay': datetime.timedelta(minutes=5),
    'dagrun_timeout': datetime.timedelta(minutes=15)
}


@dag(
    dag_id="process_etl_spotify_data",
    description="ETL process for Spotify playlist data, separating the dataset into training and testing sets.",
    doc_md=markdown_text,
    tags=["ETL", "Spotify Playlist"],
    default_args=default_args,
    catchup=False,
)
def process_etl_spotify_data():

    @task.virtualenv(
        task_id="obtain_original_data",
        requirements=["requests", "pandas", "awswrangler"],
        system_site_packages=True
    )
    def get_data():
        import requests
        import awswrangler as wr
        import pandas as pd

        gdrive_url = "https://drive.google.com/uc?export=download&id=1c1l3OMUFjQWcydV0srKe95LLEfkBMKia"
        local_file = "playlist_data.csv"

        with requests.get(gdrive_url, stream=True) as r:
            r.raise_for_status()
            with open(local_file, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)

        df = pd.read_csv(local_file)

        # Save to S3
        data_path = "s3://data/raw/playlist_data.csv"
        wr.s3.to_csv(df=df, path=data_path, index=False)


    @task.virtualenv(
        task_id="process_features",
        requirements=["awswrangler==3.6.0"],
        system_site_packages=True
    )
    def process_features():
        import json
        import datetime
        import boto3
        import botocore.exceptions
        import mlflow

        import awswrangler as wr
        import pandas as pd

        from airflow.models import Variable

        data_original_path = "s3://data/raw/playlist_data.csv"
        data_end_path = "s3://data/raw/playlist_final.csv"
        df = wr.s3.read_csv(data_original_path)

        # Clean irrelevant features
        clean_df = df.drop(columns=['instrumentalness', 'duration', 'mode', 'time_signature'])

        # Clean outliers
        for column in clean_df.select_dtypes(include=['float64', 'int64']).columns:
            if column == "speechiness" or column == "liveness" or column == "loudness":
                Q1 = clean_df[column].quantile(0.25)
                Q3 = clean_df[column].quantile(0.75)
                IQR = Q3 - Q1

                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                clean_df[column] = clean_df[column].clip(lower=lower_bound, upper=upper_bound)

        # Tempo bands
        tempo_bands = pd.get_dummies(
            pd.cut(clean_df['tempo'], bins=[0, 80, 120, float('inf')], labels=['tempo_slow', 'tempo_moderate', 'tempo_fast']))
        clean_df = pd.concat([clean_df, tempo_bands], axis=1)
        clean_df.drop(columns=['tempo'], inplace=True)

        # Speechiness bands
        speechiness_bands = pd.get_dummies(pd.cut(clean_df['speechiness'], bins=[0, 0.20, 0.66],
                                            labels=['speechiness_low', 'speechiness_moderate']))
        clean_df = pd.concat([clean_df, speechiness_bands], axis=1)
        clean_df.drop(columns=['speechiness'], inplace=True)

        wr.s3.to_csv(df=clean_df,
                     path=data_end_path,
                     index=False)

        # Save information of the dataset
        client = boto3.client('s3')

        data_dict = {}
        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            if e.response['Error']['Code'] != "404":
                # Something else has gone wrong.
                raise e

        target_col = Variable.get("target_col_spotify")
        dataset_log = df.drop(columns=target_col)
        dataset_final = clean_df.drop(columns=target_col)

        # Upload JSON String to an S3 Object
        data_dict['columns'] = dataset_log.columns.to_list()
        data_dict['columns_after_processing'] = dataset_final.columns.to_list()
        data_dict['target_col'] = target_col
        data_dict['columns_dtypes'] = {k: str(v) for k, v in dataset_log.dtypes.to_dict().items()}
        data_dict['columns_dtypes_after_processing'] = {k: str(v) for k, v in dataset_final.dtypes.to_dict().items()}

        data_dict['date'] = datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"')
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key='data_info/data.json',
            Body=data_string
        )

        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Spotify Playlist")

        mlflow.start_run(run_name='ETL_run_' + datetime.datetime.today().strftime('%Y/%m/%d-%H:%M:%S"'),
                         experiment_id=experiment.experiment_id,
                         tags={"experiment": "etl", "dataset": "Spotify Playlist"},
                         log_system_metrics=True)

        mlflow_dataset = mlflow.data.from_pandas(df,
                                                 source="https://drive.google.com/uc?export=download&id=1c1l3OMUFjQWcydV0srKe95LLEfkBMKia",
                                                 targets=target_col,
                                                 name="spotify_playlist_data_complete")
        mlflow_dataset_dummies = mlflow.data.from_pandas(clean_df,
                                                         source="https://drive.google.com/uc?export=download&id=1c1l3OMUFjQWcydV0srKe95LLEfkBMKia",
                                                         targets=target_col,
                                                         name="spotify_playlist_data_processed")
        mlflow.log_input(mlflow_dataset, context="Dataset")
        mlflow.log_input(mlflow_dataset_dummies, context="Dataset")

    @task.virtualenv(
        task_id="split_dataset",
        requirements=["awswrangler==3.6.0",
                      "scikit-learn==1.3.2",
                      "mlflow==2.10.2"],
        system_site_packages=True
    )
    def split_dataset():
        """
        Generate a dataset split into a training part and a test part
        """
        import awswrangler as wr
        import mlflow
        from sklearn.model_selection import train_test_split
        from airflow.models import Variable
        from sklearn.preprocessing import StandardScaler
        import pandas as pd
        import boto3
        import json
        import botocore.exceptions

        def save_to_csv(df, path):
            wr.s3.to_csv(df=df, path=path, index=False)

        data_original_path = "s3://data/raw/playlist_final.csv"
        dataset = wr.s3.read_csv(data_original_path)

        test_size = Variable.get("test_size_spotify")
        target_col = Variable.get("target_col_spotify")

        X = dataset.drop(columns=target_col)
        y = dataset[[target_col]]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=test_size)

        X_train_df = pd.DataFrame(X_train, columns=X.columns)
        X_test_df = pd.DataFrame(X_test, columns=X.columns)
        y_train_df = pd.DataFrame(y_train.values, columns=y.columns)
        y_test_df = pd.DataFrame(y_test.values, columns=y.columns)

        save_to_csv(X_train_df, "s3://data/final/train/spotify_X_train.csv")
        save_to_csv(X_test_df, "s3://data/final/test/spotify_X_test.csv")
        save_to_csv(y_train_df, "s3://data/final/train/spotify_y_train.csv")
        save_to_csv(y_test_df, "s3://data/final/test/spotify_y_test.csv")

        # Save information of the dataset
        client = boto3.client('s3')

        try:
            client.head_object(Bucket='data', Key='data_info/data.json')
            result = client.get_object(Bucket='data', Key='data_info/data.json')
            text = result["Body"].read().decode()
            data_dict = json.loads(text)
        except botocore.exceptions.ClientError as e:
            raise e

        # Upload JSON String to an S3 Object
        data_dict['standard_scaler_mean'] = X_train_df.mean().tolist()
        data_dict['standard_scaler_std'] = X_train_df.std().tolist()
        data_string = json.dumps(data_dict, indent=2)

        client.put_object(
            Bucket='data',
            Key='data_info/data.json',
            Body=data_string
        )

        mlflow.set_tracking_uri('http://mlflow:5000')
        experiment = mlflow.set_experiment("Spotify Playlist")

        # Obtain the last experiment run_id to log the new information
        list_run = mlflow.search_runs([experiment.experiment_id], output_format="list")

        with mlflow.start_run(run_id=list_run[0].info.run_id):

            mlflow.log_param("Train observations", X_train.shape[0])
            mlflow.log_param("Test observations", X_test.shape[0])
            mlflow.log_param("Standard Scaler feature names", X_train_df.columns.tolist())
            mlflow.log_param("Standard Scaler mean values", X_train_df.mean())
            mlflow.log_param("Standard Scaler scale values", X_train_df.std())

    get_data() >> process_features() >> split_dataset()


dag = process_etl_spotify_data()
