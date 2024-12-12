import json
import pickle
import boto3
import mlflow

import numpy as np
import pandas as pd

from typing import Literal
from fastapi import FastAPI, Body, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
from typing_extensions import Annotated


def load_model(model_name: str, alias: str):
    """
    Load a trained model and associated data dictionary.

    This function attempts to load a trained model specified by its name and alias. If the model is not found in the
    MLflow registry, it loads the default model from a file. Additionally, it loads information about the ETL pipeline
    from an S3 bucket. If the data dictionary is not found in the S3 bucket, it loads it from a local file.

    :param model_name: The name of the model.
    :param alias: The alias of the model version.
    :return: A tuple containing the loaded model, its version, and the data dictionary.
    """

    try:
        # Load the trained model from MLflow
        mlflow.set_tracking_uri('http://mlflow:5000')
        client_mlflow = mlflow.MlflowClient()

        model_data_mlflow = client_mlflow.get_model_version_by_alias(model_name, alias)
        model_ml = mlflow.sklearn.load_model(model_data_mlflow.source)
        version_model_ml = int(model_data_mlflow.version)
    except:
        # If there is no registry in MLflow, open the default model
        file_ml = open('/app/files/model.pkl', 'rb')
        model_ml = pickle.load(file_ml)
        file_ml.close()
        version_model_ml = 0

    try:
        # Load information of the ETL pipeline from S3
        s3 = boto3.client('s3')

        s3.head_object(Bucket='data', Key='data_info/data.json')
        result_s3 = s3.get_object(Bucket='data', Key='data_info/data.json')
        text_s3 = result_s3["Body"].read().decode()
        data_dictionary = json.loads(text_s3)

        data_dictionary["standard_scaler_mean"] = np.array(data_dictionary["standard_scaler_mean"])
        data_dictionary["standard_scaler_std"] = np.array(data_dictionary["standard_scaler_std"])
    except:
        # If data dictionary is not found in S3, load it from local file
        file_s3 = open('/app/files/data.json', 'r')
        data_dictionary = json.load(file_s3)
        file_s3.close()

    return model_ml, version_model_ml, data_dictionary


def check_model():
    """
    Check for updates in the model and update if necessary.

    The function checks the model registry to see if the version of the champion model has changed. If the version
    has changed, it updates the model and the data dictionary accordingly.

    :return: None
    """

    global model
    global data_dict
    global version_model

    try:
        model_name = "spotify_playlist_model_prod"
        alias = "champion"

        mlflow.set_tracking_uri('http://mlflow:5000')
        client = mlflow.MlflowClient()

        # Check in the model registry if the version of the champion has changed
        new_model_data = client.get_model_version_by_alias(model_name, alias)
        new_version_model = int(new_model_data.version)

        # If the versions are not the same
        if new_version_model != version_model:
            # Load the new model and update version and data dictionary
            model, version_model, data_dict = load_model(model_name, alias)

    except:
        # If an error occurs during the process, pass silently
        pass


class ModelInput(BaseModel):
    acousticness: float = Field(
        description="Track's Acousticness",
        ge=0,
        le=1,
    )
    danceability: float = Field(
        description="Track's Danceability",
        ge=0,
        le=1,
    )
    duration: int = Field(
        description="Track's duration in milliseconds",
        ge=0,
        le=99999999999999,
    )
    energy: float = Field(
        description="Track's Energy",
        ge=0,
        le=1,
    )
    instrumentalness: float = Field(
        description="Track's Instrumentalness",
        ge=0,
        le=1,
    )
    key: int = Field(
        description="Track's Key",
        ge=0,
        le=11,
    )
    liveness: float = Field(
        description="Track's Liveness",
        ge=0,
        le=1,
    )
    loudness: float = Field(
        description="Track's Loudness",
        ge=-30,
        le=0,
    )
    mode: int = Field(
        description="Track's Mode",
        ge=0,
        le=1,
    )
    speechiness: float = Field(
        description="Track's Speechiness",
        ge=0,
        le=1,
    )
    tempo: float = Field(
        description="Track's Tempo",
        ge=0,
        le=200,
    )
    time_signature: int = Field(
        description="Track's Time Signature",
        ge=0,
        le=5,
    )
    valence: float = Field(
        description="Track's Valence",
        ge=0,
        le=1,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "acousticness": 0.441,
                    "danceability": 0.737,
                    "duration": 111422,
                    "energy": 0.516,
                    "instrumentalness": 0.000035,
                    "key": 2,
                    "liveness": 0.122,
                    "loudness": -8.023,
                    "mode": 1,
                    "speechiness": 0.0411,
                    "tempo": 117.027,
                    "time_signature": 4,
                    "valence": 0.532,
                }
            ]
        }
    }


class ModelOutput(BaseModel):
    """
    Output schema for the heart disease prediction model.

    This class defines the output fields returned by the heart disease prediction model along with their descriptions
    and possible values.

    :param int_output: Output of the model. True if the patient has a heart disease.
    :param str_output: Output of the model in string form. Can be "Healthy patient" or "Heart disease detected".
    """

    int_output: bool = Field(
        description="Output of the model. True if the user likes the song",
    )
    str_output: Literal["Likes song", "Dislikes song"] = Field(
        description="Output of the model in string form",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "int_output": True,
                    "str_output": "Likes the song",
                }
            ]
        }
    }


# Load the model before start
model, version_model, data_dict = load_model("spotify_playlist_model_prod", "champion")

app = FastAPI()


@app.get("/")
async def read_root():
    """
    Root endpoint of the Spotify Detector API.

    This endpoint returns a JSON response with a welcome message to indicate that the API is running.
    """
    return JSONResponse(content=jsonable_encoder({"message": "Welcome to the Spotify playlist API"}))


@app.post("/predict/", response_model=ModelOutput)
def predict(
    features: Annotated[
        ModelInput,
        Body(embed=True),
    ],
    background_tasks: BackgroundTasks
):
    """
    Endpoint for predicting liked Spotify Songs.

    This endpoint receives features related to a Spotify Track and predicts whether the user likes it
    or not using a trained model. It returns the prediction result in both integer and string formats.
    """

    # Extract features from the request and convert them into a list and dictionary
    features_list = [*features.dict().values()]
    features_key = [*features.dict().keys()]

    # Convert features into a pandas DataFrame
    features_df = pd.DataFrame(np.array(features_list).reshape([1, -1]), columns=features_key)

    tempo_bands = pd.get_dummies(
        pd.cut(features_df['tempo'], bins=[0, 80, 120, float('inf')],
               labels=['tempo_slow', 'tempo_moderate', 'tempo_fast'])
    )
    features_df = pd.concat([features_df, tempo_bands], axis=1)
    features_df.drop(columns=['tempo'], inplace=True)

    # Speechiness bands
    speechiness_bands = pd.get_dummies(
        pd.cut(features_df['speechiness'], bins=[0, 0.20, 0.66],
               labels=['speechiness_low', 'speechiness_moderate'])
    )
    features_df = pd.concat([features_df, speechiness_bands], axis=1)
    features_df.drop(columns=['speechiness'], inplace=True)

    # Reorder DataFrame columns
    features_df = features_df[data_dict["columns_after_processing"]]

    # Scale the data using standard scaler
    features_df = (features_df-data_dict["standard_scaler_mean"])/data_dict["standard_scaler_std"]

    # Make the prediction using the trained model
    prediction = model.predict(features_df)

    # Convert prediction result into string format
    str_pred = "Dislikes song"
    if prediction[0] > 0:
        str_pred = "Likes song"

    # Check if the model has changed asynchronously
    background_tasks.add_task(check_model)

    # Return the prediction result
    return ModelOutput(int_output=bool(prediction[0].item()), str_output=str_pred)
