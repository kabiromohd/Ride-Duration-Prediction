import os
import pickle
import click
import mlflow
import numpy as np
#import optuna

#from optuna.samplers import TPESampler
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll import scope

import mlflow
import pandas as pd
import pickle

from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline

import pandas as pd

import time
from datetime import date

EXPERIMENT_NAME = "RamdomForestRegression_ZCMlopsCapstone1"
mlflow.set_tracking_uri("http://mlflow:5000")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.sklearn.autolog()




def run_optimization(data, dv, num_trials = 10):
    mlflow.sklearn.autolog(disable=True)
    
    month_uniq = data['started_at'].dt.month.unique()

    month1 = month_uniq[0]
    month2 = month_uniq[1]
    month3 = month_uniq[2]

    df_month1 = extract_data_for_month(data, 'started_at', month1)
    df_month2 = extract_data_for_month(data, 'started_at', month2)
    df_month3 = extract_data_for_month(data, 'started_at', month3)

    df_month1.to_csv('./data/df_month1.csv', index=False)
    df_month2.to_csv('./data/df_month2.csv', index=False)
    df_month3.to_csv('./data/df_month3.csv', index=False)

    X_train, y_train = split_data(df_month3, dv, fit_dv=True)
    X_val, y_val = split_data(df_month2, dv, fit_dv=False)
    X_test, y_test = split_data(df_month1, dv, fit_dv=False)

    def objective(params):
        with mlflow.start_run():
            mlflow.log_params(params)
            rf = RandomForestRegressor(**params)
            rf.fit(X_train, y_train)

            # Log model on Mlflow
            mlflow.sklearn.log_model(rf, artifact_path = "models")

            y_pred = rf.predict(X_val)
            rmse = mean_squared_error(y_val, y_pred, squared=False)
            mlflow.log_metric("rmse", rmse)
            with open('preprocess.bin', 'wb') as f_out : 
                pickle.dump(dv, f_out)
            mlflow.log_artifact("preprocess.bin", artifact_path="preprocess")

        return {'loss': rmse, 'status': STATUS_OK}

    search_space = {
        'max_depth': scope.int(hp.quniform('max_depth', 1, 20, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 10, 50, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 10, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    }
    
    rstate = np.random.default_rng(42)  # for reproducible results
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )

RF_PARAMS = ['max_depth', 'n_estimators', 'min_samples_split', 'min_samples_leaf', 'random_state']

def extract_data_for_month(df, date_column, month):
    
    # Filter DataFrame for the specified month
    filtered_df = df[df[date_column].dt.month == month]

    return filtered_df

def preprocess_data(df):
    
    df['duration'] = df["ended_at"] - df["started_at"]
    df['duration'] = df['duration'].apply(lambda td: td.total_seconds() / 60)

    #df = df[(df.duration >= 1) & (df.duration <= 60)]

    categorical = ['start_station_id', 'end_station_id']
    df[categorical] = df[categorical].astype(str)
    

    reqd_cols = ['ride_id', 'start_station_id', 'end_station_id', 'duration']

    df = df[reqd_cols]
    
    return df

def split_data(df, dv: DictVectorizer, fit_dv: bool = False):
    df = preprocess_data(df)
    target = "duration"
    y = df[target].values
    del df["duration"]
    X = df

    dicts = X.to_dict(orient='records')
    if fit_dv:
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)

    return X, y

def train_and_log_model(dv, data, params):
    month_uniq = data['started_at'].dt.month.unique()

    month1 = month_uniq[0]
    month2 = month_uniq[1]
    month3 = month_uniq[2]

    df_month1 = extract_data_for_month(data, 'started_at', month1)
    df_month2 = extract_data_for_month(data, 'started_at', month2)
    df_month3 = extract_data_for_month(data, 'started_at', month3)

    df_month1.to_csv('./data/df_month1.csv', index=False)
    df_month2.to_csv('./data/df_month2.csv', index=False)
    df_month3.to_csv('./data/df_month3.csv', index=False)

    X_train, y_train = split_data(df_month3, dv, fit_dv=True)
    X_val, y_val = split_data(df_month2, dv, fit_dv=False)
    X_test, y_test = split_data(df_month1, dv, fit_dv=False)

    with mlflow.start_run():
        for param in RF_PARAMS:
            params[param] = int(params[param])

        rf = RandomForestRegressor(**params)
        rf.fit(X_train, y_train)

        # log model on mlflow
        mlflow.sklearn.log_model(rf, artifact_path = "models")

        # Evaluate model on the validation and test sets
        val_rmse = mean_squared_error(y_val, rf.predict(X_val), squared=False)
        mlflow.log_metric("val_rmse", val_rmse)
        test_rmse = mean_squared_error(y_test, rf.predict(X_test), squared=False)
        mlflow.log_metric("test_rmse", test_rmse)
        with open('preprocess/preprocess.bin', 'wb') as f_out : 
                pickle.dump(dv, f_out)
        mlflow.log_artifact("preprocess/preprocess.bin", artifact_path="preprocess")



def run_register_model(data_path, dv):

    client = MlflowClient()

    # Retrieve the top_n model runs and log the models
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.rmse ASC"]
    )
    for run in runs:
        train_and_log_model(dv, data_path, params=run.data.params)

    # Select the model with the lowest test RMSE
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=5,
        order_by=["metrics.test_rmse ASC"]
    )[0]

    # Register the best model
    run_id_best_model = best_run.info.run_id
    model_uri = f"runs:/{run_id_best_model}/models"
    mlflow.register_model(model_uri, name="RF-best-model")

    #Load Artifact from Mlflow
    path = client.download_artifacts(run_id=run_id_best_model, path='preprocess/preprocess.bin')

    # Save Artifact for best model to local file system
    with open('preprocess.bin', 'wb') as f_out:
        pickle.dump(path, f_out)

    logged_model = f'runs:/{run_id_best_model}/models'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    # Save best model to local file system
    with open('model.bin', 'wb') as f_out:
        pickle.dump(loaded_model, f_out)


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(data, *args, **kwargs):
    """
    Exports data to some source.

    Args:
        data: The output from the upstream parent block
        args: The output from any additional upstream blocks (if applicable)

    Output (optional):
        Optionally return any object and it'll be logged and
        displayed when inspecting the block run.
    """
    dv = DictVectorizer()
    run_optimization(data, dv)
    run_register_model(data, dv)