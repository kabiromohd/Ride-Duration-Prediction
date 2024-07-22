## Ride-Duration-Prediction
# Mlops Zoomcamp Capstone Project
## The purpose of the project
The project is to build an end-to-end MLOPS project. 

Important: The project main focus of the project is to show the MLOps flow and not to build the best model.

The underlying ML task is to predict bike ride duration given the start and end station, start time, bike type, and type of membership.

# Potential use case is the following:
A potential customer takes a bike from a station and wants to know how long it will take to get to the destination station. They enter the destination station and the rest of the features are logged automatically. The request is sent to the web service that returns the predicted duration and the customer can decide if they want to take the bike or not.

# The data
The data is provided by Capital Bikeshare and contains information about bike rides in Washington DC. Downloadable files are available on the following [link]https://s3.amazonaws.com/capitalbikeshare-data/index.html](https://divvy-tripdata.s3.amazonaws.com/index.html) The data used for the project model training is April 2020.

# Project flow
- Raw data download
- Exploratory Data Analyis (EDA)
- Experiment Tracking (Mlflow) and Workflow Orchestration (Mage)
  - Data preparation
  - Modelling
  - Baseline model
  - Hyperparameter tuning using Weights and Biases Sweeps
  - Training the model with the best hyperparameters
  - Using Mlflow for Experiment tracking.
  - Registration of Best model on Mlflow
- Local Webservice Model deployment (Docker + Flask)
- Cloud Webservice Model deployment [Render](Render.com)
- Project Monitoring (Evidently AI)
