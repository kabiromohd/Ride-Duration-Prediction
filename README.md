## Ride-Duration-Prediction
# Mlops Zoomcamp Capstone Project
## Purpose of project

The project is to build an end-to-end Batch Deployment MLOPS project. 

Important: The project main focus of the project is to show the MLOps flow and not to build the best model.

The underlying ML task is to predict bike ride duration given the start and end station, start time, bike type, and type of membership.

# Potential use case::
A potential customer takes a bike from a station and wants to know how long it will take to get to the destination station. They enter the destination station and the rest of the features are logged automatically. The request is sent to the web service that returns the predicted duration and the customer can decide if they want to take the bike or not.

# The data
The data is provided by Divvy Bikeshare and contains information about bike rides in Washington DC. Downloadable files are available on the following [link](https://divvy-tripdata.s3.amazonaws.com/index.html) The data used for the project model training is April 2020.

# Project flow

- Raw data download
- Exploratory Data Analyis (EDA)
- Experiment Tracking (Mlflow) and Workflow Orchestration (Mage)
  - Data preparation
  - Modelling
  - Baseline model
  - Hyperparameter tuning using Mage
  - Training the model with the best hyperparameters
  - Experiment tracking with Mlflow.
  - Registration of Best model on Mlflow
- Local Webservice Model deployment (Docker + Flask)
- Cloud Webservice Model deployment [Render Cloud Service](Render.com)
- Project Monitoring (Evidently AI)

## Steps to reproduce:
Important: This project is intended to be easily reproducible, but for that you'll need a pipenv environment for the project and conda environment for the monitoring installation.

# General
- Setup Pipenv Virtual Environment, by opening new terminal on your system and run this series of command
  
```
mkdir project

cd project

pip install pipenv
```

- Get copies of the project and dependencies, you can clone the repo.
- The files should be placed in the virtual environment folder after being cloned via below command.

```
git clone https://github.com/kabiromohd/Ride-Duration-Prediction.git
```

## Explore and prepare the data
- Explore data
  - Checked the Data Structure and columns
  - Checked the numbers of features and observations in the data
  - Checked the inconsistency in column names and corrected.
- prepare data
  - Checked for missing values (filled with 0)
  - Checked for outliers
  - Checked for Duplicates

## Experiment Tracking (Mlflow) and Workflow Orchestration (Mage)
Launch Mage, Mlflow and the database service (PostgreSQL) a docker-compose.yml is available to launch all the services

Navigate to the following directory by running the following commands on your terminal

```
cd orchestration/mlops

./scripts/start.sh

```

This will launch mlfow and Mage which can be accessed via the browser

Mage: http://localhost:6789

Mlflow: http://localhost:5000

Through this you can access the mlflow experiment runs and the orchestration code on Mage.

After training of the model and registration of the best model on Mlflow, you download the Artifacts to your local file system for model deployment.

## Deploy model locally with Docker

You can deploy the model on Docker locally by following these steps.
To do so, you need to have Docker installed on your environment, then you build the image with the following command:

We will eventually deploy the docker image created to cloud via Render hence the following needs to be in place.
- Create a Docker Account 
- Creating an account on Docker enables setting up of Docker repository which can be used to push the docker image created locally to docker hub.
- The docker repo is create on the docker web login
- Docker repo created for the purpose of this project is *"kabiromohd/data_science"*
- Docker repository was created to enable getting URL for the mlopscapstone image.

```
pipenv shell
```

Create docker image by running the following:

```
docker build -t <your docker user>/<docker repo>:mlopscapstone -f Dockerfile.deploy .
```

followed by this docker command which runs the docker image created

```
docker run -it --rm -p 9090:9090 <your docker user>/<docker repo>:mlopscapstone
```

### To interact with the locally Deployed webservice
if all the above command run successfully, open another terminal and run below command to see Ride prediction for local Docker deployment:

```
pipenv shell
```

Note: *predict_test.py* has already prepared with data point to test the model deployed locally on docker

Run below command. 

```
python predict_test.py
```

This ends the local deployment to docker.

## Deploy docker image to the cloud

For cloud deployment [Render Cloud Service](render.com) was used.

To deploy the docker image to cloud, open a terminal and run the following commands:

```
pipenv shell
```

- Push the docker image created above to the repo created with the following command:

```
docker push <your docker user>/<docker repo>:mlopscapstone
```

- copy the docker image URL to render from the docker Hub repo
  
- deploy docker image to render cloud service.
  
### To interact with the docker image deployed to cloud via Render Cloud service

- copy the render deployment link, [e.g for my project](https://ride-duration-prediction.onrender.com)  and place in the *predict_test_cloud.py* script as "host".
- *predict-test_cloud.py* has already prepared data point to be used to test the model deployed to cloud.
- for this project, the deployment link has already been provided in the predict_test_cloud.py script. It can be executed as illustrated below:
  
- open a new terminal and run the following: 

  ```
  pipenv shell
  ```

  followed by:
  
  ```
  python predict_test_cloud.py
  ```

# Monitoring with Evidently AI

We quarter prediction data on use of ride is collected and used for monitoring of the model performance

to view the monitoring dashbaord on Evidently AI follow the following:

```
cd project/monitoring

conda create --name myenv python=3.11

conda activate myenv

pip install -r requirements.txt

python baseline_model_Bikeride_capstone.py

evidently ui
```
Go to your browser and open

http://localhost:8000

and you see the deployed dashboard
