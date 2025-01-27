import requests
import zipfile
from io import BytesIO

import pandas as pd
import requests
from zipfile import ZipFile
import os
from mage_ai.io.file import FileIO


if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def load_data(*args, **kwargs):
    """
    Template code for loading data from any source.

    Returns:
        Anything (e.g. data frame, dictionary, array, int, str, etc.)
    """
    # Specify your data loading logic here

    response = requests.get("https://divvy-tripdata.s3.amazonaws.com/Divvy_Trips_2020_Q1.zip")

    if response.status_code != 200:
           raise Exception(response.text)

    data_d = BytesIO(response.content)

    with zipfile.ZipFile(data_d, 'r') as zip_ref:
        # Extract all the contents into the specified directory
        zip_ref.extractall()

    
    year = "2020"
    quarter = "Q1"
    filepath = f'./Divvy_Trips_{year}_{quarter}.csv'

    return FileIO().load(filepath)


@test
def test_output(output, *args) -> None:
    """
    Template code for testing the output of the block.
    """
    assert output is not None, 'The output is undefined'