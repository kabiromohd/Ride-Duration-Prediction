#!/usr/bin/env python
# coding: utf-8

import requests

host = "ride-duration-prediction.onrender.com"

url = f'https://{host}/predict'

client_pred = {
    "ride_id": "EACB19130B0CDA4A",
    "start_station_id": "239",
    "end_station_id": "326.0"    
}

response = requests.post(url, json=client_pred).json()
print(response)
