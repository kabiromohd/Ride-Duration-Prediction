#!/usr/bin/env python
# coding: utf-8

import warnings
warnings.filterwarnings("ignore")

import requests
import pandas as pd
import evidently
import pickle

import time
from datetime import date

from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import ColumnDriftMetric, DatasetDriftMetric, DatasetMissingValuesMetric
from evidently.metrics import ColumnQuantileMetric


def preprocess_data(df):
    
    df['duration'] = df["ended_at"] - df["started_at"]
    df['duration'] = df['duration'].apply(lambda td: td.total_seconds() / 60)

    categorical = ['start_station_id', 'end_station_id']
    df[categorical] = df[categorical].astype(str)
    

    reqd_cols = ['ride_id', 'start_station_id', 'end_station_id', 'duration']

    df = df[reqd_cols]
    
    return df

def extract_data_for_month(df, date_column, month):
    
    # Filter DataFrame for the specified month
    filtered_df = df[df[date_column].dt.month == month]

    return filtered_df


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


year = "2020"
quarter = "Q1"
filepath = f'./data/Divvy_Trips_{year}_{quarter}.csv'

data = pd.read_csv(filepath)

path = 'preprocess.bin'
loaded_model = 'model.pkl'

with open(path, 'rb') as f_in:
    dv = pickle.load(f_in)
    
    
with open(loaded_model, 'rb') as f_in:
    model = pickle.load(f_in)

# Change to datetime
data['started_at'] = pd.to_datetime(data['started_at'], errors='coerce')
data['ended_at'] = pd.to_datetime(data['ended_at'], errors='coerce')

data.fillna(0, inplace = True)
    

month_uniq = data['started_at'].dt.month.unique()

month1 = month_uniq[0]
month2 = month_uniq[1]
month3 = month_uniq[2]

df_month1 = extract_data_for_month(data, 'started_at', month1)
df_month2 = extract_data_for_month(data, 'started_at', month2)
df_month3 = extract_data

X_train, y_train = split_data(df_month3, dv, fit_dv=True)
X_val, y_val = split_data(df_month2, dv, fit_dv=False)
X_test, y_test = split_data(df_month1, dv, fit_dv=False)

train_data = df_month3
val_data = df_month2

train_preds = model.predict(X_train)
train_data['prediction'] = train_preds

val_preds = model.predict(X_val)
val_data['prediction'] = val_preds

val_data.to_parquet('data/reference.parquet')

# data labeling
target = "duration"
cat_features = ['ride_id', 'start_station_id', 'end_station_id']


# # Evidently Report
column_mapping = ColumnMapping(
    target=target,
    prediction='prediction',
    categorical_features=cat_features
)

report = Report(metrics=[
    ColumnDriftMetric(column_name='prediction'),
    DatasetDriftMetric(),
    DatasetMissingValuesMetric()
]
)

report.run(reference_data=train_data, current_data=val_data, column_mapping=column_mapping)




from evidently.metric_preset import DataDriftPreset, DataQualityPreset

from evidently.ui.workspace import Workspace
from evidently.ui.dashboards import DashboardPanelCounter, DashboardPanelPlot, CounterAgg, PanelValue, PlotType, ReportFilter
from evidently.renderers.html_widgets import WidgetSize

ws = Workspace("workspace")

project = ws.create_project("Ride Prediction Data Quality Project")
project.description = "Capstone final description"
project.save()

regular_report = Report(
    metrics=[
        DataQualityPreset(),
    ],
    timestamp=datetime.datetime(2020,2,28)
)

regular_report.run(reference_data=None,
                  current_data=val_data.loc[val_data.lpep_pickup_datetime.between('2020-02-27', '2020-02-28', inclusive="left")],
                  column_mapping=column_mapping)

regular_report


# In[28]:


regular_report = Report(
    metrics=[
        DataQualityPreset(),
    ],
    timestamp=datetime.datetime(2024,2,28)
)

regular_report.run(reference_data=None,
                  current_data=val_data.loc[val_data.lpep_pickup_datetime.between('2020-02-27', '2020-02-28', inclusive="left")],
                  column_mapping=column_mapping)



ws.add_report(project.id, regular_report)


#configure the dashboard
project.dashboard.add_panel(
    DashboardPanelCounter(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        agg=CounterAgg.NONE,
        title="Ride Prediction dashboard"
    )
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        title="Inference Count",
        values=[
            PanelValue(
                metric_id="DatasetSummaryMetric",
                field_path="current.number_of_rows",
                legend="count"
            ),
        ],
        plot_type=PlotType.BAR,
        size=WidgetSize.HALF,
    ),
)

project.dashboard.add_panel(
    DashboardPanelPlot(
        filter=ReportFilter(metadata_values={}, tag_values=[]),
        title="Number of Missing Values",
        values=[
            PanelValue(
                metric_id="DatasetSummaryMetric",
                field_path="current.number_of_missing_values",
                legend="count"
            ),
        ],
        plot_type=PlotType.LINE,
        size=WidgetSize.HALF,
    ),
)


project.save()


column_mapping = ColumnMapping(
    target=None,
    prediction='prediction',
    categorical_features=cat_features
)


day = 1
dateslist = list(pd.to_datetime(mar_data.lpep_pickup_datetime).dt.date)
while day < 32:
    if datetime.date(2020,3,day) in dateslist:
        
        regular_report = Report(
            metrics=[
                DataQualityPreset(),
            ],
            timestamp=datetime.datetime(2020,3,day)
        )
            
        regular_report.run(reference_data=None,
                          current_data=train_data.loc[mar_data.lpep_pickup_datetime.between(f"2020-03-{day}", f"2020-03-{day+1}", inclusive="left")],
                          column_mapping=column_mapping)
        
        ws.add_report(project.id, regular_report)
    day+=1

# Save the dashboard to an HTML file
project.save()
