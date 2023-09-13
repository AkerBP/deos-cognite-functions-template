#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""This is a sample python file for testing functions from the source code."""
from __future__ import annotations
import pytest

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from cognite.client import CogniteClient
from cognite.client.config import ClientConfig
from cognite.client.credentials import OAuthInteractive, OAuthClientCredentials
from cognite.client.data_classes import TimeSeries
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import seasonal_decompose

from src.initialize import initialize_client
from src.handler import handle

def create_timeseries(client: CogniteClient, ts_test_name: str):
    dataset_test_id = 123456789
    new_ts = client.time_series.create(TimeSeries(name=ts_test_name, external_id=ts_test_name, data_set_id=dataset_test_id))
    return new_ts

def lowess_smoothing(df: pd.DataFrame, ts_name: str):
    """Filter input signal using lowess smoothing

    Args:
        df (pd.DataFrame): Aggregated time series
        ts_name (str): name of time series
    """
    frac_data = 0.01
    smooth = lowess(df[ts_name], df['time_sec'], is_sorted=True, frac=frac_data, it=0)
    return smooth, frac_data


def test_timeseries_retrieval(retrieve_timeseries):
    df_test = retrieve_timeseries
    # Test time series aggregation of 2 min
    dt = np.diff(df_test.index)
    assert np.all(dt == 120) # 2 min = 120 sec
    # Test number of data points. Should be 30*24*20 = 14400 datapoints for timedelta of 20 days
    assert len(df_test.index) == 14400
    # Test start and end dates
    assert(df_test.index[0] == datetime(2023, 3, 21, 1, 0, 0))
    assert(df_test.index[-1] == datetime(2023, 3, 21, 1, 0, 0) + timedelta(days=20))

def test_timeseries_create():
    client = initialize_client()
    new_ts = create_timeseries(client, "ts_test_create")
    num_datapoints = len(client.time_series.data.retrieve(id=new_ts.id))
    # Check that correct data object created
    assert(type(new_ts) == TimeSeries)
    # Check that new time series is empty
    assert(num_datapoints == 0)

def test_timeseries_insert():
    client = initialize_client()
    ts_name = "ts_test_insert"
    new_ts = create_timeseries(client, ts_name)
    t = pd.date_range(start="2023", freq="1d", periods=40)
    y = np.random.normal(0, 1, 40)
    df = pd.DataFrame({ts_name: y}, index=t)

    client.time_series.data.insert_dataframe(df, external_id_headers=False)

    ts_retrieved = client.time_series.data.retrieve(id=new_ts.id)
    num_datapoints = len(ts_retrieved)
    vals = ts_retrieved.value
    # Check that all datapoints have been inserted in time series
    assert(num_datapoints == 40)
    # Check that all datapoints are correct
    assert(np.all(vals == y))


def test_smoothing(retrieve_timeseries):
    """Test lowess_smoothing function
    """
    df, ts_name = retrieve_timeseries
    original = df[ts_name]
    smooth, frac_data = lowess_smoothing(df, ts_name)
    # Test that smooth signal is bounded by extreme values of sampled values from original signal
    num_data = len(df.index)
    smooth_threshold = round((frac_data/2)*num_data)
    for i in range(num_data):
        if i < smooth_threshold or i > num_data-smooth_threshold:
            continue
        else:
            assert smooth[i] >= min(smooth[i-smooth_threshold:i+smooth_threshold]) and smooth[i] <= max(smooth[i-smooth_threshold:i+smooth_threshold])


def test_derivative(retrieve_timeseries):
    """Test derivative calculation
    """
    df, ts_name = retrieve_timeseries
    smooth, frac_data = lowess_smoothing(df, ts_name)
    df['smooth'] = smooth
    # Test boundedness of derivative signal
    deriv_excl = 0.001
    df["deriv"] = np.gradient(df['smooth'], df["time_sec"]) # Unit: vol_percentage/time [% of tank vol / sec]
    deriv_bnd = df["deriv"].apply(lambda x: 0 if x > deriv_excl or pd.isna(x) else x)
    assert np.all(deriv_bnd <= deriv_excl)
