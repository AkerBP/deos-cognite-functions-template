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
# from cognite.client import CogniteClient
# from cognite.client.config import ClientConfig
# from cognite.client.credentials import OAuthInteractive, OAuthClientCredentials
from cognite.client.data_classes import TimeSeries

from src.initialize import initialize_client
from src.handler import handle
from tests.utils import create_timeseries, lowess_smoothing, make_dummy_df

# ----- CDF specific tests -----


@pytest.mark.unit
def test_timeseries_retrieval(retrieve_timeseries):
    df_test = retrieve_timeseries
    # Test time series aggregation of 2 min
    dt = np.diff(df_test.index)
    assert np.all(dt == 120)  # 2 min = 120 sec
    # Test number of data points. Should be 30*24*20 = 14400 datapoints for timedelta of 20 days
    assert len(df_test.index) == 14400
    # Test start and end dates
    assert (df_test.index[0] == datetime(2023, 3, 21, 1, 0, 0))
    assert (df_test.index[-1] == datetime(2023,
            3, 21, 1, 0, 0) + timedelta(days=20))


@pytest.mark.unit
def test_timeseries_create(cognite_client_mock):
    client = cognite_client_mock
    new_ts = create_timeseries(client, "ts_test_create")
    num_datapoints = len(client.time_series.data.retrieve(id=new_ts.id))
    # Check that correct data object created
    assert (type(new_ts) == TimeSeries)
    # Check that new time series is empty
    assert (num_datapoints == 0)


@pytest.mark.unit
def test_timeseries_insert(cognite_client_mock):
    # client = initialize_client()
    client = cognite_client_mock
    ts_name = "ts_test_insert"
    new_ts = create_timeseries(client, ts_name)
    df = make_dummy_df(ts_name, 40)

    client.time_series.data.insert_dataframe(df, external_id_headers=False)

    ts_retrieved = client.time_series.data.retrieve(id=new_ts.id)
    num_datapoints = len(ts_retrieved)
    vals = ts_retrieved.value
    # Check that all datapoints have been inserted in time series
    assert (num_datapoints == 40)
    # Check that all datapoints are correct
    y = df[ts_name].values
    assert (np.all(vals == y))

# --------------------------------


# ----- UaT Tests -----

@pytest.mark.unit
def test_initial_transformation(cognite_client_mock):
    # Test that Cognite Function correctly transforms entire signal from initial date until recent date
    client = cognite_client_mock
    ts_input_name = "ts_input_initial"
    ts_output_name = "ts_output_initial"
    ts_input = client.time_series.retrieve(external_id=ts_input_name)
    ts_output = create_timeseries(client, ts_output_name)
    # NB: Only create time series here - insertion done in handle()
    data = {"ts_input": ts_input, "test_run": True,
            "ts_output": ts_output}
    new_df = handle(client, data)
    # Save/insert initial data in the created time series, so it can be accessed by test_continuous_transformation
    # (can be saved inside handle)


@pytest.mark.unit
def test_continuous_transformation(cognite_client_mock):
    # Test that Cognite Function works for calculating daily average of today only (or most recent write)
    client = cognite_client_mock
    ts_input_name = "ts_input_continuous"
    ts_output_name = "ts_output_continuous"
    ts_input = client.time_series.retrieve(external_id=ts_input_name)
    ts_output = client.time_series.retrieve(external_id=ts_output_name) # retrieve time series populated by test_initial_transformation
    data = {"ts_input": ts_input, "test_run": True,
            "ts_output": ts_output}
    # Do transformation test inside handle() by querying data[test_run]
    new_df = handle(client, data)

# --------------------------------


# ----- Mathematical tests -----

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
            assert smooth[i] >= min(smooth[i-smooth_threshold:i+smooth_threshold]
                                    ) and smooth[i] <= max(smooth[i-smooth_threshold:i+smooth_threshold])


def test_derivative(retrieve_timeseries):
    """Test derivative calculation
    """
    df, ts_name = retrieve_timeseries
    smooth, frac_data = lowess_smoothing(df, ts_name)
    df['smooth'] = smooth
    # Test boundedness of derivative signal
    deriv_excl = 0.001
    # Unit: vol_percentage/time [% of tank vol / sec]
    df["deriv"] = np.gradient(df['smooth'], df["time_sec"])
    deriv_bnd = df["deriv"].apply(
        lambda x: 0 if x > deriv_excl or pd.isna(x) else x)
    assert np.all(deriv_bnd <= deriv_excl)

# --------------------------------
