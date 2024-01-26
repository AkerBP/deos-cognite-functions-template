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

from initialize import initialize_client
from cf_test.handler import handle
from tests.utils import create_timeseries, lowess_smoothing, make_dummy_df

# ----- UaT Tests -----

@pytest.mark.unit
def test_initial_transformation(cognite_client_mock):
    # Test that Cognite Function correctly transforms entire signal from initial date until recent date
    client = cognite_client_mock
    ts_input_name = "ts_input"
    ts_output_name = "ts_test"
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
    ts_input_name = "ts_input"
    ts_output_name = "ts_test"
    ts_input = client.time_series.retrieve(external_id=ts_input_name)
    # retrieve time series populated by test_initial_transformation
    ts_output = client.time_series.retrieve(external_id=ts_output_name)
    data = {"ts_input": ts_input, "test_run": True,
            "ts_output": ts_output}
    # Do transformation test inside handle() by querying data[test_run]
    new_df = handle(client, data)


class TestBackfilling:
    def __init__(self, cognite_client_mock, ts_input_name, ts_output_name):
        self.client = cognite_client_mock
        self.ts_input_name = ts_input_name
        self.ts_output_name = ts_output_name
        self.ts_input = self.client.time_series.data.retrieve(
            external_id=ts_input_name).to_pandas()
        self.ts_output = self.client.time_series.data.retrieve(
            external_id=ts_output_name).to_pandas()

        self.data = {"ts_input": self.ts_input, "test_run": True,
                     "ts_output": self.ts_output}

        self.ts_input_old = self.client.time_series.data.retrieve(external_id=self.ts_input_name,
                                                                  start=self.ts_input.index[int(
                                                                      len(self.ts_input)/4)],
                                                                  end=self.ts_input.index[int(len(self.ts_input)/2)]).to_pandas()
        self.ts_input_new = self.client.time_series.data.retrieve(external_id=self.ts_input_name,
                                                                  start=self.ts_input.index[int(
                                                                      len(self.ts_input)/3)],
                                                                  end=self.ts_input.index[int(2*len(self.ts_input)/3)]).to_pandas()

        common_dates = self.ts_input_new[self.ts_input_new.index.isin(
                                         self.ts_input_old.index)].index
        self.modified_date = common_dates[np.random.randint(
            0, len(common_dates))]

    def test_update_transformation(self):
        # Test that inserting new daily avg value for a date doesn't duplicate the date, but actually updates the value
        insert = pd.DataFrame([300, 200, 100], index=pd.date_range(
            start=self.ts_output.index[1], end=self.ts_output.index[3]), columns=[self.ts_output_name])
        self.client.time_series.data.insert_dataframe(insert)
        ts_output_new = self.client.time_series.data.retrieve(
            external_id=self.ts_output_name).to_pandas()
        assert ts_output_new.index == self.ts_output.index  # SAME - should overwrite dates
        # DIFFERENT - new values inserted
        assert ts_output_new[self.ts_output_name] != self.ts_output[self.ts_output_name]

    def test_start_end_dates_original_ts(self):
        # Assertion: Start date of old+new original signals are equal, and end date of new signal is equal or later than old signal
        ts_input_old = self.ts_input_old[self.ts_input_old.index >=
                                         self.ts_input_new.index[0]]
        assert ts_input_old.index[0] == self.ts_input_new.index[0]
        assert self.ts_input_new.index[-1] >= ts_input_old.index[-1]

    def test_inserted_data(self):
        # Assertion: If new data INSERTED for a date, cognite function should rerun and update daily avg value
        insert_data = pd.DataFrame(
            [1001, 2002], index=self.modified_date.append(self.modified_date), columns=self.ts_input_name)

        data = self.data
        data["ts_input"] = self.ts_input_new
        orig_transform_df = handle(self.client, data)

        # 3a. INSERTED data
        self.client.time_series.data.insert_dataframe(insert_data)

        ts_input_new_inserted = self.client.time_series.data.retrieve(external_id=self.ts_input_name,
                                                                      start=self.ts_input.index[int(
                                                                          len(self.ts_input)/3)],
                                                                      end=self.ts_input.index[int(2*len(self.ts_input)/3)])
        data["ts_input"] = ts_input_new_inserted
        new_transform_df = handle(self.client, data)

        self.orig_transform_date = orig_transform_df[orig_transform_df.index ==
                                                     self.modified_date].values
        new_transform_date = new_transform_df[new_transform_df.index ==
                                              self.modified_date].values

        assert new_transform_date != self.orig_transform_date  # different daily avg

    def test_deleted_data(self):
        # Assertion: If data DELETED for a date, cognite function should rerun and update daily avg value
        ts_input_new_del = self.ts_input_new
        del_idx = ts_input_new_del.index == self.modified_date
        start_date_del = None
        last_date_del = None

        for i, value in enumerate(del_idx):
            if value:
                if start_date_del is None:
                    start_date_del = i
                last_date_del = i

        ts_input_new_del = ts_input_new_del.drop(
            index=[start_date_del, last_date_del])  # drop two values for given date

        data = self.data
        data["ts_input"] = ts_input_new_del
        del_transform_df = handle(self.client, data)

        del_transform_date = del_transform_df[del_transform_df.index ==
                                              self.modified_date].values

        assert del_transform_date != self.orig_transform_date  # different daily avg

# --------------------------------


def test_aggregate():
    """Test that running aggregates return correct data over aggregating period (here: hourly).
    """
    date_range = pd.date_range(start=datetime(2023,12,1,0),
                               end=datetime(2023,12,1,2,30),
                               periods=15)
    data = [4,3,6,6,3,8,-3,2,5,10,20,2,5,1,10]
    df = pd.DataFrame(data, index=date_range, columns=["data"])

    df["date"] = pd.to_datetime(df.index).date
    df["hour"] = pd.to_datetime(df.index).hour

    hourly_avg = df.groupby(["date", "hour"])["data"].mean()

    hourly_avg_df = pd.DataFrame(hourly_avg, index=hourly_avg.index)
    hourly_avg_df = hourly_avg_df.rename(columns={hourly_avg_df.columns[0]: "data"})
    hourly_avg_df.index = pd.to_datetime(hourly_avg_df.index.get_level_values("Date").astype(str) + " " + hourly_avg_df.index.get_level_values("Hour").astype(str) + ":00:00")

    assert hourly_avg_df[hourly_avg_df.index.hour == 0]["data"] == 5
    assert hourly_avg_df[hourly_avg_df.index.hour == 1]["data"] == 6
    assert hourly_avg_df[hourly_avg_df.index.hour == 2]["data"] == 16/3


# --------------------------------
