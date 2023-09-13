#   ---------------------------------------------------------------------------------
#   Copyright (c) Microsoft Corporation. All rights reserved.
#   Licensed under the MIT License. See LICENSE in project root for information.
#   ---------------------------------------------------------------------------------
"""
This is a configuration file for pytest containing customizations and fixtures.

In VSCode, Code Coverage is recorded in config.xml. Delete this file to reset reporting.
"""

from __future__ import annotations

from typing import List
from src.initialize import initialize_client
import pandas as pd
from datetime import datetime, timedelta

import pytest
from _pytest.nodes import Item

@pytest.fixture(scope="module")
def retrieve_timeseries() -> pd.DataFrame:
    """Retrieve a given CDF time series

    Returns:
        pd.DataFrame: time series stored in dataframe
    """
    client = initialize_client()
    ts_name = "VAL_11-LT-95034A:X.Value"

    ts_all = client.time_series.search(name=ts_name) # find time series by name
    cdf_ext_id = ts_all[0].external_id # extract its external id
    start_date = datetime(2023, 3, 21, 1, 0, 0)
    df_cdf = client.time_series.data.retrieve(external_id=cdf_ext_id,
                                        aggregates="average",
                                        granularity="2m",
                                        start=start_date,
                                        end=start_date + timedelta(days=20)) # load time series by external id

    df = df_cdf.to_pandas()
    df = df.rename(columns = {cdf_ext_id + "|average": ts_name})

    df['time_sec'] = (df.index - datetime(1970,1,1)).total_seconds() # total seconds elapsed of each data point since 1970
    return df, ts_name

def pytest_collection_modifyitems(items: list[Item]):
    for item in items:
        if "spark" in item.nodeid:
            item.add_marker(pytest.mark.spark)
        elif "_int_" in item.nodeid:
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def unit_test_mocks(monkeypatch: None):
    """Include Mocks here to execute all commands offline and fast."""
    pass
