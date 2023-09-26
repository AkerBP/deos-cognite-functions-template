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
import pandas as pd
from datetime import datetime, timedelta

import pytest
from _pytest.nodes import Item

from cognite.client.testing import monkeypatch_cognite_client
from src.initialize import initialize_client


@pytest.fixture
def cognite_client_mock():
    with monkeypatch_cognite_client() as client:
        yield client


@pytest.fixture(scope="module")
def retrieve_timeseries():
    """Retrieve a given CDF time series

    Returns:
        pd.DataFrame: time series stored in dataframe
    """
    client = initialize_client(
        run_sandbox=True, cache_token=True)  # Instantiate sandbox environment
    ts_name = "PI-70445:X.Value"

    ts_elem = client.time_series.list(
        name=ts_name).to_pandas()
    cdf_ext_id = ts_elem.external_id[0]  # extract its external id
    # start_date = datetime(2023, 3, 21, 1, 0, 0)
    df_cdf = client.time_series.data.retrieve(external_id=cdf_ext_id,
                                              aggregates="average",
                                              granularity="15m")

    df = df_cdf.to_pandas()
    df = df.rename(columns={cdf_ext_id + "|average": ts_name})

    # total seconds elapsed of each data point since 1970
    df['time_sec'] = (df.index - datetime(1970, 1, 1)).total_seconds()
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
