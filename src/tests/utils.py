import pandas as pd
import numpy as np
import random
from cognite.client.data_classes import TimeSeries
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.tsa.seasonal import seasonal_decompose


def create_timeseries(cognite_client_mock, ts_test_name):
    ccm = cognite_client_mock
    new_ts = ccm.time_series.create(TimeSeries(
        name=ts_test_name, external_id=ts_test_name))  # data_set_id=dataset_test_id
    return new_ts


def lowess_smoothing(df: pd.DataFrame, ts_name: str):
    """Filter input signal using lowess smoothing

    Args:
        df (pd.DataFrame): Aggregated time series
        ts_name (str): name of time series
    """
    frac_data = 0.01
    smooth = lowess(df[ts_name], df['time_sec'],
                    is_sorted=True, frac=frac_data, it=0)
    return smooth, frac_data


def make_dummy_df(ts_name: str, periods: int):
    t = pd.date_range(start="2022", freq="1d", periods=periods)
    random.seed(345)
    y = np.random.normal(0, 1, periods)
    df = pd.DataFrame({ts_name: y}, index=t)
    return df
