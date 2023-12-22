
import pandas as pd

import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from datetime import datetime
import numpy as np

def main_hourly_avg_drainage(data, ts):
    """Calculation function

    Args:
        data (dict): calculation-specific parameters for Cognite Function
        ts (pd.DataFrame): (single) input time series

    Returns:
        pd.DataFrame: data points for transformed signal
    """
    ts, ts_input_name = filter_ts(ts, data)

    try:
        ts["derivative"] = np.gradient(ts['smooth'], ts["time_sec"])
    except:
        raise IndexError(
            "No datapoints found for selected date range. Cannot compute drainage rate.")

    derivative_value_excl = data['derivative_value_excl']
    ts['derivative_excl_filling'] = ts["derivative"].apply(
        lambda x: 0 if x > derivative_value_excl or pd.isna(x) else x)  # not interested in large INLET fluxes

    ts.reset_index(inplace=True)
    ts.index = pd.to_datetime(ts['time_stamp'])
    ts['Date'] = ts.index.date
    ts["Date"] = pd.to_datetime(ts["Date"])
    ts['Hour'] = ts.index.hour

    daily_avg_hour = ts.groupby(['Date', 'Hour'])['derivative_excl_filling'].mean(
    )*data['tank_volume']/100  # avg drainage rate per HOUR
    out_df = pd.DataFrame(daily_avg_hour, index=daily_avg_hour.index)
    out_df = out_df.rename(columns={out_df.columns[0]: ts_input_name})
    # Convert MultiIndex to single Datetime index
    out_df.index = pd.to_datetime(out_df.index.get_level_values("Date").astype(str) + " " + out_df.index.get_level_values("Hour").astype(str) + ":00:00")

    return out_df

def filter_ts(ts, data):
    """Helper function: performs lowess smoothing

    Args:
        ts (pd.DataFrame): (single) input time series
        data (dict): calculation-specific parameters for Cognite Function

    Returns:
        pd.DataFrame: smoothed signal
    """
    ts_input_name = ts.columns[0]
    vol_perc = ts[ts_input_name]
    ts["time_sec"] = (ts.index - datetime(1970, 1, 1)).total_seconds()

    if "lowess_frac" in data:
        frac = data["lowess_frac"]
    else:
        frac = 0.01
    if "lowess_delta" in data:
        delta = data["lowess_delta"]
    else:
        delta = 0
    smooth = lowess(vol_perc, ts['time_sec'], is_sorted=True,
                    frac=frac, it=0, delta=delta*len(ts))

    df_smooth = pd.DataFrame(smooth, columns=["time_sec", "smooth"])

    ts.reset_index(inplace=True)
    ts = ts.rename(columns={'index': 'time_stamp'})
    ts = pd.merge(ts, df_smooth, on='time_sec')
    ts.set_index('time_stamp', drop=True, append=False,
                 inplace=True, verify_integrity=False)

    return ts, ts_input_name

def main_B(data, ts_inputs):
    """Other sample main function for transforming a set of input timeseries to
    produce a set of associated output time series.

    Args:
        data (dict): calculation-specfic parameters for Cognite Function
        ts_inputs (pd.DataFrame): input time series to transform, one column per time series

    Returns:
        (pd.DataFrame): transformed time series, one column per time series
    """
    ts_0 = ts_inputs.iloc[:,0] # first time series in dataframe
    ts_1 = ts_inputs.iloc[:,1] # second time series ...
    ts_2 = ts_inputs.iloc[:,2]

    ts_out = ts_0.max() + 2*ts_1 + ts_0*ts_2/5

    return pd.DataFrame(ts_out) # ensure output given as dataframe

def main_exp(data, ts_inputs):
    """Other sample main function for transforming a set of input timeseries to
    produce a set of associated output time series.

    Args:
        data (dict): calculation-specfic parameters for Cognite Function
        ts_inputs (pd.DataFrame): input time series to transform, one column per time series

    Returns:
        (pd.DataFrame): transformed time series, one column per time series
    """
    ts_out = exp(ts_inputs.iloc[:,0])
    ts_out = ts_out / 10.0

    return pd.DataFrame(ts_out) # ensure output given as dataframe
