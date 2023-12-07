

import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from datetime import datetime
import numpy as np

def main_daily_avg_drainage(data, ts):
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

    daily_avg_drainage = ts.groupby('Date')['derivative_excl_filling'].mean(
    )*data['tank_volume']/100  # avg drainage rate per DAY

    out_df = pd.DataFrame(daily_avg_drainage, index=daily_avg_drainage.index)
    out_df = out_df.rename(columns={out_df.columns[0]: ts_input_name})
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