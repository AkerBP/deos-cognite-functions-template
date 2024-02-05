
import os
import sys
# Set file to system path to allow relative import from parent folder
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from utilities import AGG_PERIOD
import pandas as pd

def main_aggregate(data, ts_inputs):
    """Sample main function for performing aggregations (here: average drainage).

    Args:
        data (dict): parameters for Cognite Function
        ts_inputs (pd.DataFrame): input time series to transform, one column per time series

    Returns:
        (pd.DataFrame): transformed time series, one column per time series
    """
    import numpy as np

    try:
        aggregate = data["optional"]["aggregate"]
    except:
        raise KeyError(f"Aggregated calculations require 'aggregate' to be specified as an optional parameter to input data dictionary. Only found the optional parameters: {data['optional'].keys()}")

    out_index = ts_inputs.index.floor(AGG_PERIOD[aggregate["period"]]).unique()

    calc_params = data["calc_params"]
    ts_out = filter_ts(ts_inputs, calc_params)

    derivative_value_excl = calc_params['derivative_value_excl']

    input_cols = list([col for col in ts_inputs.columns if col != "time_sec"])
    output_cols = list([col for col in data["ts_output"].keys() if col != "time_sec"])

    df_out = pd.DataFrame(index=out_index, columns=output_cols)

    for in_col, out_col in zip(input_cols, output_cols):

        try:
            ts_out[in_col+"_derivative"] = np.gradient(ts_out[in_col], ts_out["time_sec"])
        except:
            raise IndexError(
                f"No datapoints found for selected date range for time series {in_col}. Cannot compute drainage rate.")

        ts_out[in_col+"_drainage"] = ts_out[in_col+"_derivative"].apply(
            lambda x: 0 if x > derivative_value_excl or pd.isna(x) else x)  # not interested in large INLET fluxes

        df_out[out_col] = ts_out.resample(AGG_PERIOD[aggregate["period"]])[in_col+"_drainage"].agg(aggregate["type"]).values

    return df_out

def filter_ts(ts_inputs, data):
    """Helper function: performs lowess smoothing

    Args:
        ts (pd.DataFrame): input time series
        data (dict): calculation-specific parameters for Cognite Function

    Returns:
        pd.DataFrame: smoothed signal
    """
    from datetime import datetime, timezone
    from statsmodels.nonparametric.smoothers_lowess import lowess

    df_smooth = pd.DataFrame()
    ts_inputs["time_sec"] = (ts_inputs.index - datetime(1970, 1, 1,
                                                        tzinfo=timezone.utc)).total_seconds()

    if "lowess_frac" in data:
        frac = data["lowess_frac"]
    else:
        frac = 0.01
    if "lowess_delta" in data:
        delta = data["lowess_delta"]
    else:
        delta = 0

    ts_input_names = [name for name in ts_inputs.columns if name != "time_sec"]
    for ts in ts_input_names:
        data = ts_inputs[ts]

        smooth = lowess(data, ts_inputs["time_sec"], is_sorted=True,
                        frac=frac, it=0, delta=delta*len(data))

        df_smooth[ts] = smooth[:,1] # only extract values, not timestamp

    df_smooth["time_sec"] = smooth[:,0]
    df_smooth.index = pd.to_datetime(df_smooth["time_sec"], unit="s")

    return df_smooth

def main_test(data, ts_inputs):
    """Sample main function for transforming a set of input timeseries to
    produce a set of associated output time series.

    Args:
        data (dict): parameters for Cognite Function
        ts_inputs (pd.DataFrame): input time series to transform, one column per time series

    Returns:
        (pd.DataFrame): transformed time series, one column per time series
    """
    ts_out = pd.DataFrame()

    for ts in ts_inputs.columns:
        ts_out[ts] = ts_inputs[ts].dropna().rolling(window=10).mean().reindex(ts_inputs[ts].index, method="nearest")

    return ts_out

def main_B(data, ts_inputs):
    """Other sample main function for transforming a set of input timeseries to
    produce a set of associated output time series.

    Args:
        data (dict): parameters for Cognite Function
        ts_inputs (pd.DataFrame): input time series to transform, one column per time series

    Returns:
        (pd.DataFrame): transformed time series, one column per time series
    """
    ts_0 = ts_inputs.iloc[:,0] # first time series in dataframe
    ts_1 = ts_inputs.iloc[:,1] # second time series ...
    ts_2 = ts_inputs.iloc[:,2]

    ts_out = ts_0.max() + 2*ts_1 + ts_0*ts_2/5

    return pd.DataFrame(ts_out) # ensure output given as dataframe
