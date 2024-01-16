
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

    aggregate = data["aggregate"]
    df_out = pd.DataFrame(index=ts_inputs.index)
    calc_params = data["calc_params"]

    ts_out = filter_ts(ts_inputs, calc_params)

    derivative_value_excl = calc_params['derivative_value_excl']
    agg_period = {"second":"S", "minute":"T", "hour":"H", "day":"D", "month":"M", "year":"Y"}

    for ts in ts_inputs.columns:
        try:
            ts_out[ts+"_derivative"] = np.gradient(ts_out[ts], ts_out["time_sec"])
        except:
            raise IndexError(
                f"No datapoints found for selected date range for time series {ts}. Cannot compute drainage rate.")

        ts_out[ts+"_drainage"] = ts_out[ts+"_derivative"].apply(
            lambda x: 0 if x > derivative_value_excl or pd.isna(x) else x)  # not interested in large INLET fluxes

        if "period" in aggregate and "type" in aggregate:
            df_out[ts] = ts_out.resample(agg_period[aggregate["period"]])[ts+"_drainage"].agg(aggregate["type"])
        else:
            df_out[ts] = ts_inputs[ts] # if no aggregates, make no transformations

    # CF requires dataframe to only have transformed time series as columns. Delete others.
    df_out = df_out.drop("time_sec", axis=1)

    return df_out

def filter_ts(ts_inputs, data):
    """Helper function: performs lowess smoothing

    Args:
        ts (pd.DataFrame): input time series
        data (dict): calculation-specific parameters for Cognite Function

    Returns:
        pd.DataFrame: smoothed signal
    """
    from datetime import datetime
    from statsmodels.nonparametric.smoothers_lowess import lowess

    df_smooth = pd.DataFrame()
    ts_inputs["time_sec"] = (ts_inputs.index - datetime(1970, 1, 1)).total_seconds()

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
