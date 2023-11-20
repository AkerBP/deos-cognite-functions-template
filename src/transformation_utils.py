import numpy as np
import pandas as pd

def get_input_ts(data):
    ts_data = data["ts_input_today"]
    ts_data = [ts_data[name] for name in ts_data]
    return ts_data

def align_time_series(ts_df, data):
    latest_start_date = np.max([ts_df[i].index[0] for i in range(len(ts_df))])
    earliest_end_date = np.min([ts_df[i].index[-1] for i in range(len(ts_df))])

    for i in range(len(ts_df)): # omit dates where some of time series have nan values
        ts_df[i] = ts_df[i][ts_df[i].index >= latest_start_date]
        ts_df[i] = ts_df[i][ts_df[i].index <= earliest_end_date]

    time_index = pd.date_range(start=latest_start_date, end=earliest_end_date, freq=f"{data['granularity']}s")

    for i in range(len(ts_df)):
        ts_df[i] = ts_df[i].reindex(time_index, copy=False) # missing internal dates are filled with nan

    return ts_df

def store_output_ts(ts_output, data):
    """Store output time series in separate dataframes

    Args:
        ts_output (list): datapoints for output time series
        data (dict): input parameters for Cognite Function

    Returns:
        pd.DataFrame: dataframes of output time series
    """
    out_df = pd.DataFrame()

    for ts_out_name, ts_out_val in zip(data["ts_output"].keys(), ts_output):
        added_df = pd.DataFrame({ts_out_name: ts_out_val})
        out_df = pd.concat([out_df, added_df], axis=1) # concatenate on dates

    out_df.index = pd.to_datetime(out_df.index)

    return out_df
