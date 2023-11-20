import pandas as pd
import numpy as np
import sys
import os

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from transformation_utils import get_input_ts, align_time_series, store_output_ts

def run_transformation(data):
    """Run calculations on input time series, producing a new time series.

    Args:
        data (dict): input parameters for the Cognite Function

    Returns:
        list: output time series listed as dataframes
    """
    ts_data = get_input_ts(data)
    ts_data = align_time_series(ts_data, data) # align input time series to cover same time period
    # --- calculation function to be defined below ---
    ts_output = calculation(data, *ts_data) # NB: order of time series in ts_data must correspond to order of input arguments (ts_0, ts_1, ...)
    # ----------------------------------------------------------------------
    out_list = store_output_ts(ts_output, data)

    return out_list


def calculation(data, ts_4, ts_5, ts_ipc):
    """Compute time series of wasted energy.

    Args:
        data (dict): parameters and data for Cognite Function
        ts_0 (pd.DataFrame): time series input 'VAL_11-XT-95067B:Z.X.Value'
        ts_1 (pd.DataFrame): time series of fixed value
        ts_ipc (pd.DataFrame): time series calculated by cf_ideal-power-consumption, 'VAL_11-PT-92363B:X.Value'

    Returns:
        list: time series of ideal power consumption
    """
    ts_0 = np.mean(ts_0, dtype=int)
    if ts_4 > 0:
        wasted_energy = ts_5 - ts_ipc
    else:
        wasted_energy = np.zeros(len(ts_ipc))

    return [wasted_energy]

