import pandas as pd
import numpy as np
import os
import sys

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
    out_df = store_output_ts(ts_output, data)

    return out_df


def calculation(data, ts_0, ts_1, ts_2, ts_3):
    """Calculate ideal power consumption from four input time series.

    Args:
        data (dict): parameters and data for Cognite Function
        ts_0 (pd.DataFrame): time series input 'VAL_17-FI-9101-286:VALUE'
        ts_1 (pd.DataFrame): time series input 'VAL_17-PI-95709-258:VALUE'
        ts_2 (pd.DataFrame): time series input 'VAL_11-PT-92363B:X.Value'
        ts_3 (pd.DataFrame): time series input 'VAL_11-XT-95067B:Z.X.Value'

    Returns:
        list: time series of ideal power consumption
    """
    ts_inputs = [ts_0, ts_1, ts_2, ts_3]

    ideal_power_consumption = pd.Series(np.zeros(len(ts_inputs[0])), index=ts_inputs[0].index)
    zero_condition = ts_inputs[3] > 0

    ts0_masked = ts_inputs[0][zero_condition]
    ts1_masked = ts_inputs[1][zero_condition]
    ts2_masked = ts_inputs[2][zero_condition]

    ideal_power_consumption[zero_condition] = ((ts0_masked / 3600) * ((ts1_masked * 1.2) - ts2_masked) * 10**5 / 1000) / (0.9 * 0.93 * 0.775)

    return [ideal_power_consumption]

