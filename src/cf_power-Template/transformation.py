import pandas as pd
import numpy as np
from indsl.ts_utils.operators import power

def main_power(data, ts_inputs):
    """Sample main function for transforming a set of input timeseries to
    produce a set of associated output time series.

    Args:
        data (dict): calculation-specfic parameters for Cognite Function
        ts_inputs (pd.DataFrame): input time series to transform, one column per time series

    Returns:
        (pd.DataFrame): transformed time series, one column per time series
    """
    ts_out = power(ts_inputs.squeeze(), 3)
    return pd.DataFrame(ts_out, index=ts_out.index)