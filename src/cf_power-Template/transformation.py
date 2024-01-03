
import pandas as pd
from indsl.ts_utils.operators import power

def main_A(data, ts_inputs):
    """Sample main function for transforming a set of input timeseries to
    produce a set of associated output time series.

    Args:
        data (dict): calculation-specfic parameters for Cognite Function
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

if __name__ == "__main__":
    ts = pd.DataFrame([1,4,2,3,76,3,1,6,34])
    ts_out = main_power({}, ts)
    print(ts_out)