
import pandas as pd
import numpy as np

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

def main_wasted_energy(data, ts_inputs):
    """Compute time series of wasted energy.

    Args:
        data (dict): calculation-specfic parameters for Cognite Function
        ts_inputs (pd.DataFrame): time series inputs: 'VAL_11-XT-95067B:Z.X.Value', float, 'TEST_IdealPowerConsumption'

    Returns:
        pd.DataFrame: data points of output time series
    """
    wasted_energy = pd.DataFrame(np.zeros(ts_inputs.shape[0]), columns=["TEST_WastedEnergy"], index=ts_inputs.index)
    if isinstance(ts_inputs.iloc[:,1], float) or isinstance(ts_inputs.iloc[:,1], int):
        ts_1 = pd.DataFrame(ts_inputs.iloc[:,1]*np.ones(ts_inputs.shape[0]), index=ts_inputs.index)

    zero_condition = ts_inputs.iloc[:,0] > 0

    ts1_masked = ts_inputs.iloc[:,1][zero_condition].to_frame().values
    ts_ipc_masked = ts_inputs.iloc[:,2][zero_condition].to_frame().values

    wasted_energy[zero_condition] = ts1_masked - ts_ipc_masked

    return wasted_energy
