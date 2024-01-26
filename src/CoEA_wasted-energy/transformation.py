import pandas as pd
import numpy as np

def main_wasted_energy(data, ts_inputs):
    """Compute time series of wasted energy.

    Args:
        data (dict): calculation-specfic parameters for Cognite Function
        ts_inputs (pd.DataFrame): time series inputs: 'VAL_11-XT-95067B:Z.X.Value', float, 'TEST_IdealPowerConsumption'

    Returns:
        pd.DataFrame: data points of output time series
    """
    wasted_energy = pd.DataFrame(np.zeros(ts_inputs.shape[0]), columns=list(data["ts_output"].keys()), index=ts_inputs[ts_inputs.columns[0]].index)

    zero_condition = ts_inputs.iloc[:,0] > 0
    nan_condition = ts_inputs.isna().any(axis=1)

    ts1_masked = ts_inputs.iloc[:,1][zero_condition].to_frame().values
    ts_ipc_masked = ts_inputs.iloc[:,2][zero_condition].to_frame().values

    wasted_energy[zero_condition] = ts1_masked - ts_ipc_masked
    wasted_energy[nan_condition] = np.nan

    return wasted_energy