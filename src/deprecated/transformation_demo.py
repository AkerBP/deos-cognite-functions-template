import pandas as pd
import numpy as np

def main_ideal_power_consumption(data, ts_inputs):
    """Calculate ideal power consumption from four input time series.

    Args:
        data (dict): calculation-specific parameters for Cognite Function
        ts_inputs (pd.DataFrame): time series inputs 'VAL_17-FI-9101-286:VALUE', 'VAL_17-PI-95709-258:VALUE', 'VAL_11-PT-92363B:X.Value', 'VAL_11-XT-95067B:Z.X.Value'

    Returns:
        pd.DataFrame: data points of output time series
    """
    ideal_power_consumption = pd.DataFrame(np.zeros(ts_inputs.shape[0]), columns=["TEST_IdealPowerConsumption"], index=ts_inputs[ts_inputs.columns[0]].index)
    zero_condition = ts_inputs.iloc[:,3] > 0

    ts0_masked = ts_inputs.iloc[:,0][zero_condition].to_frame().values
    ts1_masked = ts_inputs.iloc[:,1][zero_condition].to_frame().values
    ts2_masked = ts_inputs.iloc[:,2][zero_condition].to_frame().values

    ideal_power_consumption[zero_condition] = ((ts0_masked / 3600) * ((ts1_masked * 1.2) - ts2_masked) * 10**5 / 1000) / (0.9 * 0.93 * 0.775)

    return ideal_power_consumption

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