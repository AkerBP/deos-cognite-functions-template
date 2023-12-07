import pandas as pd
import numpy as np

def main_ideal_power_consumption(data, ts_inputs):
    """Calculate ideal power consumption from four input time series.

    Args:
        data (dict): calculation-specific parameters for Cognite Function
        ts_0 (pd.DataFrame): time series input 'VAL_17-FI-9101-286:VALUE'
        ts_1 (pd.DataFrame): time series input 'VAL_17-PI-95709-258:VALUE'
        ts_2 (pd.DataFrame): time series input 'VAL_11-PT-92363B:X.Value'
        ts_3 (pd.DataFrame): time series input 'VAL_11-XT-95067B:Z.X.Value'

    Returns:
        pd.Series: data points of output time series
    """
    ideal_power_consumption = pd.DataFrame(np.zeros(ts_inputs.shape[0]), columns=["TEST_IdealPowerConsumption"], index=ts_inputs[ts_inputs.columns[0]].index)
    zero_condition = ts_inputs.iloc[:,3] > 0

    ts0_masked = ts_inputs.iloc[:,0][zero_condition].to_frame().values
    ts1_masked = ts_inputs.iloc[:,1][zero_condition].to_frame().values
    ts2_masked = ts_inputs.iloc[:,2][zero_condition].to_frame().values

    ideal_power_consumption[zero_condition] = ((ts0_masked / 3600) * ((ts1_masked * 1.2) - ts2_masked) * 10**5 / 1000) / (0.9 * 0.93 * 0.775)

    return ideal_power_consumption