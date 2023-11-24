import pandas as pd
import numpy as np

def calc_ideal_power_consumption(data, ts_0, ts_1, ts_2, ts_3):
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
    ts_inputs = [ts_0, ts_1, ts_2, ts_3]

    ideal_power_consumption = pd.Series(0., index=ts_inputs[0].index)
    zero_condition = ts_inputs[3] > 0

    ts0_masked = ts_inputs[0][zero_condition]
    ts1_masked = ts_inputs[1][zero_condition]
    ts2_masked = ts_inputs[2][zero_condition]

    ideal_power_consumption[zero_condition] = ((ts0_masked / 3600) * ((ts1_masked * 1.2) - ts2_masked) * 10**5 / 1000) / (0.9 * 0.93 * 0.775)

    return ideal_power_consumption

