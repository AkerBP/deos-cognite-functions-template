import numpy as np
import pandas as pd

def calculation(data, ts_0, ts_1, ts_ipc):
    """Compute time series of wasted energy.

    Args:
        data (dict): parameters and data for Cognite Function
        ts_0 (pd.DataFrame): time series input 'VAL_11-XT-95067B:Z.X.Value'
        ts_1 (pd.DataFrame | int): time series or constant
        ts_ipc (pd.DataFrame): time series calculated by cf_ideal-power-consumption, 'VAL_11-PT-92363B:X.Value'

    Returns:
        list: time series of ideal power consumption
    """
    wasted_energy = pd.Series(0., index=ts_0.index)
    if isinstance(ts_1, float) or isinstance(ts_1, int):
        ts_1 = pd.Series(ts_1, index=ts_0.index)

    zero_condition = ts_0 > 0

    ts0_masked = ts_0[zero_condition]
    ts1_masked = ts_1[zero_condition]
    ts_ipc_masked = ts_ipc[zero_condition]

    wasted_energy[zero_condition] = ts1_masked - ts_ipc_masked

    return [wasted_energy]

