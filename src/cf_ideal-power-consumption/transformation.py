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

def main_calculation_A(data, ts_inputs):
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

def main_calculation_B(data, ts_inputs):
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
