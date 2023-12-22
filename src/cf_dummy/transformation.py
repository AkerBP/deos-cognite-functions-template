

import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
from datetime import datetime
import numpy as np
from indsl.ts_utils import exp

def main_daily_avg_drainage(data, ts):
    """Calculation function

    Args:
        data (dict): calculation-specific parameters for Cognite Function
        ts (pd.DataFrame): (single) input time series

    Returns:
        pd.DataFrame: data points for transformed signal
    """
    ts, ts_input_name = filter_ts(ts, data)

    try:
        ts["derivative"] = np.gradient(ts['smooth'], ts["time_sec"])
    except:
        raise IndexError(
            "No datapoints found for selected date range. Cannot compute drainage rate.")

    derivative_value_excl = data['derivative_value_excl']
    ts['derivative_excl_filling'] = ts["derivative"].apply(
        lambda x: 0 if x > derivative_value_excl or pd.isna(x) else x)  # not interested in large INLET fluxes

    ts.reset_index(inplace=True)
    ts.index = pd.to_datetime(ts['time_stamp'])
    ts['Date'] = ts.index.date
    ts["Date"] = pd.to_datetime(ts["Date"])

    daily_avg_drainage = ts.groupby('Date')['derivative_excl_filling'].mean(
    )*data['tank_volume']/100  # avg drainage rate per DAY

    out_df = pd.DataFrame(daily_avg_drainage, index=daily_avg_drainage.index)
    out_df = out_df.rename(columns={out_df.columns[0]: ts_input_name})
    return out_df

def main_exp(data, ts_inputs):
    """Other sample main function for transforming a set of input timeseries to
    produce a set of associated output time series.

    Args:
        data (dict): calculation-specfic parameters for Cognite Function
        ts_inputs (pd.DataFrame): input time series to transform, one column per time series

    Returns:
        (pd.DataFrame): transformed time series, one column per time series
    """
    ts_out = exp(ts_inputs.iloc[:,0])
    ts_out = ts_out / 10.0

    return pd.DataFrame(ts_out) # ensure output given as dataframe

def filter_ts(ts, data):
    """Helper function: performs lowess smoothing

    Args:
        ts (pd.DataFrame): (single) input time series
        data (dict): calculation-specific parameters for Cognite Function

    Returns:
        pd.DataFrame: smoothed signal
    """
    ts_input_name = ts.columns[0]
    vol_perc = ts[ts_input_name]
    ts["time_sec"] = (ts.index - datetime(1970, 1, 1)).total_seconds()

    if "lowess_frac" in data:
        frac = data["lowess_frac"]
    else:
        frac = 0.01
    if "lowess_delta" in data:
        delta = data["lowess_delta"]
    else:
        delta = 0
    smooth = lowess(vol_perc, ts['time_sec'], is_sorted=True,
                    frac=frac, it=0, delta=delta*len(ts))
    smooth = vol_perc

    df_smooth = pd.DataFrame(smooth, columns=["time_sec", "smooth"])

    ts.reset_index(inplace=True)
    ts = ts.rename(columns={'index': 'time_stamp'})
    ts = pd.merge(ts, df_smooth, on='time_sec')
    ts.set_index('time_stamp', drop=True, append=False,
                 inplace=True, verify_integrity=False)

    return ts, ts_input_name

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