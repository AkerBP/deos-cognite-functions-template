import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import numpy as np

def run_transformation(data):
    """Run calculations on input time series to produce output time series of wasted energy.

    Args:
        data (dict): dictionary of dictionaries, each sub-dictionary containing data associated with a particular time series.


    Returns:
        pd.DataFrame: output time series of wasted energy
    """
    ts_input = {}
    for ts in data.keys():
        if "df_orig_today" in data[ts]:
            ts_input[ts] = data[ts]["df_orig_today"]

    # ts_input["ts_ipc"] is time series outputted from Cognite Function "ideal_power_consumption"
    ts_we = calculate_wasted_energy(ts_input["ts_4"], ts_input["ts_5"], ts_input["ts_ipc"])

    out_df = pd.DataFrame({data["ts_output_name"]: ts_we})
    out_df.index = pd.to_datetime(out_df.index)

    return out_df


def calculate_wasted_energy(ts_4, ts_5, ts_ipc):
    ts_4 = np.mean(ts_4, dtype=int)
    if ts_4 > 0:
        wasted_energy = ts_5 - ts_ipc
    else:
        wasted_energy = np.zeros(len(ts_ipc))

    return wasted_energy

