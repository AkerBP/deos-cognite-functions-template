import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import numpy as np

def run_transformation(data):
    ts_input = {}
    for ts in data.keys():
        if "df_orig_today" in data[ts]:
            ts_input[ts] = data[ts]["df_orig_today"]

    ts_ipc = calculate_ideal_power_consumption(ts_input["ts_1"], ts_input["ts_2"],
                                               ts_input["ts_3"], ts_input["ts_4"])

    out_df = pd.DataFrame({data["ts_output_name"]: ts_ipc})
    out_df.index = pd.to_datetime(out_df.index)

    return out_df


def calculate_ideal_power_consumption(ts_1, ts_2, ts_3, ts_4):
    ts_4 = np.mean(ts_4, dtype=int)
    if ts_4 > 0:
        ideal_power_consumption = ((ts_1 / 3600) * ((ts_2 * 1.2) - ts_3) * 10**5 / 1000) / (0.9 * 0.93 * 0.775)
    else:
        ideal_power_consumption = np.zeros(len(ts_1))

    return ideal_power_consumption

