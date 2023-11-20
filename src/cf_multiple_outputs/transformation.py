import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from transformation_utils import get_input_ts, align_time_series, store_output_ts

def run_transformation(data, *ts_data):
    # --- calculation function to be defined below ---
    ts_output = calculation(data, *ts_data) # NB: order of time series in ts_data must correspond to order of input arguments (ts_0, ts_1, ...)
    # ----------------------------------------------------------------------
    out_df = store_output_ts(ts_output, data)

    return out_df

def calculation(data, *ts_data):
    ts_out = []
    for ts in ts_data:
        ts_out.append(ts.rolling(window=int(len(ts)/10)).mean())
    return ts_out