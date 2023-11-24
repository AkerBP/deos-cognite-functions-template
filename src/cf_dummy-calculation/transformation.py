import pandas as pd
import numpy as np
import os
import sys

def calc_A(data, ts_0):
    ts_out = ts_0.rolling(window=int(len(ts_0)/10)).mean()
    return [ts_out]

def calc_B(data, ts_0, ts_1):
    window_size = np.ceil(len(ts_0)/100)
    ts_out = ts_0.rolling(window=window_size).mean() \
                + ts_1.rolling(window=window_size).median()
    return [ts_out]