import pandas as pd
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import numpy as np

def run_transformation(df, data):
    ts_input_name = data["ts_input_name"]
    ts_output_name = data["ts_output_name"]
    agg_df = df.resample("30T").agg({ts_input_name: "mean"})

    # external ID is column name
    # new_df = pd.DataFrame({data["ts_output_name"]: agg_df})
    new_df = agg_df.rename(columns={ts_input_name: ts_output_name})
    new_df.index = pd.to_datetime(new_df.index)

    return new_df

