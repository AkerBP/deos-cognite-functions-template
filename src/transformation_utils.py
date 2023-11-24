from typing import Any
import numpy as np
import pandas as pd
import ast

class RunTransformations:
    def __init__(self, data_dict, ts_input_df):
        self.data = data_dict
        self.ts_df = ts_input_df

    def __call__(self, calc_func):
        """Transform input time series using provided calculation

        Args:
            calc_func (function): function to run calculation

        Returns:
            (list): list of datapoints for resulting calculation, one dataframe per output time series
        """
        ts_out = calc_func(self.data["calc_params"], *self.ts_df)

        if not isinstance(ts_out, list): # convert to list of one output time series
            ts_out = [ts_out]

        return ts_out

    def store_output_ts(self, ts_output):
        """Store output time series in separate dataframes

        Args:
            ts_output (list): datapoints for output time series (typically result from calling the class)
            data (dict): input parameters for Cognite Function

        Returns:
            pd.DataFrame: dataframes of output time series
        """
        data = self.data
        out_df = pd.DataFrame()

        for ts_out_name, ts_out_val in zip(data["ts_output"].keys(), ts_output):
            added_df = pd.DataFrame({ts_out_name: ts_out_val})
            out_df = pd.concat([out_df, added_df], axis=1) # concatenate on dates

        out_df.index = pd.to_datetime(out_df.index)

        return out_df
