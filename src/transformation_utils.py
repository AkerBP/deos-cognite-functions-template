from typing import Any
import numpy as np
import pandas as pd
import ast

class RunTransformations:
    """Perform transformations/calculations on time series
    prepared by an instance of PrepareTimeSeries class.
    """
    def __init__(self, data_dict: dict, ts_input_df: list):
        """Initialize data dictionary and input time series.

        Args:
            data_dict (dict): data dictionary for Cognite Function
            ts_input_df (list): input time series provided as pd.DataFrames
        """
        self.data = data_dict
        self.ts_df = ts_input_df

    def __call__(self, calc_func) -> list:
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

    def store_output_ts(self, ts_output: list) -> pd.DataFrame:
        """Store output time series in separate dataframes

        Args:
            ts_output (list): datapoints for output time series (typically result from calling the class)

        Returns:
            pd.DataFrame: dataframes of output time series
        """
        data = self.data
        out_df = pd.DataFrame()

        for ts_out_name, ts_out_val in zip(data["ts_output"].keys(), ts_output):
            added_df = pd.DataFrame({ts_out_name: ts_out_val}, index=ts_out_val.index)
            out_df = pd.concat([out_df, added_df], axis=1) # concatenate on dates

        out_df.index = pd.to_datetime(out_df.index)

        return out_df
