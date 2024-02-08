from typing import Any
import numpy as np
import pandas as pd
import os
import sys

# parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# if parent_path not in sys.path:
#     sys.path.append(parent_path)

class RunTransformations:
    """Perform transformations/calculations on time series
    prepared by an instance of PrepareTimeSeries class.
    """
    def __init__(self, data_dict: dict, ts_input_df: pd.DataFrame):
        """Initialize data dictionary and input time series.

        Args:
            data_dict (dict): data dictionary for Cognite Function
            ts_input_df (pd.DataFrame): input time series provided as columns of a pd.DataFrame
        """
        self.data = data_dict
        self.ts_df = ts_input_df

    def __call__(self, calc_func) -> pd.DataFrame:
        """Transform input time series using provided calculation

        Args:
            calc_func (function): function to run calculation

        Returns:
            (pd.DataFrame): transformed data, one column per time series
        """
        ts_out = calc_func(self.data, self.ts_df)
        print(f"Output: {ts_out.shape[1]} time series, each with {ts_out.shape[0]} datapoints.")

        return ts_out

    def store_output_ts(self, ts_output: pd.DataFrame) -> pd.DataFrame:
        """Store output time series in dataframe with appropriate labels

        Args:
            ts_output (pd.DataFrame): data for output time series (typically result from calling the class)

        Returns:
            pd.DataFrame: output time series
        """
        df_out = ts_output.rename(columns={ts_in: ts_out for ts_in, ts_out in zip(ts_output.columns, self.data["ts_output"].keys())})
        return df_out
