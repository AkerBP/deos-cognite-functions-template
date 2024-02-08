
import os
import sys
import pandas as pd
from datetime import datetime

# Set file to system path to allow relative import from parent folder
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from cognite.client._cognite_client import CogniteClient
from prepare_timeseries import PrepareTimeSeries
from run_transformation import RunTransformations
from transformation import *

def handle(client: CogniteClient, data: dict) -> str:
    """Main entry point for Cognite Functions fetching input time series,
    transforming the signals, and storing the output in new time series.

    Args:
        client (CogniteClient): client used to authenticate cognite session
        data (dict): data input to the handle

    Returns:
        str: jsonified data from input signals spanning backfilling period
    """
    calculation = data["calculation_function"]
    # STEP 1: Load (and backfill) and organize input time series'
    PrepTS = PrepareTimeSeries(data["ts_input_names"], data["ts_output"], client, data)
    PrepTS.data = PrepTS.get_orig_timeseries(eval(calculation))

    ts_in = PrepTS.data["ts_input_data"]
    ts_out = PrepTS.data["ts_output"]
    all_inputs_empty = any([ts_in[name].empty if isinstance(ts_in[name], (pd.Series, pd.DataFrame)) else False for name in ts_in])

    if not all_inputs_empty: # can't run calculations if any time series is empty for defined interval
        df_in = PrepTS.get_ts_df()
        df_in = PrepTS.align_time_series(df_in) # align input time series to cover same time period

        # STEP 2: Run transformations
        transform_timeseries = RunTransformations(PrepTS.data, df_in)
        df_out = transform_timeseries(eval(calculation))

        # STEP 3: Ensure output is correctly formatted dataframe as required by template
        assert_df(df_out, ts_out)

        # STEP 4: Insert transformed signal for new time range (done simultaneously for multiple time series outputs)
        client.time_series.data.insert_dataframe(df_out)

    # Store original signal (for backfilling)
    return df_out.to_json()

def assert_df(df_out, ts_out):
    """Check requirements that needs to be satisfied for
    the output dataframe from the calculation.

    Args:
        df_out (pd.DataFrame): output dataframe of calculation
        ts_out (list): names of output time series
    """
    assert isinstance(df_out, pd.DataFrame), f"Output of calculation must be a Dataframe"
    assert type(df_out.index) == pd.DatetimeIndex, f"Dataframe index must be of type DatetimeIndex, not {type(df_out.index)}."
    assert (list(df_out.columns) == list(ts_out.keys())), f"Dataframe columns for calculated time series, {list(df_out.columns)}, not equal to output names, {list(ts_out.keys())}, specified in data dictionary"
