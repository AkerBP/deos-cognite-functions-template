
import os
import sys
from datetime import datetime

# Set file to system path to allow relative import from parent folder
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from cognite.client._cognite_client import CogniteClient
from prepare_timeseries import PrepareTimeSeries
from transform_timeseries import RunTransformations
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
    PrepTS = PrepareTimeSeries(data["ts_input_names"], data["ts_output_names"], client, data)
    PrepTS.data = PrepTS.get_orig_timeseries(eval(calculation))

    ts_in = PrepTS.data["ts_input_today"]
    all_inputs_empty = any([ts_in[name].empty if isinstance(ts_in[name], (pd.Series, pd.DataFrame)) else False for name in ts_in])

    if not all_inputs_empty: # can't run calculations if any time series is empty for defined interval
        df_in = PrepTS.get_ts_df()
        df_in = PrepTS.align_time_series(df_in) # align input time series to cover same time period

        # STEP 2: Run transformations
        transform_timeseries = RunTransformations(PrepTS.data, df_in)
        df_out = transform_timeseries(eval(calculation))

        # STEP 2.5: Assemble aggregations (if relevant) - think it will work for schedules that overlap aggregation periods
        df_out_prev_exists = not client.time_series.data.retrieve_dataframe(external_id=list(PrepTS.data["ts_output"].keys())).empty

        if "period" in PrepTS.data["aggregate"] and "type" in PrepTS.data["aggregate"] and df_out_prev_exists:

            end_time_previous = PrepTS.data["start_time"]
            start_time_previous = PrepTS.get_aggregated_start_time()

            df_out_prev = client.time_series.data.retrieve_dataframe(external_id=list(PrepTS.data["ts_output"].keys()),
                                                                    start=start_time_previous,
                                                                    end=end_time_previous)
            df_out_prev.index = pd.to_datetime(df_out_prev.index)
            df_out.index = pd.to_datetime(df_out.index)

            df_out = df_out.join(df_out_prev)
            df_out = df_out.apply(lambda row: getattr(row, PrepTS.data["aggregate"]["type"])(), axis=1)

            df_out = pd.DataFrame(df_out, columns=PrepTS.data["ts_input_names"])

        # STEP 3: Structure and insert transformed signal for new time range (done simultaneously for multiple time series outputs)
        df_out = transform_timeseries.store_output_ts(df_out)
        client.time_series.data.insert_dataframe(df_out)

    # Store original signal (for backfilling)
    return PrepTS.data["ts_input_backfill"]
