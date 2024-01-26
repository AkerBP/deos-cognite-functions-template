
import os
import sys
from datetime import datetime

# Set file to system path to allow relative import from parent folder
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from cognite.client._cognite_client import CogniteClient
from handler_utils import PrepareTimeSeries
from transformation_utils import RunTransformations
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
        # Ensure output is correctly formatted dataframe as required by template
        assert isinstance(df_out, pd.DataFrame), f"Output of calculation must be a Dataframe"
        assert type(df_out.index) == pd.DatetimeIndex, f"Dataframe index must be of type DatetimeIndex, not {type(df_out.index)}."
        assert (list(df_out.columns) == list(ts_in.keys()))                 | (list(df_out.columns) == list(ts_out.keys())), f"df_out {list(df_out.columns)} not equal to ts_in {list(ts_in.keys())}"

        # STEP 3: Structure and insert transformed signal for new time range (done simultaneously for multiple time series outputs)
        df_out = transform_timeseries.store_output_ts(df_out)
        client.time_series.data.insert_dataframe(df_out)

    # Store original signal (for backfilling)
    return df_out.to_json() # PrepTS.data["ts_input_backfill"]

if __name__ == "__main__":
    from initialize import initialize_client
    from dotenv import load_dotenv
    from deploy_cognite_functions import deploy_cognite_functions
    import os

    cdf_env = "dev"

    client = initialize_client(cdf_env, path_to_env="../../authentication-ids.env")
    load_dotenv("../../handler-data.env")

    ts_input_names = ["VAL_11-LT-95107A:X.Value"]
    ts_output = {"names": ["hourly_avg_drainage_test"],
                "description": [None], #["Daily average drainage from pump"]
                "unit": [None]} #["m3/min"]

    function_name = "hourly-avg-drainage"
    calculation_function = "aggregate"
    schedule_name = "ipc"#ts_input_names[0]

    sampling_rate = "1m" #
    cron_interval_min = str(59) #
    assert int(cron_interval_min) < 60 and int(cron_interval_min) >= 1
    backfill_period = 3
    backfill_hour = 23 # 23
    backfill_min_start = 30
    backfill_min_start = min(59, backfill_min_start)

    optional = {
        "historic_start_time": {
            "year": 2023,
            "month": 1,
            "day": 1
        },
        "aggregate": {
            "period": "day",
            "type": "mean"
        }
    }

    calc_params = {
        "tank_volume": 240,
        "derivative_value_excl": 0.002,
        "lowess_frac": 0.001,
        "lowess_delta": 0.01,
    }

    data_dict = {'ts_input_names':ts_input_names,
            'ts_output':ts_output,
            'function_name': f"cf_{function_name}",
            'schedule_name': schedule_name,
            'calculation_function': f"main_{calculation_function}",
            'granularity': sampling_rate,
            'dataset_id': 1832663593546318, # Center of Excellence - Analytics dataset
            'cron_interval_min': cron_interval_min,
            'testing': False,
            'backfill_period': backfill_period, # days by default (if not doing aggregates)
            'backfill_hour': backfill_hour, # 23: backfilling to be scheduled at last hour of day as default
            'backfill_min_start': backfill_min_start, 'backfill_min_end': min(59.9, backfill_min_start + int(cron_interval_min)),
            'optional': optional,
            'calc_params': calc_params
            }

    # deploy_cognite_functions(data_dict=data_dict, client=client, single_call=False, scheduled_call=False)
    handle(client, data_dict)