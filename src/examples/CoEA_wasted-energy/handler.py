
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
        assert_df(df_out, ts_in, ts_out)

        # STEP 4: Structure and insert transformed signal for new time range (done simultaneously for multiple time series outputs)
        df_out = transform_timeseries.store_output_ts(df_out)
        client.time_series.data.insert_dataframe(df_out)

    # Store original signal (for backfilling)
    return df_out.to_json()

def assert_df(df_out, ts_in, ts_out):
    """Check requirements that needs to be satisfied for
    the output dataframe from the calculation.

    Args:
        df_out (pd.DataFrame): output dataframe of calculation
        ts_in (list): names of input time series
        ts_out (list): names of output time series
    """
    assert isinstance(df_out, pd.DataFrame), f"Output of calculation must be a Dataframe"
    assert type(df_out.index) == pd.DatetimeIndex, f"Dataframe index must be of type DatetimeIndex, not {type(df_out.index)}."
    if len(ts_in.keys()) > len(ts_out.keys()): # If one time series calculated from multiple inputs
        assert (list(df_out.columns) == list(ts_out.keys())), f"Dataframe columns for calculated time series, {list(df_out.columns)}, not equal to output names, {list(ts_out.keys())}, specified in data dictionary"
    else: # If each time series input corresponds to one time series output
        assert (list(df_out.columns) == list(ts_in.keys())), f"Dataframe columns for calculated time series, {list(df_out.columns)}, not equal to input names, {list(ts_in.keys())}, specified in data dictionary"

if __name__ == "__main__":
    import pandas as pd
    from datetime import datetime
    from cognite.client.data_classes import functions
    from cognite.client.data_classes.functions import FunctionSchedulesList
    from cognite.client.data_classes.functions import FunctionSchedule
    from dotenv import load_dotenv
    from initialize import initialize_client
    from deploy_cognite_functions import deploy_cognite_functions
    from generate_cf import generate_cf

    cdf_env = "dev"
    client = initialize_client(cdf_env, path_to_env="../../authentication-ids.env")
    load_dotenv("../../handler-data.env")

    ts_input_names = ["VAL_11-XT-95067B:Z.X.Value", 87.8, "CoEA_IdealPowerConsumption"] # Inputs to IdealPowerConsumption function # ["VAL_11-XT-95067B:Z.X.Value", 87.8, "CF_IdealPowerConsumption"] # Inputs to WastedEnergy function
    # ts_input_names = ["VAL_11-LT-95107A:X.Value"]
    ts_output = {"names": ["CoEA_WastedEnergy"],
                "description": ["Wasted energy from equipment, calculated from ideal power consumption."], #["Daily average drainage from pump"]
                "unit": ["J/s"]} #["m3/min"]
    dataset_id = 1832663593546318

    function_name = "wasted-energy"
    calculation_function = "wasted_energy"
    schedule_name = "we"#ts_input_names[0]

    sampling_rate = "1m"
    cron_interval_min = str(15) #
    assert int(cron_interval_min) < 60 and int(cron_interval_min) >= 1

    backfill_period = 20
    backfill_hour = 14 # 23
    backfill_min_start = 0
    backfill_min_start = min(59, backfill_min_start)

    optional = {
        "historic_start_time": {
            "year": 2022,
            "month": 1,
            "day": 1
        },
        # "aggregate": {
        #     "period": "day",
        #     "type": "mean"
        # }
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

    handle(client, data_dict)