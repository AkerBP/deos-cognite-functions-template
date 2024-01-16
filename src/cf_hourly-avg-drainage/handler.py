
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
    all_inputs_empty = any([ts_in[name].empty if isinstance(ts_in[name], (pd.Series, pd.DataFrame)) else False for name in ts_in])

    if not all_inputs_empty: # can't run calculations if any time series is empty for defined interval
        df_in = PrepTS.get_ts_df()
        df_in = PrepTS.align_time_series(df_in) # align input time series to cover same time period

        # STEP 2: Run transformations
        transform_timeseries = RunTransformations(PrepTS.data, df_in)
        df_out = transform_timeseries(eval(calculation))

        # STEP 3: Structure and insert transformed signal for new time range (done simultaneously for multiple time series outputs)
        df_out = transform_timeseries.store_output_ts(df_out)
        client.time_series.data.insert_dataframe(df_out)

    # Store original signal (for backfilling)
    return PrepTS.data["ts_input_backfill"]

if __name__ == "__main__":
    from initialize import initialize_client
    from dotenv import load_dotenv
    from deploy_cognite_functions import deploy_cognite_functions
    import os

    cdf_env = "dev"

    client = initialize_client(cdf_env, path_to_env="../../authentication-ids.env")
    load_dotenv("../../handler-data.env")

    # ts_input_names = ["VAL_17-FI-9101-286:VALUE", "VAL_17-PI-95709-258:VALUE", "VAL_11-PT-92363B:X.Value", "VAL_11-XT-95067B:Z.X.Value"] # Inputs to IdealPowerConsumption function # ["VAL_11-XT-95067B:Z.X.Value", 87.8, "CF_IdealPowerConsumption"] # Inputs to WasterEnergy function
    ts_input_names = ["VAL_11-LT-95107A:X.Value"]
    # ts_output_names = ["CF_IdealPowerConsumption"]
    ts_output = {"names": ["hourly_avg_drainage_description_unit"],
                "description": ["Hourly average drainage from pump"], #["Daily average drainage from pump"]
                "unit": ["m3/min"]} #["m3/min"]

    function_name = "hourly-avg-drainage"
    calculation_function = "aggregate"
    schedule_name = ts_input_names[0]

    aggregate = {}
    aggregate["period"] = "hour"
    aggregate["type"] = "mean"

    sampling_rate = 60 #
    cron_interval_min = str(15) #
    assert int(cron_interval_min) < 60 and int(cron_interval_min) >= 1
    backfill_period = 3
    backfill_hour = 15 # 23
    backfill_min_start = 30

    tank_volume = 240
    derivative_value_excl = 0.002
    lowess_frac = 0.001
    lowess_delta = 0.01

    backfill_min_start = min(59, backfill_min_start)

    data_dict = {'ts_input_names':ts_input_names,
            'ts_output':ts_output,
            'function_name': f"cf_{function_name}",
            'schedule_name': schedule_name,
            'calculation_function': f"main_{calculation_function}",
            'granularity': sampling_rate,
            'dataset_id': 1832663593546318, # Center of Excellence - Analytics dataset
            'cron_interval_min': cron_interval_min,
            'aggregate': aggregate,
            'testing': False,
            'backfill_period': backfill_period, # days by default (if not doing aggregates)
            'backfill_hour': backfill_hour, # 23: backfilling to be scheduled at last hour of day as default
            'backfill_min_start': backfill_min_start, 'backfill_min_end': min(59.9, backfill_min_start + int(cron_interval_min)),
            'calc_params': {
                'derivative_value_excl':derivative_value_excl, 'tank_volume':tank_volume,
                'lowess_frac': lowess_frac, 'lowess_delta': lowess_delta, #'aggregate_period': aggregate["period"]
            }}

    # deploy_cognite_functions(data_dict=data_dict, client=client, single_call=False, scheduled_call=False)
    handle(client, data_dict)