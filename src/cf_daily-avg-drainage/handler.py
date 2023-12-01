import os
import sys

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from cognite.client._cognite_client import CogniteClient
from handler_utils import PrepareTimeSeries #get_orig_timeseries
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
    PrepTS = PrepareTimeSeries(data["ts_input_names"], data["ts_output_names"], client, data)
    data = PrepTS.get_orig_timeseries(eval(calculation))
    ts_df = PrepTS.get_ts_df()
    ts_df = PrepTS.align_time_series(ts_df) # align input time series to cover same time period

    # STEP 2: Run transformations
    transform_timeseries = RunTransformations(data, ts_df)
    ts_out = transform_timeseries(eval(calculation))

    # STEP 3: Structure and insert transformed signal for new time range (done simultaneously for multiple time series outputs)
    df_out = transform_timeseries.store_output_ts(ts_out)
    client.time_series.data.insert_dataframe(df_out)

    # Store original signal (for backfilling)
    return data["ts_input_backfill"]


if __name__ == '__main__':
    from initialize import initialize_client
    from dotenv import load_dotenv
    import os

    cdf_env = "dev"
    if cdf_env not in ["dev", "test", "prod"]:
        token = True
    else:
        token = False

    client = initialize_client(cdf_env, cache_token=token, path_to_env="../../authentication-ids.env")
    load_dotenv("../../handler-data.env")

    # ts_input_names = ["VAL_17-FI-9101-286:VALUE", "VAL_17-PI-95709-258:VALUE", "VAL_11-PT-92363B:X.Value", "VAL_11-XT-95067B:Z.X.Value"]
    ts_input_names = ["VAL_11-LT-95034A:X.Value"]
    # ts_output_names = ["VAL_17-FI-9101-286:CDF.IdealPowerConsumption"]
    ts_output_names = ["VAL_11-LT-95034A:X.CDF.D.AVG.LeakValue"]
    function_name = "daily-avg-drainage" #
    calculation_function = "daily_avg_drainage" #

    sampling_rate = 60 #
    cron_interval_min = str(15) #
    assert int(cron_interval_min) < 60 and int(cron_interval_min) >= 1
    backfill_days = 3

    cdf_env = "dev"

    tank_volume = 1400
    derivative_value_excl = 0.002
    lowess_frac = 0.001
    lowess_delta = 0.01

    data_dict = {'ts_input_names':ts_input_names, # empty dictionary for each time series input
            'ts_output_names':ts_output_names,
            'function_name': f"cf_{function_name}",
            'calculation_function': f"calc_{calculation_function}",
            'granularity': sampling_rate,
            'dataset_id': 1832663593546318, # Center of Excellence - Analytics dataset
            'backfill_days': backfill_days,
            'backfill_hour': 10, # backfilling to be scheduled at last hour of day as default
            'backfill_min_start': 10, 'backfill_min_end': 10 + int(cron_interval_min),
            'calc_params': {
                'derivative_value_excl':derivative_value_excl, 'tank_volume':tank_volume,
                'lowess_frac': lowess_frac, 'lowess_delta': lowess_delta
            }}

    # client.time_series.delete(external_id=str(os.getenv("TS_OUTPUT_NAME")))
    new_df = handle(client, data_dict)
