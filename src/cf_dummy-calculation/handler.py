import os
import sys

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from handler_utils import PrepareTimeSeries #get_orig_timeseries
from transformation_utils import RunTransformations
from transformation import *

def handle(client, data):
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
    PrepTS = PrepareTimeSeries(data["ts_input"], data["ts_output"], client, data)
    data = PrepTS.get_orig_timeseries(eval(calculation))
    ts_df = PrepTS.get_ts_df()
    ts_df = PrepTS.align_time_series(ts_df) # align input time series to cover same time period

    # STEP 2: Run transformations
    transform_timeseries = RunTransformations(data, ts_df)
    # df_new = run_transformation(data, ts_df)
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
    import re

    cdf_env = "dev"
    if cdf_env not in ["dev", "test", "prod"]:
        token = True
    else:
        token = False

    client = initialize_client(cdf_env, cache_token=token, path_to_env="../../authentication-ids.env")
    load_dotenv("../../handler-data.env")

    ts_input_names = ["VAL_17-FI-9101-286:VALUE"]
    ts_output_names = ["VAL_17-FI-9101-286:VALUE.DUMMY"]

    tank_volume = 1400
    derivative_value_excl = 0.002
    # start_date = datetime(2023, 3, 21, 1, 0, 0)
    func_name = re.search("[^/]+$", os.getcwd().replace("\\","/"))[0]
    calc_func = "calc_A"

    data_dict = {'ts_input': {name:{} for name in ts_input_names},
                 'ts_output': {name:{} for name in ts_output_names},
                 'granularity': 60,
                 'derivative_value_excl': derivative_value_excl, 'tank_volume': tank_volume,
                 # NB: change dataset id when going to dev/test/prod!
                 'cdf_env': cdf_env, 'dataset_id': int(os.getenv("DATASET_ID")),
                 'backfill': False, 'backfill_days': 10,
                 'function_name': func_name,
                 'calculation_function': calc_func,
                 'lowess_frac': 0.001, 'lowess_delta': 0.01}

    # client.time_series.delete(external_id=str(os.getenv("TS_OUTPUT_NAME")))
    new_df = handle(client, data_dict)