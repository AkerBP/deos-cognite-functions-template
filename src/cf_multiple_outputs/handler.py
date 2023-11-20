import os
import sys
from cognite.client.data_classes import TimeSeries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytz
import time
from datetime import datetime, timedelta

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from handler_utils import get_orig_timeseries, get_ts_df, align_time_series
from transformation import run_transformation

def handle(client, data_dict):
    """Calculate drainage rate per timestamp and per day from tank,
    using Lowess filtering on volume percentage data from the tank.
    Large positive derivatives of signal are excluded to ignore
    human interventions (filling) of tank.
    Data of drainage rate helps detecting leakages.

    Args:
        client (CogniteClient): client used to authenticate cognite session
        data_dict (dict): data input to the handle

    Returns:
        pd.DataFrame: dataframe with drainage rate and trend (derivative)
    """
    # STEP 1: Load (and backfill) and organize input time series'
    data_dict = get_orig_timeseries(client, data_dict, run_transformation)
    ts_df = get_ts_df(data_dict)
    ts_df = align_time_series(ts_df, data_dict) # align input time series to cover same time period

    # STEP 2: Run transformations
    df_new = run_transformation(data_dict, ts_df)
    # STEP 3: Insert transformed signal(s) for new time range (done simultaneously for multiple time series outputs)
    client.time_series.data.insert_dataframe(df_new)

    # Store original signal (for backfilling)
    return data_dict["ts_input_backfill"]


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

    ts_input_names = ["VAL_17-FI-9101-286:VALUE", "VAL_17-PI-95709-258:VALUE", "VAL_11-PT-92363B:X.Value"]
    ts_output_names = ["VAL_17-FI-9101-286:VALUE.MULTIPLE", "VAL_17-PI-95709-258:VALUE.MULTIPLE", "VAL_11-PT-92363B:X.Value.MULTIPLE"]

    tank_volume = 1400
    derivative_value_excl = 0.002
    # start_date = datetime(2023, 3, 21, 1, 0, 0)
    func_name = re.search("[^/]+$", os.getcwd().replace("\\","/"))[0]

    data_dict = {'ts_input': {name:{} for name in ts_input_names},
                 'ts_output': {name:{} for name in ts_output_names},
                 'granularity': 60,
                 'derivative_value_excl': derivative_value_excl, 'tank_volume': tank_volume,
                 # NB: change dataset id when going to dev/test/prod!
                 'cdf_env': cdf_env, 'dataset_id': int(os.getenv("DATASET_ID")),
                 'backfill': False, 'backfill_days': 7,
                 'function_name': func_name,
                 'lowess_frac': 0.001, 'lowess_delta': 0.01}

    # client.time_series.delete(external_id=str(os.getenv("TS_OUTPUT_NAME")))
    new_df = handle(client, data_dict)
