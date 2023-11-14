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

from handler_utils import get_orig_timeseries
from transformation import run_transformation

def handle(client, data):
    """Calculate drainage rate per timestamp and per day from tank,
    using Lowess filtering on volume percentage data from the tank.
    Large positive derivatives of signal are excluded to ignore
    human interventions (filling) of tank.
    Data of drainage rate helps detecting leakages.

    Args:
        client (CogniteClient): client used to authenticate cognite session
        data (dict): data input to the handle

    Returns:
        pd.DataFrame: dataframe with drainage rate and trend (derivative)
    """
    # STEP 1: Load (and backfill) original time series
    df_orig_today, df_orig_full, data = get_orig_timeseries(
        client, data, run_transformation)

    # STEP 2: Run transformations
    df_new = run_transformation(df_orig_today, data)

    # STEP 3: Insert transformed signal for new time range
    client.time_series.data.insert_dataframe(df_new)

    # Store original signal (for backfilling)
    return df_orig_full


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

    in_name = "VAL_11-LT-95007A:X.Value"
    out_name = "VAL_11-LT-95007A:X.CDF.D.AVG.LeakValue"

    tank_volume = 1400
    derivative_value_excl = 0.002
    # start_date = datetime(2023, 3, 21, 1, 0, 0)
    func_name = "VAL_11-LT-95007A"

    data_dict = {'tot_days': 0, 'tot_minutes': 15,  # convert date to str to make it JSON serializable
                 'ts_input_name': in_name, 'ts_output_name': out_name,
                 'derivative_value_excl': derivative_value_excl, 'tank_volume': tank_volume,
                 # NB: change dataset id when going to dev/test/prod!
                 'cdf_env': cdf_env, 'dataset_id': int(os.getenv("DATASET_ID")),
                 'backfill': False, 'backfill_days': 7,
                 'function_name': func_name,
                 'lowess_frac': 0.001, 'lowess_delta': 0.01}

    # client.time_series.delete(external_id=str(os.getenv("TS_OUTPUT_NAME")))
    new_df = handle(client, data_dict)
