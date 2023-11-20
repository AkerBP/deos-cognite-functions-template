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
    data = get_orig_timeseries(client, data, run_transformation)

    # STEP 2: Run transformations
    df_new = run_transformation(data)

    # STEP 3: Insert transformed signal(s) for new time range (done simultaneously for multiple time series outputs)
    client.time_series.data.insert_dataframe(df_new)

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

    ts_input_names = ["VAL_17-FI-9101-286:VALUE", "VAL_17-PI-95709-258:VALUE", "VAL_11-PT-92363B:X.Value", "VAL_11-XT-95067B:Z.X.Value"]
    ts_output_names = ["VAL_11-LT-95007B:X.CDF.D.AVG.LeakValue"]
    tank_volume = 1400
    derivative_value_excl = 0.002
    # start_date = datetime(2023, 3, 21, 1, 0, 0)
    function_name = "ideal-power-consumption"

    data_dict = {'granularity':60,
                'ts_input':{name:{} for name in ts_input_names}, # empty dictionary for each time series input
                'ts_output':{name:{} for name in ts_output_names},
                'derivative_value_excl':derivative_value_excl, 'tank_volume':tank_volume,
                'cdf_env':"dev", 'dataset_id': 1832663593546318,
                'backfill': False, 'backfill_days': 10,
                'function_name': function_name,
                'lowess_frac': 0.001, 'lowess_delta': 0.01} # NB: change dataset id when going to dev/test/prod!

    # client.time_series.delete(external_id="VAL_11-LT-95007B:X.CDF.D.AVG.LeakValue")
    new_df = handle(client, data_dict)
