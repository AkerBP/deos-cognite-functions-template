import os
import sys
import pandas as pd
from datetime import datetime

# Set file to system path to allow relative import from parent folder
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_path not in sys.path:
    sys.path.append(parent_path)

from cognite.client._cognite_client import CogniteClient
from time_series_calculation.prepare_timeseries import PrepareTimeSeries
from time_series_calculation.transform_timeseries import TransformTimeseries
from transformation import transformation

def handle(client: CogniteClient, data: dict) -> str:
    """Main entry point for Cognite Functions fetching input time series,
    transforming the signals, and storing the output in new time series.

    Args:
        client (CogniteClient): client used to authenticate cognite session
        data (dict): data input to the handle

    Returns:
        str: jsonified data from input signals spanning backfilling period
    """
    # STEP 1: Load (and backfill) and organize input time series'
    PrepTS = PrepareTimeSeries(data["input_ts_names"], data["output_ts"], client, data)
    PrepTS.data = PrepTS.get_orig_timeseries(transformation)

    ts_in = PrepTS.data["input_ts_names"]
    ts_out = PrepTS.data["output_ts"]
    all_inputs_empty = any([ts_in[name].empty if isinstance(ts_in[name], (pd.Series, pd.DataFrame)) else False for name in ts_in])

    if not all_inputs_empty: # can't run calculations if any time series is empty for defined interval
        df_in = PrepTS.get_ts_df()
        df_in = PrepTS.align_time_series(df_in) # align input time series to cover same time period

        # STEP 2: Run transformations
        transform_timeseries = TransformTimeseries(PrepTS.data, df_in)
        df_out = transform_timeseries(transformation)

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


if __name__ == "__main__":
    import pandas as pd
    from datetime import datetime
    from dotenv import load_dotenv
    from time_series_calculation.initialize_cdf_client import initialize_cdf_client

    cdf_env = "dev"
    client = initialize_cdf_client(cdf_env, path_to_env="../../../authentication-ids.env")

    ts_input_names = ["VAL_11-XT-95067B:Z.X.Value"]
    output_ts = {"names": ["CoEA_refactor_test"],
                "description": ["Test if deployment works for new refactoring"], #["Daily average drainage from pump"]
                "unit": [None]} #["m3/min"]
    dataset_id = 1832663593546318

    function_name = "CoEA_test"
    schedule_name = "test_schedule_name"#ts_input_names[0]

    data_dict = {
            "cdf_env": cdf_env,
            "client": client,
            "description": "Test of refactordeployment",
            "input_ts_names": ts_input_names,
            "input_ts_sampling_rate": "1m",
            "output_ts": output_ts,
            "output_ts_agg_method": None,
            "output_ts_agg_freq": None,
            "dataset_id": dataset_id,
            "name_of_function": function_name,
            "name_of_schedule": schedule_name,
            "cron_interval_in_minutes": str(15),
            "backfill_period": 7,
            "backfill_hour": 12,
            "backfill_min_start": 0,
            "historic_start_time": "2023-1-1 00:00",
            "deployment_single_call": False,
            "deployment_scheduled_call": True,
        }

    handle(client, data_dict)