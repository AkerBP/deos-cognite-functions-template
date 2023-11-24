import pytest
import pandas as pd
import numpy as np
import ast
from src.handler_utils import PrepareTimeSeries
from src.initialize import initialize_client
from src.deploy_cognite_functions import deploy_cognite_functions, list_scheduled_calls

@pytest.fixture
def prepare_ts():
    ts_input_names = ["VAL_17-FI-9101-286:VALUE"]
    ts_output_names = ["CF_PrepareTimeSeries_test"]

    data_dict = {"cdf_env":"dev"}

    client = initialize_client(data_dict["cdf_env"], cache_token=False)

    return PrepareTimeSeries(ts_input_names, ts_output_names, client, data_dict)

@pytest.fixture
def deploy_schedule(prepare_ts):
    import time
    from datetime import datetime, timedelta
    ts_input_name = "VAL_17-FI-9101-286:VALUE"

    cdf_env = "dev"
    prepare_ts.data["cdf_env"] = cdf_env
    cron_interval = 1
    prepare_ts.data["cron_interval"] = cron_interval
    prepare_ts.data["function_name"] = "cf_test"
    # First: Make single call to function to perform initial transformation
    deploy_cognite_functions(prepare_ts.data, prepare_ts.client, cron_interval,
                            True, False)

    # Let backfill period be current timestamp - return argument to be compared to
    now = pd.Timestamp.now()
    while now.minute > 45:
        time.sleep(1)
    prepare_ts.data["backfill_hour"] = now.hour
    prepare_ts.data["backfill_min_start"] = now.minute
    prepare_ts.data["backfill_min_end"] = now.minute + 1
    # Then: Run function on schedule
    now = pd.Timestamp.now()
    deployed_time_start = datetime(now.year, now.month, now.day, now.hour, now.minute)

    deploy_cognite_functions(prepare_ts.data, prepare_ts.client, cron_interval,
                            False, True)

    schedule_id, all_calls = list_scheduled_calls(prepare_ts.data, prepare_ts.client)
    while len(all_calls) < 10:
        time.sleep(60)
        schedule_id, all_calls = list_scheduled_calls(prepare_ts.data, prepare_ts.client)

    schedule_data = {"schedule_id":schedule_id, "calls":all_calls, "deploy_start":deployed_time_start}

    get_func = prepare_ts.client.functions.retrieve(external_id=prepare_ts.data["function_name"])
    last_backfill_call = get_func.retrieve_call(id=schedule_id)
    ts_input_name = prepare_ts.data["ts_input"][ts_input_name]
    ts_orig_extid = ts_input_name["ts_orig_extid"]
    output_dict = ast.literal_eval(last_backfill_call.get_response())[
            ts_input_name]

    output_df = pd.DataFrame.from_dict([output_dict]).T
    output_df.index = pd.to_datetime(
        output_df.index.astype(np.int64), unit="ms")
    output_df["Date"] = output_df.index.date  # astype(int)*1e7 for testing
    previous_df = output_df.rename(columns={0: ts_input_name})

    end_date = pd.Timestamp.now()
    start_date = end_date - timedelta(days=prepare_ts.data["backfill_days"])
    ts_orig_all = prepare_ts.client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                    aggregates="average",
                                                    granularity=f"{prepare_ts.data['granularity']}s",
                                                    start=start_date,
                                                    end=end_date,
                                                    limit=-1,
                                                    ).to_pandas()

    ts_orig_all = ts_orig_all.rename(
        columns={ts_orig_extid + "|average": ts_input_name})

    schedule_data["df_ts_previous"] = previous_df
    schedule_data["df_ts_current"] = ts_orig_all

    return schedule_data


def test_get_schedules_and_calls(prepare_ts):
    schedule_id, calls = prepare_ts.get_schedules_and_calls()

    valid_schedule_id = 1234567
    assert schedule_id == valid_schedule_id
    assert not calls.empty
    # Some other tests needed for schedule calls ??

    prepare_ts.data["function_name"] = "cf_test_fail"
    schedule_id, calls = prepare_ts.get_schedules_and_calls()

    assert schedule_id == None
    assert calls.empty

def test_create_timeseries(prepare_ts):
    prepare_ts.ts_outputs = ["CF_PrepareTimeSeries_test_create_timeseries"]
    prepare_ts.update_ts("ts_output")

    true_asset_id = [7210022650246556]
    returned_asset_id = prepare_ts.create_timeseries()
    assert returned_asset_id == true_asset_id

    # Assert that new time series object created
    prepare_ts.data["scheduled_calls"] = pd.DateFrame()
    ts_output_retrieved = prepare_ts.client.time_series.list(name=prepare_ts.ts_outputs[0]).to_pandas()
    assert not ts_output_retrieved.empty

    # Assert no duplicated time series object if already exists
    prepare_ts.data["ts_output"][prepare_ts.ts_outputs[0]]["exists"] = True
    prepare_ts.create_timeseries()
    ts_output_retrieved_new = prepare_ts.client.time_series.list(name=prepare_ts.ts_outputs[0]).to_pandas()
    assert ts_output_retrieved_new == ts_output_retrieved

def test_retrieve_orig_ts(prepare_ts):
    prepare_ts.ts_outputs = ["CF_PrepareTimeSeries_test_create_timeseries"]
    prepare_ts.update_ts("ts_output")
    ts_in = prepare_ts.ts_inputs[0]
    ts_out = prepare_ts.ts_outputs[0]

    prepare_ts.data["ts_output"][prepare_ts.ts_outputs[0]]["exists"] = False
    df = prepare_ts.retrieve_orig_ts(ts_in, ts_out)
    first_date = df.index[0]
    last_date = df.index[-1]
    df = prepare_ts.client.time_series.list(name=ts_in).to_pandas()
    first_date_true = df.index[0]
    last_date_true = df.index[-1]
    assert first_date == first_date_true
    assert last_date == last_date_true

    prepare_ts.data["ts_output"][prepare_ts.ts_outputs[0]]["exists"] = True
    df = prepare_ts.retrieve_orig_ts(ts_in, ts_out)
    first_date = df.index[0]
    last_date = df.index[-1]
    assert first_date.date == first_date_true.date
    assert last_date == last_date_true

def test_backfilling_schedule_id(prepare_ts, deploy_schedule):
    from datetime import timedelta
    # Assert that correct historical scheduled call is extracted at time when doing backfilling
    deployed_time_start = deploy_schedule["deploy_start"]
    all_calls = deploy_schedule["calls"]
    # Get the fifth last scheduled call
    deployed_time_end = deployed_time_start + timedelta(minute=1) # scheduled every 1 min

    mask_start = all_calls["scheduled_time"] >= deployed_time_start
    mask_end = all_calls["scheduled_time"] < deployed_time_end

    # Check that extracted schedule from a given historical time is correct
    initial_scheduled_id = all_calls[mask_start & mask_end]["id"].iloc[0]
    get_func = prepare_ts.client.functions.retrieve(external_id=prepare_ts.data["function_name"])
    get_schedule_id = prepare_ts.client.functions.schedules.list(
                name=prepare_ts.data["function_name"]).to_pandas().id[0]
    get_calls = get_func.list_calls(
                schedule_id=get_schedule_id, limit=-1).to_pandas()
    true_initial_scheduled_id = get_calls["id"].iloc[0] # first call was time of backfilling - this is the one to compare with
    assert initial_scheduled_id == true_initial_scheduled_id





