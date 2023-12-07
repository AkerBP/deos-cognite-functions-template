import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import ast
from cognite.client.testing import monkeypatch_cognite_client
from cognite.client.data_classes import TimeSeries

from src.handler_utils import PrepareTimeSeries
from src.initialize import initialize_client
from src.deploy_cognite_functions import deploy_cognite_functions, list_scheduled_calls

CREATED_TIMESERIES = [
    TimeSeries(
        external_id="test_CF",
        name="test_CF",
        data_set_id=1832663593546318,
        asset_id="17-FI-95710-286"
    )
]

@pytest.fixture
def cognite_client_mock():
    with monkeypatch_cognite_client() as client:
        client.time_series.create.return_value = CREATED_TIMESERIES

@pytest.fixture
def prepare_ts():
    ts_input_names = ["VAL_17-FI-9101-286:VALUE"]
    ts_output_names = ["test_CF"]

    data_dict = {"cdf_env":"dev",
                 "backfill_days": 1}

    client = initialize_client(data_dict["cdf_env"], cache_token=False)

    return PrepareTimeSeries(ts_input_names, ts_output_names, client, data_dict)

@pytest.fixture
def deploy_schedule(prepare_ts):
    """Fixture for deploying a test CF schedule

    Args:
        prepare_ts (pytest.fixture): PrepareTimeSeries object

    Returns:
        schedule_data (dict): parameters for deployed schedule
    """
    import time
    from datetime import datetime, timedelta
    ts_input_name = "VAL_17-FI-9101-286:VALUE.COPY"
    ts_output_name = "test_CF"
    prepare_ts.data["ts_output"] = ts_input_name # this is NOT output - it is a copy of the input, but ts_output is demanded by create_timeseries() function
    prepare_ts.create_timeseries()
    prepare_ts.data["ts_output"] = ts_output_name

    # Insert datapoints into copied input time series
    input_data = prepare_ts.client.time_series.data.retrieve(external_id="VAL_17-FI-9101-286:VALUE",
                                                             start=pd.to_datetime(prepare_ts.data["start_time"]),
                                                             end=pd.to_datetime(prepare_ts.data["end_time"]),
                                                             limit=-1).to_pandas()

    input_data = input_data.rename(columns={input_data.columns[0]: ts_input_name})
    prepare_ts.client.time_series.data.insert_dataframe(input_data)

    ts_input_name = prepare_ts.data["ts_input"][ts_input_name]
    ts_orig_extid = ts_input_name["ts_orig_extid"]

    cdf_env = "dev"
    prepare_ts.data["cdf_env"] = cdf_env
    cron_interval = 1
    prepare_ts.data["cron_interval"] = cron_interval
    prepare_ts.data["function_name"] = "cf_test"
    prepare_ts.data["calculation_function"] = "calc_test"
    # First: Make single call to function to perform initial transformation
    deploy_cognite_functions(prepare_ts.data, prepare_ts.client, cron_interval,
                            True, False)

    # Let backfill period be current timestamp - return argument to be compared to
    now = pd.Timestamp.now()
    while now.minute >= 59:
        time.sleep(1)
        now = pd.Timestamp.now()
    prepare_ts.data["backfill_hour"] = now.hour
    prepare_ts.data["backfill_min_start"] = now.minute
    prepare_ts.data["backfill_min_end"] = now.minute + 1
    # Then: Run function on schedule
    deployed_time_start = datetime(now.year, now.month, now.day, now.hour, now.minute)

    deploy_cognite_functions(prepare_ts.data, prepare_ts.client, cron_interval,
                            False, True)

    schedule_id, all_calls = list_scheduled_calls(prepare_ts.data, prepare_ts.client)
    while len(all_calls) < 10: # wait until we have total of 10 calls from schedule
        time.sleep(60)
        schedule_id, all_calls = list_scheduled_calls(prepare_ts.data, prepare_ts.client)

    schedule_data = {"schedule_id":schedule_id, "calls":all_calls, "deploy_start":deployed_time_start}

    get_func = prepare_ts.client.functions.retrieve(external_id=prepare_ts.data["function_name"])
    last_backfill_id = all_calls.id[0]#get_func.retrieve_call(id=schedule_id)
    last_backfill_call = get_func.retrieve_call(id=last_backfill_id)
    output_dict = ast.literal_eval(last_backfill_call.get_response())[
            ts_input_name]

    output_df = pd.DataFrame.from_dict([output_dict]).T
    output_df.index = pd.to_datetime(
        output_df.index.astype(np.int64), unit="ms")
    output_df["Date"] = output_df.index.date  # astype(int)*1e7 for testing
    previous_df = output_df.rename(columns={0: ts_input_name})

    end_date = pd.Timestamp.now()
    start_date = end_date - timedelta(days=prepare_ts.data["backfill_days"])
    schedule_data["start_time"] = start_date
    schedule_data["end_time"] = end_date
    ts_orig_all = prepare_ts.client.time_series.data.retrieve(external_id=ts_orig_extid,
                                                    aggregates="average",
                                                    granularity=f"{prepare_ts.data['granularity']}s",
                                                    start=start_date,
                                                    end=end_date,
                                                    limit=-1,
                                                    ).to_pandas()

    ts_orig_all = ts_orig_all.rename(
        columns={ts_orig_extid + "|average": ts_input_name})

    ts_orig_dates = pd.DataFrame(
            {"Date": pd.to_datetime(ts_orig_all.index.date),
            ts_input_name: ts_orig_all[ts_input_name]},
            index=pd.to_datetime(ts_orig_all.index))

    schedule_data["df_ts_previous"] = previous_df # dataframe of previous time series signal
    schedule_data["df_ts_current"] = ts_orig_dates # dataframe of current time series signal

    return schedule_data

@pytest.fixture
def calculation(data, ts):
    return (ts - 10).squeeze()

def test_get_schedules_and_calls(prepare_ts):
    # IS THIS TEST POSSIBLE ???
    schedule_id, calls = prepare_ts.get_schedules_and_calls()

    valid_schedule_id = 1234567
    assert schedule_id == valid_schedule_id
    assert not calls.empty
    # Some other tests needed for schedule calls ??

    prepare_ts.data["function_name"] = "cf_test_fail"
    schedule_id, calls = prepare_ts.get_schedules_and_calls()

    assert schedule_id == None
    assert calls.empty

def test_create_timeseries(cognite_client_mock, prepare_ts):
    """Test that new Time Series object is created upon call.

    Args:
        cognite_client_mock(cognite.client.testing.monkeypatch_cognite_client): mock used for CogniteClient
        prepare_ts (_type_): _description_
    """
    prepare_ts.ts_outputs = ["test_CF_create_timeseries"]
    prepare_ts.update_ts("ts_output")

    true_asset_id = [7210022650246556]
    returned_asset_id = prepare_ts.create_timeseries()
    # Assert correct cognite object has been called once
    assert cognite_client_mock.time_series.create.call_count == 1
    # Assert cognite object is called with correct arguments
    assert [ts.dump() for ts in cognite_client_mock.time_series.create.call_args[0]
            ] == [ts.dump() for ts in CREATED_TIMESERIES]
    assert returned_asset_id == true_asset_id

    # Assert that new time series object created
    prepare_ts.data["scheduled_calls"] = pd.DateFrame()
    ts_output_list = prepare_ts.client.time_series.list(name=prepare_ts.ts_outputs[0]).to_pandas()
    assert not ts_output_list.empty
    assert ts_output_list["external_id"] == prepare_ts.ts_outputs[0]

    prepare_ts.data["ts_output"][prepare_ts.ts_outputs[0]]["exists"] = True
    prepare_ts.create_timeseries()
    # Assert correct cognite object has been called TWICE
    assert cognite_client_mock.time_series.create.call_count == 2
    # Assert no duplicated time series object if already exists
    ts_output_list_new = prepare_ts.client.time_series.list(name=prepare_ts.ts_outputs[0]).to_pandas()
    assert ts_output_list_new == ts_output_list

def test_retrieve_orig_ts(prepare_ts):
    """Test that correct time series signal is retrieved.

    Args:
        prepare_ts (_type_): _description_
    """
    prepare_ts.ts_outputs = ["test_CF"]
    prepare_ts.update_ts("ts_output")
    ts_in = prepare_ts.ts_inputs[0]
    ts_out = prepare_ts.ts_outputs[0]

    # Test that full historic signal is retrieved if output not exists
    prepare_ts.data["ts_output"][prepare_ts.ts_outputs[0]]["exists"] = False
    data = prepare_ts.retrieve_orig_ts(ts_in, ts_out)
    first_date = data.index[0]
    last_date = data.index[-1]
    data_true = prepare_ts.client.time_series.list(name=ts_in).to_pandas()
    data_true = prepare_ts.client.time_series.data.retrieve(external_id=ts_in,
                                                       aggregates="average",
                                                       granularity=f"{prepare_ts.data['granularity']}s",
                                                       limit=-1).to_pandas()
    first_date_true = data_true.index[0]
    last_date_true = data_true.index[-1]
    assert first_date == first_date_true
    assert last_date == last_date_true

    # Test that data only retrieved from current date is output exists
    prepare_ts.data["ts_output"][prepare_ts.ts_outputs[0]]["exists"] = True
    data = prepare_ts.retrieve_orig_ts(ts_in, ts_out)
    first_date = data.index[0]
    last_date = data.index[-1]
    data_true = prepare_ts.client.time_series.data.retrieve(external_id=ts_in,
                                                       aggregates="average",
                                                       granularity=f"{prepare_ts.data['granularity']}s",
                                                       start=prepare_ts.data["start_time"],
                                                       end=prepare_ts.data["end_time"]).to_pandas()

    assert first_date.date == first_date_true.date
    assert last_date == last_date_true

    assert data.squeeze().values == data_true.squeeze().values

def test_get_ts_df(prepare_ts):
    ts_data = prepare_ts.get_ts_df()
    assert all([isinstance(data, float) or isinstance(data, pd.DataFrame) for data in ts_data])

def test_align_time_series(cognite_client_mock, prepare_ts):
    prepare_ts.ts_inputs = ["VAL_17-FI-9101-286:VALUE", "VAL_17-PI-95709-258:VALUE", 56.2] # also testing for scalar
    prepare_ts.ts_outputs = ["test_CF", "test_CF2"]
    prepare_ts.update_ts("ts_inputs")
    prepare_ts.update_ts("ts_output")

    prepare_ts.create_timeseries()

    prepare_ts.data = prepare_ts.get_orig_timeseries(eval(lambda x: x)) # dummy calculation (not applied here anyway)
    ts_df = prepare_ts.get_ts_df()
    ts_df_true = prepare_ts.align_time_series(ts_df)

    # Internal dates may vary (some may have NaNs), but all should be truncated to same boundary dates
    assert all(np.equal([df.index[0] for df in ts_df_true]))
    assert all(np.equal([df.index[-1] for df in ts_df_true]))

    # Assert latest possible start date
    latest_start_date_true = np.max([df.index[0] for df in ts_df])
    latest_start_date_all = [df.index[0] for df in ts_df_true]
    assert all(date == latest_start_date_true for date in latest_start_date_all)
    # Assert earliest possible end date
    earliest_end_date_true = np.min([df.index[-1] for df in ts_df])
    earliest_end_date_all = [df.index[-1] for df in ts_df_true]
    assert all(date == earliest_end_date_true for date in earliest_end_date_all)

    # Assert all return elements are DataFrames
    assert all([isinstance(df, pd.DataFrame) for df in ts_df_true])


def test_deploy_schedule(prepare_ts, deploy_schedule):
    """Test utility function used for testing backfilling

    Args:
        prepare_ts (_type_): _description_
        deploy_schedule (_type_): _description_
    """
    ts_input_name = "VAL_17-FI-9101-286:VALUE"
    df_previous_true = deploy_schedule["df_ts_previous"]
    df_current_true = deploy_schedule["df_ts_current"]
    current_start_date = deploy_schedule["start_time"]
    current_end_date = deploy_schedule["end_time"]

    df_current, df_previous, \
        _, _, _ = prepare_ts.check_backfilling(ts_input_name, testing=True)

    assert df_current[ts_input_name] == df_current_true[ts_input_name]
    assert df_previous[ts_input_name] == df_previous_true[ts_input_name]


def test_backfilling_call_id(prepare_ts, deploy_schedule):
    """Test that the retrieved backfill call is the correct one.

    Args:
        prepare_ts (_type_): _description_
        deploy_schedule (_type_): _description_
    """
    from datetime import timedelta
    # Assert that correct historical scheduled call is extracted at time when doing backfilling
    deployed_time_start = deploy_schedule["deploy_start"]
    all_calls = deploy_schedule["calls"]
    # Get the fifth last scheduled call
    deployed_time_end = deployed_time_start + timedelta(minute=1) # scheduled every 1 min

    mask_start = all_calls["scheduled_time"] >= deployed_time_start
    mask_end = all_calls["scheduled_time"] < deployed_time_end

    # Check that extracted schedule from a given historical time is correct
    initial_call_id = all_calls[mask_start & mask_end]["id"].iloc[0]
    get_func = prepare_ts.client.functions.retrieve(external_id=prepare_ts.data["function_name"])
    get_schedule_id = prepare_ts.client.functions.schedules.list(
                name=prepare_ts.data["function_name"]).to_pandas().id[0]
    get_calls = get_func.list_calls(
                schedule_id=get_schedule_id, limit=-1).to_pandas()
    true_initial_call_id = get_calls["id"].iloc[0] # first call was time of backfilling - this is the one to compare with

    assert initial_call_id == true_initial_call_id


def test_backfilling_unchanged(prepare_ts):
    """Test that no changes have been made to output signal during backfilling
    when no changes in input signal.

    Args:
        prepare_ts (_type_): _description_
    """
    ts_input_name = "VAL_17-FI-9101-286:VALUE.COPY"
    ts_output_name = "test_CF"

    calculation = prepare_ts["calculation_function"]

    _, _, \
        backfill_dates, num_dates_old, num_dates_new = prepare_ts.check_backfilling(ts_input_name, testing=True)

    # Assert that all dates within overlapping period are equal after backfilling
    assert all(num_dates_new.index.isin(num_dates_old.index))
    assert len(backfill_dates) == 0

    df_out_before = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name)
    data = prepare_ts.get_orig_timeseries(eval(calculation))
    df_out_after = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name)

    assert df_out_before == df_out_after

def test_backfilling_insert(prepare_ts):
    """Test that number of datapoints in output signal has increased after backfilling
    when datapoints have been inserted in input signal .


    Args:
        prepare_ts (_type_): _description_
    """
    ts_input_name = "VAL_17-FI-9101-286:VALUE.COPY"
    ts_output_name = "test_CF"

    calculation = prepare_ts["calculation_function"]

    df_out_before = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name).to_pandas()
    # Insert data into copied input
    now = pd.Timestamp.now()
    new_data = [
        (datetime(now.year, now.month, now.day, now.hour-2, now.minute), 111),
        (datetime(now.year, now.month, now.day, now.hour-1, now.minute), 333)
    ]
    prepare_ts.client.time_series.data.insert(new_data, external_id=ts_input_name)

    data = prepare_ts.get_orig_timeseries(eval(calculation))

    df_out_after = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name).to_pandas()

    assert len(df_out_after) > len(df_out_before)

    mask_timestamp1_before = df_out_before.index == new_data[0][0]
    mask_timestamp2_before = df_out_before.index == new_data[1][0]
    mask_timestamp1_after = df_out_after.index == new_data[0][0]
    mask_timestamp2_after = df_out_after.index == new_data[1][0]

    # Assert that new values have been inserted at given dates
    assert len(df_out_after[mask_timestamp1_after]) > 0
    assert len(df_out_after[mask_timestamp2_after]) > 0

    # Assert that datapoints outside these dates are unchanged
    assert len(df_out_after[~mask_timestamp1_after & ~mask_timestamp2_after]) \
            == len(df_out_before[~mask_timestamp1_before & ~mask_timestamp2_before])


def test_backfilling_delete(prepare_ts):
    """Test that number of datapoints in output signal has decreased after backfilling
    when datapoints have been deleted in input signal.


    Args:
        prepare_ts (_type_): _description_
        deploy_schedule (_type_): _description_
    """
    ts_input_name = "VAL_17-FI-9101-286:VALUE.COPY"
    ts_output_name = "test_CF"

    calculation = prepare_ts["calculation_function"]

    df_out_before = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name).to_pandas()
    # Delete data from copied input
    now = pd.Timestamp.now()
    start_delete = datetime(now.year, now.month, now.day, now.hour-2, now.minute)
    end_delete = datetime(now.year, now.month, now.day, now.hour, now.minute)

    prepare_ts.client.time_series.data.delete_range(start=start_delete, end=end_delete, external_id=ts_input_name)

    data = prepare_ts.get_orig_timeseries(eval(calculation))

    df_out_after = prepare_ts.client.time_series.data.retrieve(external_id=ts_output_name).to_pandas()

    start_before = df_out_before.index >= start_delete
    end_before = df_out_before.index < end_delete
    start_after = df_out_after.index >= start_delete # inclusive start date
    end_after = df_out_after.index < end_delete # exclusive end date

    # Assert that num datapoints inside deleted range has decreased
    assert len(df_out_after[start_after & end_after]) < len(df_out_before[start_before & end_before])
    # Assert that num datapoints outsie deleted range remains unchanged
    assert len(df_out_after[~(start_after & end_after)]) == len(df_out_before[~(start_before & end_before)])











